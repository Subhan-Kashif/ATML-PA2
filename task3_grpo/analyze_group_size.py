from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

TOL = 1e-6
BIN_NAMES = ["hard", "medium", "easy"]


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, rng=None):
    """Partition each prompt's 8 cached completions into 8/k groups of size k.

    Total generations stay fixed (n_prompts * 8) and the prompts stay the same, so K=2 gives 4 groups per
    prompt, K=4 gives 2 and K=8 gives 1. rng=None uses consecutive generation_index chunks; with an rng the
    8 completions are shuffled first, so averaging over many shuffles removes dependence on one arbitrary split.
    """
    groups = []
    for pid, rows in by_prompt.items():
        rows = rows[:8]
        assert len(rows) % k == 0, f"K={k} must divide {len(rows)}"
        order = np.arange(len(rows)) if rng is None else rng.permutation(len(rows))
        for gi in range(len(rows) // k):
            idx = order[gi * k:(gi + 1) * k]
            groups.append({"pid": pid, "gi": gi, "idx": [int(i) for i in idx],
                           "rewards": np.array([float(rows[i]["reward"]) for i in idx])})
    return groups


def difficulty_bins(by_prompt, n_bins=3):
    """Rule (defined once): prompt difficulty = mean cached reward over its 8 completions (lower = harder).
    Prompts are ranked by it (ties broken by id) and split into 3 equal-sized bins: hard / medium / easy."""
    pids = sorted(by_prompt, key=lambda p: (np.mean([float(r["reward"]) for r in by_prompt[p][:8]]), p))
    parts = np.array_split(np.array(pids, dtype=object), n_bins)
    return {pid: BIN_NAMES[i] for i, part in enumerate(parts) for pid in part}


def group_metrics(groups, thr):
    s = np.array([g["rewards"].std() for g in groups])                       # population std (unbiased=False)
    cs = np.concatenate([(g["rewards"] - g["rewards"].mean()) ** 2 for g in groups])
    sv = np.array([g["rewards"].var(ddof=1) for g in groups])
    inf = s > TOL
    return {
        "n_groups": len(groups),
        "informative_rate": float(inf.mean()),                              # std > 1e-6
        "practical_rate": float((s > thr).mean()),                          # std > 0.25 * global reward std
        "mean_group_std": float(s.mean()),
        "mean_std_informative": float(s[inf].mean()) if inf.any() else 0.0,
        "relative_signal_var": float(cs.mean()),                            # mean of (r_k - mu_group)^2: raw centered signal
        "relative_signal_var_unbiased": float(sv.mean()),                   # same with ddof=1 (removes the (K-1)/K effect)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    ks = [int(k) for k in cfg["group_sizes"]]
    all_r = np.array([float(r["reward"]) for rows in by_prompt.values() for r in rows[:8]])
    thr = 0.25 * float(all_r.std())
    print(f"reward: mean {all_r.mean():.3f} std {all_r.std():.3f} | distinct values {len(np.unique(all_r))} | practical threshold {thr:.3f}")
    bins = difficulty_bins(by_prompt)
    rng = np.random.default_rng(int(cfg["seed"]))
    metrics = ["informative_rate", "practical_rate", "mean_group_std", "mean_std_informative", "relative_signal_var", "relative_signal_var_unbiased"]

    records = []
    for k in ks:
        partitions = [regroup_equal_generation_budget(by_prompt, k)] + [regroup_equal_generation_budget(by_prompt, k, rng) for _ in range(args.n_perm)]
        for b in ["all"] + BIN_NAMES:
            ms = []
            for groups in partitions:
                sub = groups if b == "all" else [g for g in groups if bins[g["pid"]] == b]
                if sub:
                    ms.append(group_metrics(sub, thr))
            if not ms:
                continue
            df = pd.DataFrame(ms)
            rec = {"K": k, "bin": b, "n_groups": int(df.n_groups.iloc[0]), "n_generations": int(df.n_groups.iloc[0]) * k}
            for c in metrics:
                rec[c] = float(df[c].mean())
                rec[c + "_lo"], rec[c + "_hi"] = [float(x) for x in np.percentile(df[c], [2.5, 97.5])]
            records.append(rec)

    out = pd.DataFrame(records)
    results = repo_path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    out.to_csv(results / "group_size_summary.csv", index=False)
    show = ["K", "bin", "n_groups", "n_generations"] + metrics
    print(out[show].round(4).to_string(index=False))

    # examples: hardest and easiest prompt, rewards of all 8 completions + what K=2 (consecutive) would see
    ex = {}
    ranked = sorted(by_prompt, key=lambda p: np.mean([float(r["reward"]) for r in by_prompt[p][:8]]))
    for tag, pid in [("hardest", ranked[0]), ("easiest", ranked[-1]), ("median", ranked[len(ranked) // 2])]:
        rows = by_prompt[pid][:8]
        ex[tag] = {"source_index": pid, "rewards": [float(r["reward"]) for r in rows],
                   "k2_group_stds": [float(np.std([float(rows[i]["reward"]), float(rows[i + 1]["reward"])])) for i in range(0, 8, 2)],
                   "completions": [{"reward": float(r["reward"]), "text": str(r.get("completion", ""))[:250]} for r in sorted(rows, key=lambda r: float(r["reward"]))[::7]]}
    save_json(results / "group_examples.json", ex)
    save_json(results / "group_size_config.json", {"tolerance": TOL, "practical_threshold": thr, "n_perm": args.n_perm,
                                                    "bin_rule": "rank prompts by mean cached reward over 8 completions, split into terciles"})


if __name__ == "__main__":
    main()
