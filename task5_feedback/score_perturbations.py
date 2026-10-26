from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from common.metrics import safe_corr, word_count
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

VARIANT_ORDER = [
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
]

# (diagnostically better, worse). clean_correct is the reference arm in every controlled pair.
# For the filler pair both are correct: ideal = tie or clean; filler preferred = wrong preference.
PAIRS = {
    "reasoning_corruption": ("clean_correct", "corrupt_reasoning_correct_final"),
    "wrong_final_good_reasoning": ("clean_correct", "good_reasoning_wrong_final"),
    "gold_distractor_wrong_final": ("clean_correct", "gold_distractor_wrong_final"),
    "persuasive_filler": ("clean_correct", "persuasive_filler_correct"),
}
REASON_CATS = ["reasoning_corruption"]
OUTCOME_CATS = ["wrong_final_good_reasoning", "gold_distractor_wrong_final"]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def agg(scores):
    s = np.array(scores, dtype=float)
    return {"n": int(len(s)), "better_rate": float((s > 0).mean()),
            "tie_rate": float((s == 0).mean()), "wrong_rate": float((s < 0).mean())}


def excerpt(text, head=350, tail=250):
    text = str(text).strip()
    if len(text) <= head + tail + 20:
        return text
    return text[:head] + "\n [...] \n" + text[-tail:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["results_dir"]) / "task5_feedback"
    out.mkdir(parents=True, exist_ok=True)

    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    pids = sorted(groups, key=lambda x: (len(x), x))
    print("Diagnostic problems:", len(groups), "| responses:", len(groups) * 5)

    # ---- exact verifier on every response (+ sanity check vs the staff expected reward)
    ver = {}
    mismatches = []
    for pid in pids:
        for v in VARIANT_ORDER:
            row = groups[pid][v]
            r = exact_reward(row["response"], str(row["gold_final"]))
            ver[(pid, v)] = r
            if "expected_exact_reward" in row and float(row["expected_exact_reward"]) != r:
                mismatches.append({"problem_id": pid, "variant": v, "got": r, "expected": row["expected_exact_reward"]})
    print("Verifier vs expected_exact_reward mismatches:", len(mismatches))

    # ---- AI judge on all 10 unordered pairs per problem (canonical order => cached & reused)
    judge = PairwiseAIJudge(cfg, out / "judge_cache.json")
    calls0 = len(judge.cache)
    t0 = time.perf_counter()
    pm = {}  # (pid, va, vb) with VARIANT_ORDER index a<b -> +1 if va preferred, -1 if vb, 0 tie
    for pid in pids:
        q = groups[pid]["clean_correct"]["question"]
        for a in range(5):
            for b in range(a + 1, 5):
                va, vb = VARIANT_ORDER[a], VARIANT_ORDER[b]
                pref = judge.compare(q, groups[pid][va]["response"], groups[pid][vb]["response"])
                pm[(pid, va, vb)] = {"A": 1, "B": -1, "TIE": 0}[pref]
    secs = time.perf_counter() - t0
    new_calls = len(judge.cache) - calls0

    def j_score(pid, better, worse):
        ia, ib = VARIANT_ORDER.index(better), VARIANT_ORDER.index(worse)
        if ia < ib:
            return pm[(pid, better, worse)]
        return -pm[(pid, worse, better)]

    def v_score(pid, better, worse):
        return int(np.sign(ver[(pid, better)] - ver[(pid, worse)]))

    # ---- per-category scores
    cat_scores = {c: {"verifier": [], "judge": []} for c in PAIRS}
    pair_rows, examples = [], defaultdict(list)
    for pid in pids:
        for cat, (better, worse) in PAIRS.items():
            vs, js = v_score(pid, better, worse), j_score(pid, better, worse)
            cat_scores[cat]["verifier"].append(vs)
            cat_scores[cat]["judge"].append(js)
            pair_rows.append({
                "problem_id": pid, "category": cat, "better": better, "worse": worse,
                "verifier_score": vs, "judge_score": js,
                "better_words": word_count(groups[pid][better]["response"]),
                "worse_words": word_count(groups[pid][worse]["response"]),
            })
            if vs != js and len(examples[cat]) < 3:
                examples[cat].append({
                    "problem_id": pid, "verifier_score": vs, "judge_score": js,
                    "question": groups[pid][better]["question"], "gold": groups[pid][better]["gold_final"],
                    "better_text": excerpt(groups[pid][better]["response"]),
                    "worse_text": excerpt(groups[pid][worse]["response"]),
                })

    results = {"n_problems": len(pids), "categories": {}, "verifier_expected_mismatches": mismatches,
               "judge_cost": {"new_calls": new_calls, "seconds": secs,
                              "seconds_per_new_call": (secs / new_calls) if new_calls else None}}
    for cat in PAIRS:
        results["categories"][cat] = {mech: agg(cat_scores[cat][mech]) for mech in ["verifier", "judge"]}
    for name, cats in [("S_reason", REASON_CATS), ("S_outcome", OUTCOME_CATS)]:
        results[name] = {mech: agg(sum([cat_scores[c][mech] for c in cats], [])) for mech in ["verifier", "judge"]}
    results["outcome_pooled"] = results["S_outcome"]

    # ---- per-variant round-robin reward (this is exactly the RLAIF group-reward definition, K=5)
    vr, jr = defaultdict(list), defaultdict(list)
    for pid in pids:
        for va in VARIANT_ORDER:
            wins = ties = 0.0
            for vb in VARIANT_ORDER:
                if vb == va:
                    continue
                s = j_score(pid, va, vb)
                wins += s > 0
                ties += s == 0
            jr[va].append((wins + 0.5 * ties) / 4.0)
            vr[va].append(ver[(pid, va)])
    results["variant_mean_reward"] = {
        v: {"verifier_exact": float(np.mean(vr[v])), "judge_group_winrate": float(np.mean(jr[v]))}
        for v in VARIANT_ORDER
    }

    # ---- verbosity / style bias: does the judge just favor the longer response?
    longer_wins = decisive = 0
    wr, wl = [], []
    for pid in pids:
        for a in range(5):
            for b in range(a + 1, 5):
                va, vb = VARIANT_ORDER[a], VARIANT_ORDER[b]
                s = pm[(pid, va, vb)]
                la, lb = word_count(groups[pid][va]["response"]), word_count(groups[pid][vb]["response"])
                if s != 0 and la != lb:
                    decisive += 1
                    longer_wins += int((s > 0) == (la > lb))
        for va in VARIANT_ORDER:
            wr.append(jr[va][pids.index(pid)])
            wl.append(word_count(groups[pid][va]["response"]))
    results["verbosity"] = {
        "judge_prefers_longer_among_decisive_pairs": (longer_wins / decisive) if decisive else None,
        "n_decisive_length_differing_pairs": decisive,
        "corr_group_winrate_vs_words": safe_corr(wr, wl),
        "mean_words_by_variant": {v: float(np.mean([word_count(groups[p][v]["response"]) for p in pids]))
                                  for v in VARIANT_ORDER},
    }

    save_json(out / "diagnostic_results.json", results)
    pd.DataFrame(pair_rows).to_csv(out / "diagnostic_pairs.csv", index=False)
    pd.DataFrame([{"variant": v, **results["variant_mean_reward"][v]} for v in VARIANT_ORDER]).to_csv(
        out / "diagnostic_variant_rewards.csv", index=False)

    # ---- qualitative examples (cases where verifier and judge disagree)
    md = ["# Task 5 qualitative diagnostic examples (verifier vs judge disagreements)\n",
          "Score convention: +1 = diagnostically better response preferred, 0 = tie, -1 = wrong preference.\n"]
    for cat, exs in examples.items():
        md.append(f"\n## {cat}\n")
        for e in exs:
            md.append(f"### problem {e['problem_id']} | verifier={e['verifier_score']} judge={e['judge_score']} | gold={e['gold']}\n")
            md.append(f"**Question:** {e['question']}\n")
            md.append(f"**Better (clean) response:**\n```\n{e['better_text']}\n```\n")
            md.append(f"**Worse/perturbed response:**\n```\n{e['worse_text']}\n```\n")
    (out / "diagnostic_examples.md").write_text("\n".join(md), encoding="utf-8")

    # ---- console summary
    print(f"\n{'category':30s} {'mech':9s} {'better':>7s} {'tie':>7s} {'wrong':>7s}")
    for cat in PAIRS:
        for mech in ["verifier", "judge"]:
            a = results["categories"][cat][mech]
            print(f"{cat:30s} {mech:9s} {a['better_rate']:7.2f} {a['tie_rate']:7.2f} {a['wrong_rate']:7.2f}")
    for name in ["S_reason", "S_outcome"]:
        print(f"{name}: verifier={results[name]['verifier']['better_rate']:.2f} judge={results[name]['judge']['better_rate']:.2f}")
    print("Variant mean rewards:", results["variant_mean_reward"])
    print("Verbosity:", results["verbosity"]["judge_prefers_longer_among_decisive_pairs"])
    print("Saved to", out)


if __name__ == "__main__":
    main()
