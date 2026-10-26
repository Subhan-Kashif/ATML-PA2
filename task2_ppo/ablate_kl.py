from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_reward_model
from task2_ppo.continue_train import ensure_fork, fork_stats
from task2_ppo.evaluate import cached_eval


def pick_examples(cfg, base, other, k=3):
    a = read_jsonl(f"{cfg['results_dir']}/{base}_eval_generations.jsonl")
    b = read_jsonl(f"{cfg['results_dir']}/{other}_eval_generations.jsonl")
    rows = []
    for x, y in zip(a, b):
        rows.append({"prompt": x["prompt"][:300], "rm_base": x["rm_score"], "rm_other": y["rm_score"],
                     "d_rm": y["rm_score"] - x["rm_score"], "len_base": x["tokens"], "len_other": y["tokens"],
                     "resp_base": x["response"][:500], "resp_other": y["response"][:500]})
    by_gain = sorted(rows, key=lambda r: -r["d_rm"])[:k]
    # reward up AND much longer: candidates for "reward goes up, quality does not"
    longer = sorted([r for r in rows if r["d_rm"] > 0 and r["len_other"] > 1.3 * max(r["len_base"], 1)], key=lambda r: -r["d_rm"])[:k]
    return {"top_reward_gain": by_gain, "reward_up_and_longer": longer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--n-eval", type=int, default=40)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    eps0 = float(cfg["clip_epsilon"])
    betas = [float(b) for b in cfg["kl_values"]]
    print("KL beta conditions:", betas, "| fork update budget:", cfg["fork_updates"])

    names = {}
    for b in betas:
        names[b] = f"fork_eps{eps0:g}_kl{b:g}"        # shares the run with the clipping study when beta == 0.10
        ensure_fork(args.config, cfg, names[b], eps0, b)

    rm, rm_tok = load_reward_model(cfg)
    mid = cached_eval(cfg, cfg["paths"]["ppo_midpoint_policy"], "midpoint", rm, rm_tok, args.n_eval)
    table, traj = [{"beta_kl": "midpoint (start)", **{k: mid[k] for k in ["rm_mean", "kl_token", "entropy", "len_mean", "len_std"]}}], []
    examples = {}
    for b in betas:
        ev = cached_eval(cfg, f"outputs/task2_ppo/{names[b]}", names[b], rm, rm_tok, args.n_eval)
        row = {"beta_kl": b, **{k: ev[k] for k in ["rm_mean", "kl_token", "entropy", "len_mean", "len_std"]}, **fork_stats(cfg, names[b])}
        table.append(row)
        for r in read_jsonl(f"{cfg['results_dir']}/{names[b]}_log.jsonl"):
            traj.append({"beta_kl": b, "update": r["update"], "reward": r["reward_raw"], "kl_token": r["kl_token"],
                         "entropy": r["entropy"], "length": r["response_length"]})
        examples[str(b)] = pick_examples(cfg, "midpoint", names[b])

    pd.DataFrame(table).to_csv(results / "kl_summary.csv", index=False)
    pd.DataFrame(traj).to_csv(results / "kl_trajectories.csv", index=False)
    save_json(results / "kl_examples.json", examples)
    print(pd.DataFrame(table).T.to_string())


if __name__ == "__main__":
    main()
