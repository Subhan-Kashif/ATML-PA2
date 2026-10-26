from __future__ import annotations

import argparse
import gc

import pandas as pd
import torch

from common.data import load_yaml, repo_path
from common.models import load_reward_model
from task1_dpo.evaluate import flat_row, run_evaluation
from task1_dpo.train import run_training
from common.logging_utils import load_json


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--n-gen", type=int, default=100)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    n_short = int(cfg["short_ablation_examples"])
    betas = [float(b) for b in cfg["betas"]]

    # 1) train matched forks from the original init (skips a fork if its adapter already exists)
    for b in betas:
        out = f"outputs/task1_dpo/beta_{b:g}"
        if (repo_path(out) / "adapter_config.json").exists():
            print(f"[skip] {out} already trained")
            continue
        run_training(args.config, f"beta_{b:g}", None, out, b, n_short)
        _free()

    # 2) evaluate all forks under one protocol (same prompts, same seed, same decoding)
    rm, rm_tok = load_reward_model(cfg)
    rows = []
    for b in betas:
        res = run_evaluation(cfg, f"outputs/task1_dpo/beta_{b:g}", f"beta_{b:g}", beta=b,
                             rm=rm, rm_tok=rm_tok, n_gen=args.n_gen)
        r = flat_row(res)
        r["budget"] = f"short ({n_short} ex)"
        rows.append(r)

    # 3) include the standard run (different budget) if it was evaluated
    std = repo_path(f"{cfg['results_dir']}/eval_standard.json")
    if std.exists():
        r = flat_row(load_json(std))
        r["budget"] = "full epoch"
        rows.append(r)

    df = pd.DataFrame(rows)
    outp = repo_path(cfg["results_dir"]) / "beta_summary.csv"
    df.to_csv(outp, index=False)
    print(df.to_string())
    print("Saved:", outp)


if __name__ == "__main__":
    main()
