from __future__ import annotations

import argparse
import gc

import pandas as pd
import torch

from common.data import last_assistant_text, load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from common.metrics import word_count
from common.models import load_reward_model
from task1_dpo.evaluate import run_evaluation
from task1_dpo.train import run_training


def dataset_length_stats(rows):
    diffs = []
    for r in rows:
        c = word_count(last_assistant_text(r["chosen"]))
        j = word_count(last_assistant_text(r["rejected"]))
        diffs.append(c - j)
    n = len(diffs)
    return {
        "n": n,
        "frac_chosen_longer": sum(d > 0 for d in diffs) / n,
        "mean_word_diff_chosen_minus_rejected": sum(diffs) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--n-gen", type=int, default=100)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])

    # dataset-level length structure (a property of the data, not of the policy)
    std_rows = read_jsonl(cfg["paths"]["dpo_standard_train"])
    bal_rows = read_jsonl(cfg["paths"]["dpo_length_train"])
    ds = {
        "standard_train": dataset_length_stats(std_rows),
        "balanced_train": dataset_length_stats(bal_rows),
        "balanced_strata_counts": pd.Series([r["length_stratum"] for r in bal_rows]).value_counts().to_dict(),
    }
    save_json(results / "dataset_length_stats.json", ds)
    print(ds)

    # train the length-balanced model (same init/config/seed; full pass over the balanced file)
    std_adapter = cfg["standard_output"]
    bal_adapter = cfg["length_output"]
    if not (repo_path(std_adapter) / "adapter_config.json").exists():
        raise FileNotFoundError(f"Train the standard model first: {std_adapter}")
    if not (repo_path(bal_adapter) / "adapter_config.json").exists():
        run_training(args.config, "length_balanced", cfg["paths"]["dpo_length_train"], bal_adapter)
        gc.collect(); torch.cuda.empty_cache()

    # evaluate both under the identical protocol
    rm, rm_tok = load_reward_model(cfg)
    rows = []
    for name, adapter in [("standard", std_adapter), ("length_balanced", bal_adapter)]:
        res = run_evaluation(cfg, adapter, name, rm=rm, rm_tok=rm_tok, length=True, n_gen=args.n_gen)
        row = {
            "model": name,
            "pref_acc_overall": res["pref"]["acc"],
            "gen_len_mean": res["gen"]["len_mean"],
            "gen_len_std": res["gen"]["len_std"],
            "rm_mean": res["gen"].get("rm_mean"),
            "kl_token": res["gen"]["kl_token"],
            "wl_compliance": res["word_limit"]["compliance"],
            "wl_n_parsed": res["word_limit"]["n_parsed"],
            "wl_len_mean": res["word_limit"]["len_mean"],
        }
        for s, v in res["strata"].items():
            row[f"acc_{s}"] = v["acc"]
            row[f"n_{s}"] = v["n"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results / "length_comparison.csv", index=False)
    print(df.T.to_string())
    print("Saved:", results / "length_comparison.csv")


if __name__ == "__main__":
    main()
