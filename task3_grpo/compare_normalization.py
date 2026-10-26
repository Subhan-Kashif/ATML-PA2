from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.continue_train import collect_group, ensure_fork, fork_stats, per_seq_grad_norms, select_prompts, to_fp32_trainable
from task3_grpo.evaluate import cached_eval
from task3_grpo.grpo import group_relative_advantages, mask_truncated_sequences


def static_probe(cfg, n_prompts, rm, rm_tok, tok):
    """Controlled test: the SAME rollouts and advantages from the midpoint, per-completion gradient norm under both normalizations."""
    pol = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    to_fp32_trainable(pol)
    pol.eval()
    params = trainable_parameters(pol)
    K, Lmax, eps = int(cfg["num_generations"]), int(cfg["max_completion_length"]), float(cfg["clip_epsilon"])
    rows = select_prompts(read_jsonl(cfg["paths"]["rl_prompt_train"]), tok, n_prompts, int(cfg["max_prompt_length"]), int(cfg["seed"]) + 7)
    out = []
    for i, r in enumerate(rows):
        torch.manual_seed(int(cfg["seed"]) * 1000 + 500 + i)
        msgs = [prompt_messages(r)] * K
        g = collect_group(pol, tok, msgs, cfg)
        dev = g["response_ids"].device
        with torch.no_grad():
            old_lp, _ = response_token_logprobs(pol, g["sequences"], g["attention_mask"], g["prompt_width"], g["response_ids"])
            with reference_mode(pol):
                ref_lp, _ = response_token_logprobs(pol, g["sequences"], g["attention_mask"], g["prompt_width"], g["response_ids"])
            rewards = score_reward_pairs(rm, rm_tok, msgs, g["responses"], max_length=int(cfg.get("reward_max_length", 1280))).float().to(dev)
        adv = group_relative_advantages(rewards, torch.zeros(K, dtype=torch.long, device=dev))
        tmask = mask_truncated_sequences(g["response_mask"], g["truncated"])
        for lt in ["grpo", "dr_grpo"]:
            norms = per_seq_grad_norms(pol, params, g, adv, tmask, old_lp, ref_lp, eps, lt, Lmax)
            for k in range(K):
                out.append({"prompt_idx": i, "k": k, "loss_type": lt, "length": int(g["response_lengths"][k]), "adv": float(adv[k]),
                            "reward": float(rewards[k]), "truncated": bool(g["truncated"][k]), "grad_norm": norms[k]})
        print(f"[static probe] prompt {i + 1}/{len(rows)} done")
        del g, old_lp, ref_lp
        torch.cuda.empty_cache()
    del pol
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(out)


def length_stats(df, seed=0):
    """grad_norm / |adv| is the gradient magnitude per unit of advantage. elasticity = slope of log(that) vs log(length):
    about 0 or negative under 1/T normalization, about +1 under constant (Dr.GRPO) normalization."""
    d = df[(df.grad_norm > 0) & (df.adv.abs() > 1e-6)]
    if len(d) < 6:
        return {"n": int(len(d))}
    L = d.length.values.astype(float)
    u = (d.grad_norm / d.adv.abs()).values
    slope = float(np.polyfit(np.log(L), np.log(u), 1)[0])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(1000):
        ix = rng.integers(0, len(d), len(d))
        if len(np.unique(L[ix])) > 1:
            bs.append(np.polyfit(np.log(L[ix]), np.log(u[ix]), 1)[0])
    med = np.median(L)
    long_ = L > med
    gn = d.grad_norm.values
    return {
        "n": int(len(d)), "len_median": float(med),
        "elasticity": slope, "elasticity_lo": float(np.percentile(bs, 2.5)) if bs else float("nan"),
        "elasticity_hi": float(np.percentile(bs, 97.5)) if bs else float("nan"),
        "norm_per_adv_long_over_short": float(u[long_].mean() / u[~long_].mean()) if long_.any() and (~long_).any() else float("nan"),
        "grad_share_long_half": float(gn[long_].sum() / gn.sum()),
        "spearman_len_vs_norm_per_adv": float(pd.Series(L).corr(pd.Series(u), method="spearman")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--static-prompts", type=int, default=8)
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")

    names = {}
    for lt in ["grpo", "dr_grpo"]:
        names[lt] = f"fork_{lt}"
        ensure_fork(args.config, cfg, names[lt], lt, int(cfg["fork_updates"]), probe=True)

    tok = load_tokenizer(cfg["base_model"])
    rm, rm_tok = load_reward_model(cfg)

    if not args.skip_static:
        sp = static_probe(cfg, args.static_prompts, rm, rm_tok, tok)
        sp.to_csv(results / "norm_static_probe.csv", index=False)
        static_tbl = {lt: length_stats(sp[sp.loss_type == lt]) for lt in ["grpo", "dr_grpo"]}
        save_json(results / "norm_static_stats.json", static_tbl)
        print(pd.DataFrame(static_tbl).T.to_string())

    mid = cached_eval(cfg, cfg["paths"]["grpo_midpoint_policy"], "midpoint", rm, rm_tok, args.n_eval)
    keys = ["rm_mean", "kl_token", "entropy", "len_mean", "len_std", "trunc_rate"]
    table = [{"condition": "midpoint (start)", **{k: mid[k] for k in keys}}]
    for lt in ["grpo", "dr_grpo"]:
        ev = cached_eval(cfg, f"outputs/task3_grpo/{names[lt]}", names[lt], rm, rm_tok, args.n_eval)
        row = {"condition": lt, **{k: ev[k] for k in keys}, **fork_stats(cfg, names[lt])}
        sg = repo_path(f"{cfg['results_dir']}/{names[lt]}_seqgrads.jsonl")
        if sg.exists():
            row.update({f"run_{k}": v for k, v in length_stats(pd.DataFrame(read_jsonl(sg))).items()})
        table.append(row)
    df = pd.DataFrame(table)
    df.to_csv(results / "norm_summary.csv", index=False)
    print(df.T.to_string())

    # qualitative candidates: same held-out prompt, biggest length / reward differences between normalizations
    a = read_jsonl(f"{cfg['results_dir']}/{names['grpo']}_eval_generations.jsonl")
    b = read_jsonl(f"{cfg['results_dir']}/{names['dr_grpo']}_eval_generations.jsonl")
    rows = [{"prompt": x["prompt"][:200], "len_grpo": x["tokens"], "len_dr": y["tokens"], "rm_grpo": x["rm_score"], "rm_dr": y["rm_score"],
             "resp_grpo": x["response"][:400], "resp_dr": y["response"][:400]} for x, y in zip(a, b)]
    save_json(results / "norm_examples.json", {
        "dr_much_longer": sorted(rows, key=lambda r: r["len_dr"] - r["len_grpo"])[-3:],
        "dr_much_shorter": sorted(rows, key=lambda r: r["len_dr"] - r["len_grpo"])[:3],
        "biggest_reward_gap": sorted(rows, key=lambda r: -abs(r["rm_dr"] - r["rm_grpo"]))[:3]})


if __name__ == "__main__":
    main()
