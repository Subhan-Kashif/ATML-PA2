from __future__ import annotations

import argparse
import gc
import time
from collections import Counter

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.metrics import safe_corr, word_count
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def out_dir(cfg):
    d = repo_path(cfg["results_dir"]) / "task5_feedback"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean_num(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else x


def failure_type(response, gold, truncated):
    if exact_reward(response, gold) == 1.0:
        return "correct"
    if extract_designated_final(response) is None:
        return "truncated_no_answer" if truncated else "missing_format"
    return "wrong_answer"


def free_gpu(*_):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- generation
def generate_policy(cfg, rows, name, tokenizer, dataset, batch_size, max_new_tokens, force):
    path = out_dir(cfg) / f"gen_{dataset}_{name}.jsonl"
    if path.exists() and not force:
        cached = read_jsonl(path)
        if len(cached) == len(rows):
            print(f"[{name}] using cached generations: {path}")
            return cached

    print(f"[{name}] generating {len(rows)} responses (greedy)...")
    model = load_frozen_policy(cfg, name)
    recs = []
    t0 = time.perf_counter()
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        gen = batch_generate(
            model, tokenizer, [r["messages"] for r in chunk],
            max_prompt_length=384, max_new_tokens=max_new_tokens, do_sample=False,
        )
        for r, resp, n_tok, trunc, term in zip(
            chunk, gen["responses"], gen["response_lengths"], gen["truncated"], gen["terminated_with_eos"]
        ):
            gold = str(r["gold_final"])
            pred = extract_designated_final(resp)
            recs.append({
                "source_index": str(r["source_index"]),
                "policy": name,
                "question": r["question"],
                "gold_final": gold,
                "response": resp,
                "pred_final": pred,
                "exact": exact_reward(resp, gold),
                "format_ok": pred is not None,
                "response_tokens": int(n_tok),
                "response_words": word_count(resp),
                "truncated": bool(trunc),
                "terminated_with_eos": bool(term),
                "failure_type": failure_type(resp, gold, bool(trunc)),
            })
        print(f"  [{name}] {min(s + batch_size, len(rows))}/{len(rows)}")
    secs = time.perf_counter() - t0
    for rec in recs:
        rec["gen_seconds_total"] = secs
    del model
    free_gpu()
    write_jsonl(path, recs)
    return recs


def policy_metrics(recs):
    n = len(recs)
    ft = Counter(r["failure_type"] for r in recs)
    return {
        "n": n,
        "exact_accuracy": float(np.mean([r["exact"] for r in recs])),
        "format_compliance": float(np.mean([r["format_ok"] for r in recs])),
        "mean_response_tokens": float(np.mean([r["response_tokens"] for r in recs])),
        "mean_response_words": float(np.mean([r["response_words"] for r in recs])),
        "truncation_rate": float(np.mean([r["truncated"] for r in recs])),
        "failure_types": {k: v / n for k, v in ft.items()},
        "gen_seconds": float(recs[0]["gen_seconds_total"]),
    }


# ---------------------------------------------------------------- judging
def judge_vs_sft(judge, rows, pol_recs, sft_recs):
    """Judge the policy (as candidate A) against SFT (as B). Orientation balancing is inside the judge."""
    out = []
    for r, p, s in zip(rows, pol_recs, sft_recs):
        pref = judge.compare(r["question"], p["response"], s["response"])  # "A" => policy wins
        j = {"A": 1, "B": -1, "TIE": 0}[pref]
        v = int(np.sign(p["exact"] - s["exact"]))
        out.append({"judge_pref": pref, "j": j, "v": v})
    return out


def pair_metrics(judged):
    n = len(judged)
    j = np.array([x["j"] for x in judged])
    v = np.array([x["v"] for x in judged])
    wins, ties, losses = int((j > 0).sum()), int((j == 0).sum()), int((j < 0).sum())
    dec = v != 0
    return {
        "n": n,
        "judge_wins": wins, "judge_ties": ties, "judge_losses": losses,
        "win_rate_strict": wins / n,
        "win_rate_ties_half": (wins + 0.5 * ties) / n,
        "agreement_all": float((v == j).mean()),
        "agreement_when_verifier_decisive": clean_num(float((v[dec] == j[dec]).mean()) if dec.any() else None),
        "judge_decisive_when_verifier_tie": clean_num(float((j[~dec] != 0).mean()) if (~dec).any() else None),
        "judge_contradicts_verifier": clean_num(float((v[dec] * j[dec] < 0).mean()) if dec.any() else None),
        "verifier_decisive_fraction": float(dec.mean()),
        "corr_judge_vs_verifier_delta": clean_num(safe_corr(j, v)),
    }


def fmt(x, pct=False):
    if x is None:
        return "NA"
    return f"{100 * x:.1f}%" if pct else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--force", action="store_true", help="regenerate even if cached")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    set_seed(int(cfg["seed"]))
    if args.max_examples:
        rows = rows[: args.max_examples]
    max_new = args.max_new_tokens or int(cfg["math_max_new_tokens"])
    print("Rows:", len(rows), "| dataset:", args.dataset, "| max_new_tokens:", max_new)

    # 1) matched greedy generations; each policy loaded then freed (VRAM)
    gens = {}
    for name in policy_specs(cfg):
        gens[name] = generate_policy(cfg, rows, name, tokenizer, args.dataset, args.batch_size, max_new, args.force)

    result = {
        "dataset": args.dataset, "n": len(rows), "max_new_tokens": max_new,
        "policies": {k: policy_metrics(v) for k, v in gens.items()},
        "vs_sft": {}, "judge": {},
    }

    # 2) fixed pairwise AI judge: RLVR vs SFT and RLAIF vs SFT
    judged = {}
    if not args.skip_judge:
        judge = PairwiseAIJudge(cfg, out_dir(cfg) / "judge_cache.json")
        calls0 = len(judge.cache)
        t0 = time.perf_counter()
        for name in ["rlvr", "rlaif"]:
            judged[name] = judge_vs_sft(judge, rows, gens[name], gens["sft"])
            result["vs_sft"][name] = pair_metrics(judged[name])
        secs = time.perf_counter() - t0
        new_calls = len(judge.cache) - calls0
        result["judge"] = {"new_calls": new_calls, "seconds": secs,
                           "seconds_per_new_call": (secs / new_calls) if new_calls else None}

    # 3) per-example merged file
    per = []
    for i, r in enumerate(rows):
        rec = {"source_index": str(r["source_index"]), "question": r["question"], "gold_final": str(r["gold_final"])}
        for name in gens:
            g = gens[name][i]
            rec[name] = {k: g[k] for k in ["response", "pred_final", "exact", "format_ok", "response_tokens",
                                           "response_words", "truncated", "failure_type"]}
        for name, j in judged.items():
            rec[f"{name}_vs_sft"] = j[i]
        per.append(rec)
    write_jsonl(out_dir(cfg) / f"{args.dataset}_per_example.jsonl", per)
    save_json(out_dir(cfg) / f"{args.dataset}_metrics.json", result)

    # 4) console table
    print(f"\n=== {args.dataset.upper()} (n={len(rows)}) ===")
    print(f"{'policy':6s} {'acc':>7s} {'format':>7s} {'tokens':>8s} {'words':>7s} {'trunc':>7s} {'win/SFT':>8s} {'agree':>7s}")
    for name, m in result["policies"].items():
        w = result["vs_sft"].get(name)
        print(f"{name:6s} {fmt(m['exact_accuracy'], True):>7s} {fmt(m['format_compliance'], True):>7s} "
              f"{m['mean_response_tokens']:8.1f} {m['mean_response_words']:7.1f} {fmt(m['truncation_rate'], True):>7s} "
              f"{fmt(w['win_rate_ties_half'], True) if w else '-':>8s} "
              f"{fmt(w['agreement_all'], True) if w else '-':>7s}")
    print("Saved to", out_dir(cfg))


if __name__ == "__main__":
    main()
