from __future__ import annotations

import argparse
import gc

import numpy as np
import torch
import torch.nn.functional as F

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json, set_seed
from common.metrics import word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


@torch.no_grad()
def preference_eval(model, tokenizer, rows, beta, max_length, batch_size=2):
    """Teacher-forced held-out DPO loss / margin / accuracy. Reference = adapter disabled."""
    dev = next(model.parameters()).device
    items, skipped = [], 0
    for row in rows:
        prompt = prompt_messages_from_preference(row)
        yc, yr = preference_responses(row)
        try:
            c = encode_prompt_response(tokenizer, prompt, yc, max_length)
            r = encode_prompt_response(tokenizer, prompt, yr, max_length)
        except ValueError:
            skipped += 1
            continue
        items.append((row, c, r))

    model.eval()
    out = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        bc = {k: v.to(dev) for k, v in pad_batch(tokenizer, [x[1] for x in chunk]).items()}
        br = {k: v.to(dev) for k, v in pad_batch(tokenizer, [x[2] for x in chunk]).items()}
        pc = response_sequence_logprobs(model, bc)[0]
        pr = response_sequence_logprobs(model, br)[0]
        with reference_mode(model):
            rc = response_sequence_logprobs(model, bc)[0]
            rr = response_sequence_logprobs(model, br)[0]
        margin = (pc - pr) - (rc - rr)
        loss = -F.logsigmoid(beta * margin)
        for j, (row, _, _) in enumerate(chunk):
            out.append({
                "prompt_id": row.get("prompt_id"),
                "stratum": row.get("length_stratum"),
                "margin": float(margin[j]),
                "loss": float(loss[j]),
                "correct": float(margin[j] > 0),
            })
    return out, skipped


def summarize_pref(recs, skipped):
    return {
        "n": len(recs),
        "skipped_long_prompt": skipped,
        "loss": float(np.mean([r["loss"] for r in recs])),
        "acc": float(np.mean([r["correct"] for r in recs])),
        "margin_mean": float(np.mean([r["margin"] for r in recs])),
    }


def summarize_strata(recs):
    out = {}
    for s in sorted({r["stratum"] for r in recs if r["stratum"] is not None}):
        sub = [r for r in recs if r["stratum"] == s]
        out[s] = {
            "n": len(sub),
            "acc": float(np.mean([r["correct"] for r in sub])),
            "margin_mean": float(np.mean([r["margin"] for r in sub])),
            "loss": float(np.mean([r["loss"] for r in sub])),
        }
    return out


def generation_eval(model, tokenizer, rm, rm_tok, prompts, cfg, max_new_tokens,
                    batch_size=4, greedy=False, with_kl=True):
    """Generate, then per-response: RM score, length, sampled-KL (policy logp - ref logp)."""
    g_cfg = cfg["generation"]
    recs = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        g = batch_generate(
            model, tokenizer, chunk,
            max_prompt_length=256,
            max_new_tokens=max_new_tokens,
            temperature=float(g_cfg["temperature"]),
            top_p=float(g_cfg["top_p"]),
            do_sample=bool(g_cfg["do_sample"]) and not greedy,
        )
        kl_sum = [0.0] * len(chunk)
        if with_kl:
            with torch.no_grad():
                pol, _ = response_token_logprobs(
                    model, g["sequences"], g["attention_mask"], g["prompt_width"], g["response_ids"])
                with reference_mode(model):
                    ref, _ = response_token_logprobs(
                        model, g["sequences"], g["attention_mask"], g["prompt_width"], g["response_ids"])
                kl_sum = ((pol - ref) * g["response_mask"]).sum(-1).tolist()
        scores = score_reward_pairs(rm, rm_tok, chunk, g["responses"]).tolist() if rm is not None else [None] * len(chunk)
        for i in range(len(chunk)):
            recs.append({
                "prompt": chunk[i][-1]["content"],
                "response": g["responses"][i],
                "tokens": int(g["response_lengths"][i]),
                "kl_sum": float(kl_sum[i]),
                "rm_score": None if scores[i] is None else float(scores[i]),
                "truncated": bool(g["truncated"][i]),
            })
        del g
        torch.cuda.empty_cache()
    return recs


def summarize_gen(recs):
    toks = np.array([r["tokens"] for r in recs], dtype=float)
    kl = np.array([r["kl_sum"] for r in recs], dtype=float)
    out = {
        "n": len(recs),
        "len_mean": float(toks.mean()),
        "len_std": float(toks.std()),
        "len_median": float(np.median(toks)),
        "trunc_rate": float(np.mean([r["truncated"] for r in recs])),
        # token-mean convention (primary): total sampled log-ratio / total response tokens
        "kl_token": float(kl.sum() / max(toks.sum(), 1.0)),
        # sequence-mean convention: mean over responses of the summed log-ratio
        "kl_seq": float(kl.mean()),
    }
    sc = [r["rm_score"] for r in recs if r["rm_score"] is not None]
    if sc:
        out["rm_mean"] = float(np.mean(sc))
        out["rm_std"] = float(np.std(sc))
    return out


def run_evaluation(cfg, adapter, name, beta=None, rm=None, rm_tok=None,
                   length=False, n_gen=100, n_pref=None):
    beta = float(cfg["beta"] if beta is None else beta)
    results_dir = cfg["results_dir"]
    tok = load_tokenizer(cfg["base_model"])
    if rm is None:
        rm, rm_tok = load_reward_model(cfg)
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    max_len = int(cfg["max_sequence_length"])
    max_new = int(cfg["max_generation_tokens"])

    # 1) held-out preference metrics (standard eval pairs)
    rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    if n_pref:
        rows = rows[: int(n_pref)]
    prec, skipped = preference_eval(model, tok, rows, beta, max_len)
    res = {"name": name, "adapter": adapter, "beta": beta, "pref": summarize_pref(prec, skipped)}
    write_jsonl(f"{results_dir}/{name}_pref_rows.jsonl", prec)

    # 2) generation metrics on the first n_gen held-out prompts (same seed/settings for every condition)
    all_prompts = [prompt_messages_from_preference(r) for r in rows]
    prompts = [
        q for q in all_prompts
        if len(tok.apply_chat_template(q, tokenize=True, add_generation_prompt=True)) <= 256
    ][: int(n_gen)]
    set_seed(int(cfg["seed"]))
    grecs = generation_eval(model, tok, rm, rm_tok, prompts, cfg, max_new)
    res["gen"] = summarize_gen(grecs)
    write_jsonl(f"{results_dir}/{name}_generations.jsonl", grecs)

    # 3) length-confounding diagnostics (stratified pairs + word-limit prompts)
    if length:
        srows = read_jsonl(cfg["paths"]["dpo_length_eval"])
        if n_pref:
            srows = srows[: int(n_pref)]
        srec, sskip = preference_eval(model, tok, srows, beta, max_len)
        res["strata"] = summarize_strata(srec)
        res["strata_overall"] = summarize_pref(srec, sskip)

        wl = read_jsonl(cfg["paths"]["word_limit_prompts"])
        wl_prompts = [r["messages"] for r in wl]
        wrecs = generation_eval(model, tok, None, None, wl_prompts, cfg, max_new, greedy=True, with_kl=False)
        comp = []
        for r in wrecs:
            c = word_limit_compliance(r["prompt"], r["response"])
            r["compliant"] = c
            if c is not None:
                comp.append(c)
        res["word_limit"] = {
            "n_prompts": len(wrecs),
            "n_parsed": len(comp),
            "compliance": float(np.mean(comp)) if comp else None,
            "len_mean": float(np.mean([r["tokens"] for r in wrecs])),
            "len_std": float(np.std([r["tokens"] for r in wrecs])),
        }
        write_jsonl(f"{results_dir}/{name}_word_limit.jsonl", wrecs)

    save_json(f"{results_dir}/eval_{name}.json", res)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def flat_row(res):
    row = {"name": res["name"], "beta": res["beta"]}
    for sec in ("pref", "gen"):
        for k, v in res[sec].items():
            row[f"{sec}_{k}"] = v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--length", action="store_true", help="also run stratified + word-limit eval")
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--n-pref", type=int)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = run_evaluation(cfg, None if args.adapter == "base" else args.adapter, args.name, args.beta, length=args.length,
                         n_gen=args.n_gen, n_pref=args.n_pref)
    print(res)


if __name__ == "__main__":
    main()
