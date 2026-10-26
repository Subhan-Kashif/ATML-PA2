from __future__ import annotations

import argparse
import gc

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task3_grpo.continue_train import entropy_sum, select_prompts


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_adapter(cfg, adapter, name, rm=None, rm_tok=None, n=100, batch_size=4):
    # common protocol: same held-out prompts, seed, decoding and cap for every condition
    tok = load_tokenizer(cfg["base_model"])
    if rm is None:
        rm, rm_tok = load_reward_model(cfg)
    pol = load_policy(cfg, adapter_path=adapter, trainable=False)
    rows = select_prompts(read_jsonl(cfg["paths"]["rl_prompt_eval"]), tok, n, int(cfg["max_prompt_length"]), int(cfg["seed"]))
    max_new = int(cfg.get("eval_max_completion_length", cfg["max_completion_length"]))
    gen = cfg["generation"]
    set_seed(int(cfg["seed"]))

    recs = []
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        msgs = [prompt_messages(r) for r in chunk]
        pol.config.use_cache = True
        g = batch_generate(pol, tok, msgs, int(cfg["max_prompt_length"]), max_new,
                           temperature=float(gen["temperature"]), top_p=float(gen["top_p"]),
                           do_sample=bool(gen["do_sample"]))
        mask = g["response_mask"]
        scores = score_reward_pairs(rm, rm_tok, msgs, g["responses"], max_length=int(cfg.get("reward_max_length", 1280))).tolist()
        with torch.no_grad():
            for i in range(len(chunk)):
                sl = slice(i, i + 1)
                a = (g["sequences"][sl], g["attention_mask"][sl], g["prompt_width"], g["response_ids"][sl])
                lp, lg = response_token_logprobs(pol, *a)
                es, en = entropy_sum(lg, mask[sl])
                del lg
                with reference_mode(pol):
                    rlp, _ = response_token_logprobs(pol, *a)
                recs.append({
                    "prompt_id": chunk[i].get("prompt_id", chunk[i].get("source_index")),
                    "prompt": msgs[i][-1]["content"], "response": g["responses"][i],
                    "tokens": int(g["response_lengths"][i]), "rm_score": float(scores[i]),
                    "kl_sum": float(((lp - rlp) * mask[sl]).sum()), "ent_sum": es,
                    "truncated": bool(g["truncated"][i]), "eos": bool(g["terminated_with_eos"][i]),
                })
        del g
        torch.cuda.empty_cache()

    toks = np.array([r["tokens"] for r in recs], dtype=float)
    kl = np.array([r["kl_sum"] for r in recs])
    sc = np.array([r["rm_score"] for r in recs])
    res = {
        "name": name, "adapter": adapter, "n": len(recs), "max_new_tokens": max_new,
        "rm_mean": float(sc.mean()), "rm_std": float(sc.std()),
        "kl_token": float(kl.sum() / toks.sum()), "kl_seq": float(kl.mean()),
        "entropy": float(sum(r["ent_sum"] for r in recs) / toks.sum()),
        "len_mean": float(toks.mean()), "len_std": float(toks.std()), "len_median": float(np.median(toks)),
        "trunc_rate": float(np.mean([r["truncated"] for r in recs])), "eos_rate": float(np.mean([r["eos"] for r in recs])),
    }
    write_jsonl(f"{cfg['results_dir']}/{name}_eval_generations.jsonl", recs)
    save_json(f"{cfg['results_dir']}/eval_{name}.json", res)
    del pol
    gc.collect()
    torch.cuda.empty_cache()
    return res


def cached_eval(cfg, adapter, name, rm, rm_tok, n=100):
    p = repo_path(f"{cfg['results_dir']}/eval_{name}.json")
    if p.exists():
        print(f"[skip] eval_{name}.json exists")
        return load_json(p)
    return evaluate_adapter(cfg, adapter, name, rm, rm_tok, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'midpoint' / 'base'")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    adapter = args.adapter
    if adapter == "midpoint":
        adapter = cfg["paths"]["grpo_midpoint_policy"]
    elif adapter == "base":
        adapter = None
    print(evaluate_adapter(cfg, adapter, args.name, n=args.n))


if __name__ == "__main__":
    main()
