from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

TOL = 1e-6   # numerical tolerance for "zero" group reward std (same value as the helper's eps)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


# ---------------------------------------------------------------- shared helpers
def to_fp32_trainable(model):
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()


def entropy_sum(logits, mask):
    with torch.no_grad():
        logp = torch.log_softmax(logits.float(), dim=-1)
        ent = -(logp.exp() * logp).sum(-1)
        return float((ent * mask).sum().item()), float(mask.sum().item())


def select_prompts(rows, tok, n, max_len, seed):
    # seeded fixed order, keeping prompts that fit max_len so generation never truncates a prompt
    order = np.random.default_rng(seed).permutation(len(rows))
    out = []
    for i in order:
        r = rows[int(i)]
        if len(tok.apply_chat_template(prompt_messages(r), tokenize=True, add_generation_prompt=True)) <= max_len:
            out.append(r)
            if len(out) == n:
                break
    return out


def collect_group(policy, tok, msgs, cfg):
    gen = cfg["generation"]
    policy.config.use_cache = True
    g = batch_generate(
        policy, tok, msgs,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg["max_completion_length"]),
        temperature=float(gen["temperature"]),
        top_p=float(gen["top_p"]),
        do_sample=bool(gen["do_sample"]),
    )
    policy.config.use_cache = False
    for k in ("sequences", "attention_mask", "response_ids", "response_mask"):
        g[k] = g[k].clone()   # leave inference mode so autograd can use these tensors
    return g


def seq_lp(pol, g, mask, with_entropy=False):
    lps, es, en = [], 0.0, 0.0
    for k in range(g["sequences"].shape[0]):
        sl = slice(k, k + 1)
        lp, lg = response_token_logprobs(pol, g["sequences"][sl], g["attention_mask"][sl], g["prompt_width"], g["response_ids"][sl])
        if with_entropy:
            a, b = entropy_sum(lg, mask[sl]); es += a; en += b
        del lg
        lps.append(lp)
    return torch.cat(lps, 0), es, en


def per_seq_grad_norms(pol, params, g, adv, tmask, old_lp, ref_lp, eps, loss_type, max_len):
    """Gradient norm of the POLICY term of each completion separately (beta=0), for the chosen normalization."""
    norms = []
    for k in range(tmask.shape[0]):
        if float(tmask[k].sum()) == 0 or abs(float(adv[k])) < 1e-8:
            norms.append(0.0)
            continue
        sl = slice(k, k + 1)
        new_lp, _ = response_token_logprobs(pol, g["sequences"][sl], g["attention_mask"][sl], g["prompt_width"], g["response_ids"][sl])
        loss, _ = grpo_policy_loss(new_lp, old_lp[sl], adv[sl], tmask[sl], ref_lp[sl], eps, 0.0, loss_type, max_len)
        pol.zero_grad(set_to_none=True)
        loss.backward()
        sq = sum(float((p.grad.float() ** 2).sum()) for p in params if p.grad is not None)
        norms.append(sq ** 0.5)
        pol.zero_grad(set_to_none=True)
        del new_lp, loss
    return norms


def fork_stats(cfg, name):
    rows = read_jsonl(f"{cfg['results_dir']}/{name}_log.jsonl")
    r = np.array([x["reward_mean"] for x in rows], dtype=float)
    d = np.diff(r) if len(r) > 1 else np.array([0.0])
    k = min(3, len(r))
    return {
        "train_reward_mean": float(r.mean()),
        "train_reward_gain_last_vs_first": float(r[-k:].mean() - r[:k].mean()),
        "reward_step_std": float(d.std()),
        "kl_max": float(max(x["kl_token"] for x in rows)),
        "grad_norm_mean": float(np.mean([x["grad_norm"] for x in rows])),
        "grad_norm_max": float(max(x["grad_norm"] for x in rows)),
        "uninformative_frac": float(np.mean([x["uninformative_frac"] for x in rows])),
        "group_std_mean": float(np.mean([x["reward_std_group"] for x in rows])),
        "train_len_mean": float(np.mean([x["response_length"] for x in rows])),
        "masked_frac_mean": float(np.mean([x["masked_frac"] for x in rows])),
        "tokens_generated_total": float(sum(x["tokens_generated"] for x in rows)),
    }


def ensure_fork(config_path, cfg, name, loss_type, updates, probe=False):
    out = f"outputs/task3_grpo/{name}"
    if (repo_path(out) / "adapter_config.json").exists():
        print(f"[skip] {out} already trained")
    else:
        run_grpo(config_path, output=out, updates=updates, loss_type=loss_type, run_name=name, probe_seq_grads=probe)
        gc.collect()
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- the GRPO loop
def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard", probe_seq_grads: bool = False):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, pol = bundle["tokenizer"], bundle["policy"]
    rm, rm_tok, opt = bundle["reward_model"], bundle["reward_tokenizer"], bundle["optimizer"]
    to_fp32_trainable(pol)
    pol.eval()   # dropout off => ratio is exactly 1 on the first epoch
    params = trainable_parameters(pol)

    K, ppu = int(cfg["num_generations"]), int(cfg["prompts_per_update"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    Lmax, max_norm = int(cfg["max_completion_length"]), float(cfg["max_grad_norm"])
    mask_trunc = bool(cfg.get("mask_truncated_completions", True))
    n_upd, seed = int(cfg["updates"]), int(cfg["seed"])
    prompts = select_prompts(bundle["prompt_rows"], tok, n_upd * ppu, int(cfg["max_prompt_length"]), seed)
    assert len(prompts) == n_upd * ppu, "not enough prompts that fit max_prompt_length"

    log_path = f"{cfg['results_dir']}/{run_name}_log.jsonl"
    repo_path(log_path).unlink(missing_ok=True)
    seq_log, roll_log = [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for u in range(n_upd):
        torch.manual_seed(seed * 1000 + u)   # identical first rollout across forks
        rows_u = prompts[u * ppu:(u + 1) * ppu]
        msgs = [prompt_messages(r) for r in rows_u for _ in range(K)]
        g = collect_group(pol, tok, msgs, cfg)
        seq, attn, pw = g["sequences"], g["attention_mask"], g["prompt_width"]
        rids, rmask = g["response_ids"], g["response_mask"]
        dev = rids.device
        group_ids = torch.arange(len(rows_u), device=dev).repeat_interleave(K)

        with torch.no_grad():
            old_lp, ent_s, ent_n = seq_lp(pol, g, rmask, True)
            with reference_mode(pol):
                ref_lp, _, _ = seq_lp(pol, g, rmask)
            rewards = score_reward_pairs(rm, rm_tok, msgs, g["responses"], max_length=int(cfg.get("reward_max_length", 1280))).float().to(dev)

        adv = group_relative_advantages(rewards, group_ids)
        tmask = mask_truncated_sequences(rmask, g["truncated"]) if mask_trunc else rmask
        stds = [float(rewards[group_ids == gi].std(unbiased=False)) for gi in range(len(rows_u))]
        lengths = [int(x) for x in g["response_lengths"]]
        kl_tok = float(sampled_kl(old_lp, ref_lp, rmask))

        if probe_seq_grads:   # per-completion policy-gradient norms, BEFORE the update, for the length-normalization study
            norms = per_seq_grad_norms(pol, params, g, adv, tmask, old_lp, ref_lp, eps, loss_type, Lmax)
            for k in range(len(lengths)):
                seq_log.append({"update": u + 1, "k": k, "length": lengths[k], "adv": float(adv[k]),
                                "grad_norm": norms[k], "masked": bool(tmask[k].sum() == 0)})

        st = {"policy_loss": [], "loss": [], "grad_norm": [], "clip_fraction": [], "kl_k3": []}
        n_seq, n_tok = tmask.shape[0], tmask.sum().clamp_min(1.0)
        for ep in range(int(cfg["policy_epochs"])):
            opt.zero_grad(set_to_none=True)
            pol_t = kl_t = clip_t = 0.0
            for k in range(n_seq):
                nk = tmask[k].sum()
                if float(nk) == 0:
                    continue   # fully masked (truncated) completion: zero gradient
                sl = slice(k, k + 1)
                new_lp, _ = response_token_logprobs(pol, seq[sl], attn[sl], pw, rids[sl])
                pl, d = grpo_policy_loss(new_lp, old_lp[sl], adv[sl], tmask[sl], ref_lp[sl], eps, 0.0, loss_type, Lmax)
                dlt = ref_lp[sl] - new_lp
                kl_k = (((dlt.exp() - dlt - 1.0) * tmask[sl]).sum()) / n_tok   # token-mean KL over the whole batch
                (pl / n_seq + beta * kl_k).backward()                          # policy term = mean over all K sequences
                pol_t += float(pl.detach()) / n_seq
                kl_t += float(kl_k.detach())
                clip_t += float(d["clip_fraction"]) * float(nk) / float(n_tok)
                del new_lp, pl, kl_k
            gn = clip_grad_norm_(params, max_norm)
            if torch.isfinite(gn):
                opt.step()
            st["policy_loss"].append(pol_t); st["loss"].append(pol_t + beta * kl_t)
            st["grad_norm"].append(float(gn)); st["clip_fraction"].append(clip_t); st["kl_k3"].append(kl_t)

        rec = {
            "update": u + 1, "prompt_id": rows_u[0].get("prompt_id", rows_u[0].get("source_index")),
            "reward_mean": float(rewards.mean()), "reward_std_group": float(np.mean(stds)),
            "uninformative_frac": float(np.mean([s <= TOL for s in stds])),
            "kl_token": kl_tok, "kl_k3": float(np.mean(st["kl_k3"])), "entropy": ent_s / max(ent_n, 1.0),
            "response_length": float(np.mean(lengths)), "tokens_generated": float(sum(lengths)),
            "truncated_frac": float(np.mean(g["truncated"])), "masked_frac": float(1.0 - (tmask.sum(-1) > 0).float().mean()),
            "policy_loss": float(np.mean(st["policy_loss"])), "loss": float(np.mean(st["loss"])),
            "grad_norm": float(np.mean(st["grad_norm"])), "clip_fraction": float(np.mean(st["clip_fraction"])),
            "elapsed": time.perf_counter() - t0, "loss_type": loss_type,
        }
        append_jsonl(log_path, rec)
        for k in range(len(lengths)):
            roll_log.append({"update": u + 1, "k": k, "prompt": msgs[k][-1]["content"][:300], "response": g["responses"][k][:600],
                             "reward": float(rewards[k]), "adv": float(adv[k]), "tokens": lengths[k], "truncated": bool(g["truncated"][k])})
        print(f"[{run_name}] upd {u+1}/{n_upd} | R {rec['reward_mean']:.2f} | gstd {rec['reward_std_group']:.2f} | uninf {rec['uninformative_frac']:.0f} | KL {kl_tok:.4f} | H {rec['entropy']:.2f} | len {rec['response_length']:.0f} | trunc {rec['truncated_frac']:.2f} | gn {rec['grad_norm']:.2f}")
        del g, old_lp, ref_lp, rewards, adv
        torch.cuda.empty_cache()

    pol.save_pretrained(str(out))
    write_jsonl(f"{cfg['results_dir']}/{run_name}_rollouts.jsonl", roll_log)
    if probe_seq_grads:
        write_jsonl(f"{cfg['results_dir']}/{run_name}_seqgrads.jsonl", seq_log)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    save_json(f"{cfg['results_dir']}/{run_name}_summary.json", {
        "run": run_name, "loss_type": loss_type, "updates": n_upd, "K": K,
        "wall_s": time.perf_counter() - t0, "peak_vram_gib": peak,
    })
    print("Saved adapter:", out, "| wall", round(time.perf_counter() - t0), "s | peak VRAM", round(peak, 2), "GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--probe-seq-grads", action="store_true")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.probe_seq_grads)


if __name__ == "__main__":
    main()
