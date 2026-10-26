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
from common.metrics import masked_mean
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


# ---------------------------------------------------------------- shared helpers
def to_fp32_trainable(model):
    # fp16 trainable params + Adam eps => NaNs; keep trainable params fp32 (forward uses autocast/peft casts)
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()


def entropy_sum(logits, mask):
    # returns (sum of token entropies over valid tokens, number of valid tokens)
    with torch.no_grad():
        logp = torch.log_softmax(logits.float(), dim=-1)
        ent = -(logp.exp() * logp).sum(-1)
        return float((ent * mask).sum().item()), float(mask.sum().item())


def critic_values(value_model, seq, attn, pw, R, grad=False):
    # V(s_t) for response step t = value head at position (pw - 1 + t), same alignment as the log-probs
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        v = token_values(value_model, seq, attn)
    return v.float()[:, pw - 1 : pw - 1 + R]


def select_prompts(rows, tok, n, max_len, seed):
    # deterministic seeded order, keeping only prompts that fit max_len (so generation never truncates a prompt)
    order = np.random.default_rng(seed).permutation(len(rows))
    out = []
    for i in order:
        r = rows[int(i)]
        if len(tok.apply_chat_template(prompt_messages(r), tokenize=True, add_generation_prompt=True)) <= max_len:
            out.append(r)
            if len(out) == n:
                break
    return out


def _normalize_gen(g):
    # batch_generate runs under torch.inference_mode(); clone outside it so autograd can use these tensors
    for k in ("sequences", "attention_mask", "response_ids", "response_mask"):
        g[k] = g[k].clone()
    return g


def collect_rollout(policy, tok, msgs, cfg, max_new):
    gen = cfg["generation"]
    policy.config.use_cache = True
    g = batch_generate(
        policy, tok, msgs,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=max_new,
        temperature=float(gen["temperature"]),
        top_p=float(gen["top_p"]),
        do_sample=bool(gen["do_sample"]),
    )
    policy.config.use_cache = False
    return _normalize_gen(g)


def fork_stats(cfg, name):
    rows = read_jsonl(f"{cfg['results_dir']}/{name}_log.jsonl")
    r = np.array([x["reward_raw"] for x in rows], dtype=float)
    d = np.diff(r) if len(r) > 1 else np.array([0.0])
    k = min(3, len(r))
    return {
        "train_reward_mean": float(r.mean()),
        "train_reward_gain_last_vs_first": float(r[-k:].mean() - r[:k].mean()),
        "reward_step_std": float(d.std()),                      # stability: volatility of update-to-update reward change
        "kl_max": float(max(x["kl_token"] for x in rows)),
        "grad_norm_max": float(max(x["grad_norm"] for x in rows)),
        "clip_fraction_mean": float(np.mean([x["clip_fraction"] for x in rows])),
        "value_loss_last": float(rows[-1]["value_loss"]),
        "tokens_generated_total": float(sum(x["response_length"] for x in rows)),
    }


def ensure_fork(config_path, cfg, name, eps, beta):
    out = f"outputs/task2_ppo/{name}"
    if (repo_path(out) / "adapter_config.json").exists():
        print(f"[skip] {out} already trained")
    else:
        run_ppo(config_path, output=out, updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=beta, run_name=name)
        gc.collect()
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- the PPO loop
def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, pol, vm = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    pol_opt, val_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    to_fp32_trainable(pol)
    to_fp32_trainable(vm)
    pol.eval()   # dropout off in both nets => ratio is exactly 1 on the first epoch
    vm.eval()

    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    max_norm, vcoef = float(cfg["max_grad_norm"]), float(cfg["value_coef"])
    n_upd, ppu = int(cfg["updates"]), int(cfg["prompts_per_update"])
    rpp = int(cfg.get("responses_per_prompt", 1))
    seed, max_new = int(cfg["seed"]), int(cfg["max_response_length"])
    prompts = select_prompts(bundle["prompt_rows"], tok, n_upd * ppu, int(cfg["max_prompt_length"]), seed)
    assert len(prompts) == n_upd * ppu, "not enough prompts that fit max_prompt_length"
    pol_params, val_params = trainable_parameters(pol), trainable_parameters(vm)

    log_path = f"{cfg['results_dir']}/{run_name}_log.jsonl"
    repo_path(log_path).unlink(missing_ok=True)
    rollout_log = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for u in range(n_upd):
        torch.manual_seed(seed * 1000 + u)   # identical first rollout across forks
        rows_u = prompts[u * ppu:(u + 1) * ppu]
        msgs = [prompt_messages(r) for r in rows_u for _ in range(rpp)]

        # ---- rollout (old policy), reference, reward, critic
        g = collect_rollout(pol, tok, msgs, cfg, max_new)
        seq, attn, pw = g["sequences"], g["attention_mask"], g["prompt_width"]
        rids, mask = g["response_ids"], g["response_mask"]
        R = rids.shape[1]
        with torch.no_grad():
            old_lp, old_logits = response_token_logprobs(pol, seq, attn, pw, rids)
            ent_s, ent_n = entropy_sum(old_logits, mask)
            del old_logits
            with reference_mode(pol):
                ref_lp, _ = response_token_logprobs(pol, seq, attn, pw, rids)
            values = critic_values(vm, seq, attn, pw, R)
            rm_score = score_reward_pairs(rm, rm_tok, msgs, g["responses"], max_length=int(cfg["reward_max_length"])).float().to(old_lp.device)
        term = torch.tensor(g["terminated_with_eos"], dtype=torch.float32, device=old_lp.device)
        task_r = rm_score - float(cfg["missing_eos_penalty"]) * (1.0 - term)
        rewards = shaped_rewards(task_r, old_lp, ref_lp, mask, beta)
        adv, rets = compute_gae(rewards, values, mask, gamma, lam)
        adv_n = normalize_advantages(adv, mask)

        valid = mask.bool()
        v_, r_ = values[valid], rets[valid]
        v_mse = float(((v_ - r_) ** 2).mean())
        ev = float(1 - (r_ - v_).var(unbiased=False) / r_.var(unbiased=False).clamp_min(1e-8)) if r_.numel() > 1 else float("nan")
        kl_tok = float(masked_mean(old_lp - ref_lp, mask))
        kl_seq = float(((old_lp - ref_lp) * mask).sum(-1).mean())

        # ---- PPO epochs on this batch
        st = {k: [] for k in ["policy_loss", "value_loss", "grad_norm", "v_grad_norm", "clip_fraction", "ratio_mean"]}
        for ep in range(int(cfg["ppo_epochs"])):
            new_lp, new_logits = response_token_logprobs(pol, seq, attn, pw, rids)
            del new_logits
            ploss, ratio, clipf = ppo_policy_loss(new_lp, old_lp, adv_n, mask, eps)
            pol_opt.zero_grad(set_to_none=True)
            ploss.backward()
            gn = clip_grad_norm_(pol_params, max_norm)
            if torch.isfinite(gn):
                pol_opt.step()

            vpred = critic_values(vm, seq, attn, pw, R, grad=True)
            vloss = value_mse_loss(vpred, rets.detach(), mask)
            val_opt.zero_grad(set_to_none=True)
            (vcoef * vloss).backward()
            vgn = clip_grad_norm_(val_params, max_norm)
            if torch.isfinite(vgn):
                val_opt.step()

            st["policy_loss"].append(float(ploss.detach()))
            st["value_loss"].append(float(vloss.detach()))
            st["grad_norm"].append(float(gn))
            st["v_grad_norm"].append(float(vgn))
            st["clip_fraction"].append(float(clipf))
            st["ratio_mean"].append(float(masked_mean(ratio, mask)))
            del new_lp, ploss, vloss, vpred

        rec = {
            "update": u + 1, "prompt_id": rows_u[0].get("prompt_id", rows_u[0].get("source_index")),
            "reward_raw": float(rm_score.mean()), "reward_task": float(task_r.mean()),
            "kl_token": kl_tok, "kl_seq": kl_seq, "entropy": ent_s / max(ent_n, 1.0),
            "response_length": float(mask.sum(-1).mean()), "truncated_frac": float(np.mean(g["truncated"])),
            "value_mse_pre": v_mse, "value_explained_var": ev, "value_bias": float(v_.mean() - r_.mean()),
            "policy_loss": float(np.mean(st["policy_loss"])), "value_loss": float(np.mean(st["value_loss"])),
            "grad_norm": float(np.mean(st["grad_norm"])), "value_grad_norm": float(np.mean(st["v_grad_norm"])),
            "clip_fraction": float(np.mean(st["clip_fraction"])), "ratio_mean": float(np.mean(st["ratio_mean"])),
            "elapsed": time.perf_counter() - t0, "eps": eps, "beta_kl": beta,
        }
        append_jsonl(log_path, rec)
        rollout_log.append({"update": u + 1, "prompt": msgs[0][-1]["content"], "response": g["responses"][0], "rm_score": float(rm_score[0])})
        print(f"[{run_name}] upd {u+1}/{n_upd} | R {rec['reward_raw']:.2f} | KL {kl_tok:.4f} | H {rec['entropy']:.2f} | len {rec['response_length']:.0f} | clip {rec['clip_fraction']:.3f} | gn {rec['grad_norm']:.2f} | vloss {rec['value_loss']:.3f}")
        del g, old_lp, ref_lp, values, rewards, adv, rets, adv_n
        torch.cuda.empty_cache()

    pol.save_pretrained(str(out))
    write_jsonl(f"{cfg['results_dir']}/{run_name}_rollouts.jsonl", rollout_log)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    save_json(f"{cfg['results_dir']}/{run_name}_summary.json", {
        "run": run_name, "updates": n_upd, "clip_epsilon": eps, "kl_beta": beta,
        "wall_s": time.perf_counter() - t0, "peak_vram_gib": peak, "responses_per_prompt": rpp,
    })
    print("Saved adapter:", out, "| wall", round(time.perf_counter() - t0), "s | peak VRAM", round(peak, 2), "GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
