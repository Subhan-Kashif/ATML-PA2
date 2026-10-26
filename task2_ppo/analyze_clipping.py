from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json
from common.metrics import masked_mean
from common.models import load_policy, load_reward_model, load_tokenizer, load_value_model, trainable_parameters
from task2_ppo.continue_train import critic_values, ensure_fork, fork_stats, to_fp32_trainable
from task2_ppo.evaluate import cached_eval
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def pad_rows(tensors, R):
    out = torch.zeros((len(tensors), R), dtype=torch.float32, device=tensors[0].device)
    for i, t in enumerate(tensors):
        out[i, : len(t)] = t
    return out


def build_cached_batch(cfg, rows, tok, rm, rm_tok, critic, beta):
    pool = read_jsonl(cfg["paths"]["rl_prompt_train"]) + read_jsonl(cfg["paths"]["rl_prompt_eval"])
    by_idx = {str(r["source_index"]): r for r in pool}
    dev = next(critic.parameters()).device
    eos = tok.eos_token_id
    items, olds, refs, vals, task_r = [], [], [], [], []
    for row in rows:
        key = str(row["source_index"])
        if key not in by_idx:
            raise KeyError(f"cached source_index {key} not found in the RL prompt pools")
        msgs = prompt_messages(by_idx[key])
        old = torch.as_tensor(row["old_logprobs"]).float().flatten()
        ref = torch.as_tensor(row["ref_logprobs"]).float().flatten()
        if "response_ids" in row:
            ids = [int(x) for x in torch.as_tensor(row["response_ids"]).flatten().tolist()]
        else:
            ids = tok(str(row["response"]), add_special_tokens=False)["input_ids"]
        n = old.numel()
        if len(ids) == n - 1 and eos is not None:
            ids = ids + [eos]
        if len(ids) != n:
            print(f"[warn] {key}: {len(ids)} response ids vs {n} cached logprobs -> trimming to min")
            n = min(len(ids), n)
            ids, old, ref = ids[:n], old[:n], ref[:n]
        pids = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        seq = torch.tensor([pids + ids], device=dev)
        attn = torch.ones_like(seq)
        pw = len(pids)
        r = next((float(row[k]) for k in ("reward", "rm_reward", "rm_score", "task_reward", "score") if k in row and row[k] is not None), None)
        if r is None:
            text = str(row["response"]) if "response" in row else tok.decode(ids, skip_special_tokens=True)
            r = float(score_reward_pairs(rm, rm_tok, [msgs], [text], max_length=int(cfg["reward_max_length"]))[0])
        term = bool(ids and eos is not None and ids[-1] == eos)
        r = r - float(cfg["missing_eos_penalty"]) * (0.0 if term else 1.0)
        with torch.no_grad():
            v = critic_values(critic, seq, attn, pw, n)[0]
        items.append({"seq": seq, "attn": attn, "pw": pw, "n": n, "rids": seq[:, pw:]})
        olds.append(old.to(dev)); refs.append(ref.to(dev)); vals.append(v); task_r.append(r)

    R = max(it["n"] for it in items)
    mask = pad_rows([torch.ones(it["n"], device=dev) for it in items], R)
    old_p, ref_p, val_p = pad_rows(olds, R), pad_rows(refs, R), pad_rows(vals, R)
    tr = torch.tensor(task_r, dtype=torch.float32, device=dev)
    rewards = shaped_rewards(tr, old_p, ref_p, mask, beta)
    adv, rets = compute_gae(rewards, val_p, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv_n = normalize_advantages(adv, mask)
    return {"items": items, "old": old_p, "ref": ref_p, "mask": mask, "adv": adv_n, "adv_raw": adv, "ret": rets, "values": val_p, "task_r": tr}


@torch.no_grad()
def static_study(policy, batch, eps_list):
    outs = []
    for it in batch["items"]:
        lp, _ = response_token_logprobs(policy, it["seq"], it["attn"], it["pw"], it["rids"])
        outs.append(lp[0])
    new = pad_rows(outs, batch["old"].shape[1])
    old, adv, mask = batch["old"], batch["adv"], batch["mask"]
    ratio = torch.exp(new - old)
    dev = (ratio - 1).abs()[mask.bool()].cpu().numpy()
    res = []
    for eps in eps_list:
        loss, _, cf = ppo_policy_loss(new, old, adv, mask, eps)
        s1, s2 = ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv
        binding = masked_mean((s2 < s1).float(), mask)   # tokens where the clipped branch actually sets the objective
        res.append({"eps": eps, "surrogate": float(-loss), "affected_fraction": float(cf), "binding_fraction": float(binding),
                    "n_tokens": int(mask.sum()), "abs_ratio_dev_p50": float(np.percentile(dev, 50)),
                    "abs_ratio_dev_p90": float(np.percentile(dev, 90)), "abs_ratio_dev_p99": float(np.percentile(dev, 99))})
    return res


def probe_clipping(cfg, batch, eps, steps, lr):
    # repeated optimization steps on the fixed cached batch from the midpoint, one fresh policy per eps
    pol = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    to_fp32_trainable(pol)
    pol.eval()
    params = trainable_parameters(pol)
    opt = AdamW(params, lr=lr)
    N, recs = len(batch["items"]), []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = aff = sur = dev = 0.0
        for i, it in enumerate(batch["items"]):
            n = it["n"]
            new_lp, _ = response_token_logprobs(pol, it["seq"], it["attn"], it["pw"], it["rids"])
            loss, ratio, cf = ppo_policy_loss(new_lp, batch["old"][i:i + 1, :n], batch["adv"][i:i + 1, :n], batch["mask"][i:i + 1, :n], eps)
            (loss / N).backward()
            tot += n; aff += float(cf) * n; sur += float(-loss.detach()) * n; dev += float((ratio - 1).abs().mean()) * n
            del new_lp, loss
        gn = clip_grad_norm_(params, 1.0)
        recs.append({"eps": eps, "step": step, "affected_fraction": aff / tot, "surrogate": sur / tot,
                     "mean_abs_ratio_dev": dev / tot, "grad_norm": float(gn)})
        opt.step()
    del pol, opt
    gc.collect()
    torch.cuda.empty_cache()
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-static", action="store_true")
    ap.add_argument("--skip-forks", action="store_true")
    ap.add_argument("--probe-steps", type=int, default=8)
    ap.add_argument("--probe-lr", type=float, default=None)
    ap.add_argument("--n-eval", type=int, default=40)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    eps_list = [float(e) for e in cfg["clip_values"]]
    beta0 = float(cfg["kl_beta"])

    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", eps_list)
    print("Cache keys:", sorted(rows[0].keys()))

    if not args.skip_static:
        tok = load_tokenizer(cfg["base_model"])
        rm, rm_tok = load_reward_model(cfg)
        critic = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
        batch = build_cached_batch(cfg, rows, tok, rm, rm_tok, critic, beta0)

        pol = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
        static = static_study(pol, batch, eps_list)
        save_json(results / "clip_static.json", static)
        print(pd.DataFrame(static).to_string())
        del pol, critic, rm
        gc.collect(); torch.cuda.empty_cache()

        lr = args.probe_lr or float(cfg["policy_learning_rate"])
        probe = []
        for eps in eps_list:
            probe += probe_clipping(cfg, batch, eps, args.probe_steps, lr)
        pd.DataFrame(probe).to_csv(results / "clip_probe.csv", index=False)
        print(pd.DataFrame(probe).groupby("eps").last().to_string())

    if not args.skip_forks:
        names = {}
        for eps in eps_list:
            names[eps] = f"fork_eps{eps:g}_kl{beta0:g}"
            ensure_fork(args.config, cfg, names[eps], eps, beta0)
        rm, rm_tok = load_reward_model(cfg)
        table = []
        for eps in eps_list:
            ev = cached_eval(cfg, f"outputs/task2_ppo/{names[eps]}", names[eps], rm, rm_tok, args.n_eval)
            row = {"eps": eps, **fork_stats(cfg, names[eps])}
            row.update({f"heldout_{k}": ev[k] for k in ["rm_mean", "kl_token", "entropy", "len_mean", "len_std"]})
            table.append(row)
        df = pd.DataFrame(table)
        df.to_csv(results / "clip_forks.csv", index=False)
        print(df.T.to_string())


if __name__ == "__main__":
    main()
