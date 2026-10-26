from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import set_seed, append_jsonl, save_json
from common.models import load_policy, load_tokenizer, trainable_parameters, reference_mode
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    tokenizer = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])
    kept = [
        r for r in rows
        if len(tokenizer.apply_chat_template(
            prompt_messages_from_preference(r), tokenize=True, add_generation_prompt=True)) < max_len
    ]
    print(f"Filtered {len(rows) - len(kept)}/{len(rows)} examples with prompt >= max_sequence_length={max_len}")
    rows = kept
    if max_examples is not None:
        rows = rows[: int(max_examples)]
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model, loader, opt, beta = bundle["model"], bundle["loader"], bundle["optimizer"], bundle["beta"]
    accum = int(cfg["grad_accum_steps"])
    max_norm = float(cfg["max_grad_norm"])
    dev = next(model.parameters()).device
    params = trainable_parameters(model)
    log_path = f"{cfg['results_dir']}/{run_name}_train_log.jsonl"
    Path(repo_path(log_path)).unlink(missing_ok=True)  # fresh log per run
    n = len(loader)

    model.train()
    opt.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0, step, win = time.perf_counter(), 0, []

    for i, (ch, rj) in enumerate(loader):
        ch = {k: v.to(dev) for k, v in ch.items()}
        rj = {k: v.to(dev) for k, v in rj.items()}

        # reference = same model with LoRA adapter disabled
        with torch.no_grad(), reference_mode(model):
            ref_c = response_sequence_logprobs(model, ch)[0]
            ref_r = response_sequence_logprobs(model, rj)[0]

        pol_c = response_sequence_logprobs(model, ch)[0]
        pol_r = response_sequence_logprobs(model, rj)[0]

        loss, d = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
        (loss / accum).backward()
        win.append({
            "loss": loss.item(),
            "acc": d["preference_accuracy"].item(),
            "margin": (d["logit_mean"] / beta).item(),
            "chosen_logratio": (pol_c.detach() - ref_c).mean().item(),
            "rejected_logratio": (pol_r.detach() - ref_r).mean().item(),
        })

        if (i + 1) % accum == 0 or (i + 1) == n:
            gn = torch.nn.utils.clip_grad_norm_(params, max_norm).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {k: sum(w[k] for w in win) / len(win) for k in win[0]}
            rec.update(step=step, batch=i + 1, grad_norm=gn, beta=beta,
                       elapsed=time.perf_counter() - t0)
            append_jsonl(log_path, rec)
            print(f"step {step} | loss {rec['loss']:.4f} | acc {rec['acc']:.2f} | gn {gn:.2f}")
            win = []

    model.save_pretrained(str(output))
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    save_json(f"{cfg['results_dir']}/{run_name}_train_summary.json", {
        "run": run_name, "beta": beta, "examples": len(bundle["rows"]), "optimizer_steps": step,
        "wall_s": time.perf_counter() - t0, "peak_vram_gib": peak,
    })
    print("Saved adapter:", output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
