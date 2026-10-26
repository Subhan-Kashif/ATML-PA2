from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import load_json

POLS = ["sft", "rlvr", "rlaif"]
CATS = ["reasoning_corruption", "wrong_final_good_reasoning", "gold_distractor_wrong_final", "persuasive_filler"]


def pct(x):
    return "NA" if x is None else f"{100 * x:.1f}%"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["results_dir"]) / "task5_feedback"

    gsm = load_json(out / "gsm_metrics.json")
    tr = load_json(out / "transfer_metrics.json")
    dg = load_json(out / "diagnostic_results.json")

    md = ["# Task 5: RLVR vs RLAIF summary\n"]

    # ---- Table 1: in-domain
    rows = []
    for p in POLS:
        m, w = gsm["policies"][p], gsm["vs_sft"].get(p)
        rows.append([p, pct(m["exact_accuracy"]), pct(m["format_compliance"]), f"{m['mean_response_tokens']:.0f}",
                     f"{m['mean_response_words']:.0f}", pct(m["truncation_rate"]),
                     pct(w["win_rate_ties_half"]) if w else "-",
                     f"{w['judge_wins']}/{w['judge_ties']}/{w['judge_losses']}" if w else "-",
                     pct(w["agreement_all"]) if w else "-",
                     pct(w["agreement_when_verifier_decisive"]) if w else "-",
                     pct(w["judge_decisive_when_verifier_tie"]) if w else "-"])
    md += [f"\n## 1. In-domain GSM8K (n={gsm['n']})\n",
           md_table(["policy", "exact acc", "format", "tokens", "words", "trunc", "judge win vs SFT (ties=0.5)",
                     "W/T/L", "verifier-judge agree", "agree | verifier decisive", "judge decisive | verifier tie"], rows)]

    # ---- Table 2: diagnostics
    rows = []
    for c in CATS:
        for mech in ["verifier", "judge"]:
            a = dg["categories"][c][mech]
            rows.append([c, mech, pct(a["better_rate"]), pct(a["tie_rate"]), pct(a["wrong_rate"])])
    for name in ["S_reason", "S_outcome"]:
        for mech in ["verifier", "judge"]:
            a = dg[name][mech]
            rows.append([name, mech, pct(a["better_rate"]), pct(a["tie_rate"]), pct(a["wrong_rate"])])
    md += [f"\n## 2. Controlled diagnostics ({dg['n_problems']} problems x 5 variants)\n",
           md_table(["category", "mechanism", "better", "tie", "wrong"], rows)]
    rows = [[v, f"{r['verifier_exact']:.2f}", f"{r['judge_group_winrate']:.2f}",
             f"{dg['verbosity']['mean_words_by_variant'][v]:.0f}"] for v, r in dg["variant_mean_reward"].items()]
    md += ["\n### Mean reward per variant (judge = K=5 round-robin win rate) and mean words\n",
           md_table(["variant", "verifier", "judge win rate", "mean words"], rows)]
    vb = dg["verbosity"]
    md.append(f"\nJudge prefers the longer response in {pct(vb['judge_prefers_longer_among_decisive_pairs'])} of "
              f"{vb['n_decisive_length_differing_pairs']} decisive length-differing pairs; "
              f"corr(win rate, words)={vb['corr_group_winrate_vs_words']:.2f}.")

    # ---- Table 3: out-of-domain
    rows = []
    for p in POLS:
        g, t = gsm["policies"][p], tr["policies"][p]
        gw, tw = gsm["vs_sft"].get(p), tr["vs_sft"].get(p)
        ft = ", ".join(f"{k}={v * 100:.0f}%" for k, v in t["failure_types"].items() if k != "correct")
        rows.append([p, pct(g["exact_accuracy"]), pct(t["exact_accuracy"]),
                     f"{100 * (t['exact_accuracy'] - g['exact_accuracy']):+.1f}pp",
                     f"{g['mean_response_tokens']:.0f}", f"{t['mean_response_tokens']:.0f}",
                     pct(gw["win_rate_ties_half"]) if gw else "-", pct(tw["win_rate_ties_half"]) if tw else "-",
                     f"{100 * (tw['win_rate_ties_half'] - gw['win_rate_ties_half']):+.1f}pp" if gw and tw else "-",
                     ft or "none"])
    md += [f"\n## 3. Out-of-domain SVAMP (n={tr['n']})\n",
           md_table(["policy", "GSM acc", "SVAMP acc", "drop", "GSM tok", "SVAMP tok", "GSM win/SFT",
                     "SVAMP win/SFT", "win drop", "SVAMP failure types"], rows)]

    # ---- Table 4: cost
    rows = []
    for p in POLS:
        rows.append([p, f"{gsm['policies'][p]['gen_seconds']:.0f}s", f"{tr['policies'][p]['gen_seconds']:.0f}s"])
    md += ["\n## 4. Inference cost\n", md_table(["policy", "GSM generation", "SVAMP generation"], rows)]
    for name, d in [("GSM", gsm), ("SVAMP", tr)]:
        j = d.get("judge", {})
        if j and j.get("seconds_per_new_call"):
            md.append(f"\n{name} judge: {j['new_calls']} new pairwise calls, {j['seconds_per_new_call']:.2f}s/call.")
    j = dg["judge_cost"]
    if j.get("seconds_per_new_call"):
        md.append(f"\nDiagnostic judge: {j['new_calls']} new pairwise calls, {j['seconds_per_new_call']:.2f}s/call. "
                  "The exact verifier is a regex + float compare (~0 cost).")

    (out / "final_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    x, wd = np.arange(3), 0.38
    ax[0].bar(x - wd / 2, [gsm["policies"][p]["exact_accuracy"] for p in POLS], wd, label="GSM8K")
    ax[0].bar(x + wd / 2, [tr["policies"][p]["exact_accuracy"] for p in POLS], wd, label="SVAMP")
    ax[0].set_xticks(x); ax[0].set_xticklabels(POLS); ax[0].set_ylabel("exact accuracy"); ax[0].legend()
    ax[0].set_title("In-domain vs transfer accuracy")

    labels, b, t_, w = [], [], [], []
    for c in CATS:
        for mech in ["verifier", "judge"]:
            a = dg["categories"][c][mech]
            labels.append(f"{c[:14]}\n{mech}")
            b.append(a["better_rate"]); t_.append(a["tie_rate"]); w.append(a["wrong_rate"])
    xi = np.arange(len(labels))
    ax[1].bar(xi, b, label="better", color="tab:green")
    ax[1].bar(xi, t_, bottom=b, label="tie", color="tab:gray")
    ax[1].bar(xi, w, bottom=np.array(b) + np.array(t_), label="wrong", color="tab:red")
    ax[1].set_xticks(xi); ax[1].set_xticklabels(labels, fontsize=6); ax[1].legend(fontsize=7)
    ax[1].set_title("Controlled diagnostics")

    xx = np.arange(2)
    ax[2].bar(xx - wd / 2, [gsm["vs_sft"][p]["win_rate_ties_half"] for p in ["rlvr", "rlaif"]], wd, label="GSM8K")
    ax[2].bar(xx + wd / 2, [tr["vs_sft"][p]["win_rate_ties_half"] for p in ["rlvr", "rlaif"]], wd, label="SVAMP")
    ax[2].axhline(0.5, color="k", ls="--", lw=0.8)
    ax[2].set_xticks(xx); ax[2].set_xticklabels(["rlvr", "rlaif"]); ax[2].set_title("Judge win rate vs SFT"); ax[2].legend()
    plt.tight_layout()
    plt.savefig(out / "task5_summary.png", dpi=150)
    print("Saved:", out / "final_summary.md", "and", out / "task5_summary.png")


if __name__ == "__main__":
    main()
