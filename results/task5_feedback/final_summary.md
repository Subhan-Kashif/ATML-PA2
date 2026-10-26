# Task 5: RLVR vs RLAIF summary


## 1. In-domain GSM8K (n=300)

| policy | exact acc | format | tokens | words | trunc | judge win vs SFT (ties=0.5) | W/T/L | verifier-judge agree | agree | verifier decisive | judge decisive | verifier tie |
|---|---|---|---|---|---|---|---|---|---|---|
| sft | 28.3% | 35.0% | 274 | 156 | 1.7% | - | - | - | - | - |
| rlvr | 28.7% | 36.0% | 276 | 157 | 2.0% | 50.3% | 8/286/6 | 92.0% | 7.7% | 4.2% |
| rlaif | 28.7% | 36.0% | 275 | 157 | 2.3% | 49.5% | 5/287/8 | 93.7% | 0.0% | 4.1% |

## 2. Controlled diagnostics (20 problems x 5 variants)

| category | mechanism | better | tie | wrong |
|---|---|---|---|---|
| reasoning_corruption | verifier | 0.0% | 100.0% | 0.0% |
| reasoning_corruption | judge | 0.0% | 100.0% | 0.0% |
| wrong_final_good_reasoning | verifier | 100.0% | 0.0% | 0.0% |
| wrong_final_good_reasoning | judge | 0.0% | 100.0% | 0.0% |
| gold_distractor_wrong_final | verifier | 100.0% | 0.0% | 0.0% |
| gold_distractor_wrong_final | judge | 25.0% | 70.0% | 5.0% |
| persuasive_filler | verifier | 0.0% | 100.0% | 0.0% |
| persuasive_filler | judge | 0.0% | 75.0% | 25.0% |
| S_reason | verifier | 0.0% | 100.0% | 0.0% |
| S_reason | judge | 0.0% | 100.0% | 0.0% |
| S_outcome | verifier | 100.0% | 0.0% | 0.0% |
| S_outcome | judge | 12.5% | 85.0% | 2.5% |

### Mean reward per variant (judge = K=5 round-robin win rate) and mean words

| variant | verifier | judge win rate | mean words |
|---|---|---|---|
| clean_correct | 1.00 | 0.49 | 56 |
| corrupt_reasoning_correct_final | 1.00 | 0.50 | 56 |
| good_reasoning_wrong_final | 0.00 | 0.47 | 56 |
| persuasive_filler_correct | 1.00 | 0.59 | 66 |
| gold_distractor_wrong_final | 0.00 | 0.44 | 16 |

Judge prefers the longer response in 85.2% of 27 decisive length-differing pairs; corr(win rate, words)=0.14.

## 3. Out-of-domain SVAMP (n=100)

| policy | GSM acc | SVAMP acc | drop | GSM tok | SVAMP tok | GSM win/SFT | SVAMP win/SFT | win drop | SVAMP failure types |
|---|---|---|---|---|---|---|---|---|---|
| sft | 28.3% | 44.0% | +15.7pp | 274 | 164 | - | - | - | missing_format=43%, wrong_answer=13% |
| rlvr | 28.7% | 46.0% | +17.3pp | 276 | 164 | 50.3% | 51.5% | +1.2pp | missing_format=41%, wrong_answer=13% |
| rlaif | 28.7% | 44.0% | +15.3pp | 275 | 163 | 49.5% | 52.5% | +3.0pp | missing_format=43%, wrong_answer=13% |

## 4. Inference cost

| policy | GSM generation | SVAMP generation |
|---|---|---|
| sft | 576s | 136s |
| rlvr | 747s | 176s |
| rlaif | 751s | 175s |

GSM judge: 354 new pairwise calls, 0.61s/call.

SVAMP judge: 107 new pairwise calls, 0.48s/call.

Diagnostic judge: 200 new pairwise calls, 0.43s/call. The exact verifier is a regex + float compare (~0 cost).