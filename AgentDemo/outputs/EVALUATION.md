# Evaluation

This document records how the data-center site forecaster was evaluated and what
the results show. It is generated from the automated harness in `eval.py`, which
writes `outputs/eval_results.csv`. Reproduce it with:

```bash
python eval.py        # prints the table, writes outputs/eval_results.csv
```

## Method

Evaluation here tests **behavior**, not forecast accuracy. Each check is a
small, deterministic assertion about one capability of the system, run over the
same synthetic data the demo uses (`data/candidate_sites.json`,
`data/past_projects.json`). A capability that regresses becomes a `FAIL` row
rather than a surprise during the live demo. The checks span every module: data
grounding (M3), the reasoning and coordination path (M2/M5), and the safety
layer (M6), plus basic input and output invariants (M1).

The harness is deterministic and offline. It needs no API keys and does not
depend on the local LLM, because the autonomous pipeline it exercises is itself
deterministic. The language model drives only the interactive chat front-end.

## Results

Latest run: **9 / 9 checks passed.**

| Check | Module | What it verifies | Observed |
|---|---|---|---|
| grounding_changes_decision | M3 | Retrieved precedent actually moves the forecast | Quincy score 59.3 → 39.3, timeline 44 → 58 mo, 1 hit |
| threshold_blocks_false_analogy | M3 | The similarity floor stops an irrelevant precedent from penalizing a good site | San Antonio grounded 73.5 = base 73.5, stays "Strong candidate" |
| source_failure_flagged | M2/M5 | A failed data source is surfaced, not swallowed | Researcher flag recorded, Critic verdict FLAG |
| guardrail_downgrades_unsupported_strong | M6 | An unearned "Strong" is downgraded automatically | Strong → Conditional when ungrounded and high-risk |
| high_capital_escalated_to_human | M6 | High-capital recommendations require a human sign-off | 1 "Strong" found, all escalated for review |
| ranking_deterministic | M5 | The autonomous sweep is reproducible | Identical order across repeated runs |
| scores_in_range | M1 | Every weighted score stays within 0-100 | No out-of-range scores |
| config_validation_catches_bad_weights | M1/M6 | Invalid leadership weights are rejected before a run | Default valid, tampered weights caught |
| unknown_site_refused | M2 | An unknown site is refused, not fabricated | Returns "No site matching ..." |

## Key quantitative findings

The headline behavior is that grounding changes decisions in a way you can
measure. For Quincy, two comparable past builds that overran on grid
energization pull the score from 59.3 to 39.3, add 14 months to the timeline,
raise the risk tier from Medium to High, and flip the recommendation to "Weak /
deprioritize". For San Antonio, the one comparable build that came in on time
clears the similarity threshold while weak cross-region matches do not, so the
site keeps its 73.5 "Strong" score. That contrast is the evidence that
retrieval both helps and knows when to stay quiet.

## Strengths

1. **Grounding is real and auditable.** Retrieval measurably shifts score,
   timeline, and risk, every adjustment cites its source, and the similarity
   threshold demonstrably blocks false analogies rather than penalizing every
   site equally.
2. **Safety is enforced in code, not just requested in a prompt.** An
   unsupported "Strong" is downgraded automatically, a failed source is flagged
   instead of silently treated as safe, and any high-capital call is escalated
   to a human. All of it is repeatable, which is why the harness can assert it.
3. **The whole thing is cohesive and reproducible.** One command runs Modules 1
   through 6 end to end, the ranking is deterministic, and it runs offline with
   no API keys, which makes both the demo and the evaluation dependable.

## Limitations

1. **Small synthetic corpus.** There are six sites and six precedents, and the
   retrieval thresholds are tuned to that corpus. Real siting data would need
   re-tuning, and the default lexical retriever matches words rather than
   meaning until `USE_EMBEDDINGS` is switched on.
2. **The scoring and reasoning are deterministic heuristics, not learned.** The
   criterion weights, the mitigation utilities, and the guardrail thresholds
   encode reasonable judgment, but they are hand-set. The autonomous pipeline
   deliberately does not rely on the 4B model, which keeps it reliable but means
   the "reasoning" is transparent arithmetic rather than model inference.
3. **The evaluation tests behavior, not predictive accuracy.** These checks
   confirm the system does what it is designed to do. They do not backtest
   whether a site's forecast timeline was ultimately correct, because that would
   require real outcome data. Predictive validation is future work.
