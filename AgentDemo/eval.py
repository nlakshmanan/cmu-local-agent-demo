"""
eval.py -- MODULE 6: evaluation harness.

"It answered with numbers but called no tool" is invisible until you test for
it. This file turns the behaviors we care about into repeatable pass/fail
checks over the whole system, so a regression shows up as a red row instead of
a surprise in the demo.

Each check is a small, deterministic assertion about one capability:

    grounding actually changes a decision              (M3)
    the similarity threshold blocks a false analogy    (M3)
    a failed data source is surfaced, not swallowed    (M2/M5)
    a guardrail downgrades an unsupported 'Strong'      (M6)
    a high-capital call is escalated to a human         (M6)
    the autonomous ranking is deterministic             (M5)
    every score stays in range                          (M1)
    config validation catches bad weights               (M1/M6)
    an unknown site is refused, not fabricated          (M2)

Run:  python eval.py         (writes outputs/eval_results.csv, exits non-zero on any fail)
"""

import csv
import sys
from pathlib import Path

import course_server as srv
import guardrails
import agents

OUTPUT_DIR = Path(__file__).parent / "outputs"
CSV_PATH = OUTPUT_DIR / "eval_results.csv"

_CHECKS = []


def check(check_id, module):
    def deco(fn):
        _CHECKS.append((check_id, module, fn))
        return fn
    return deco


# ---------------------------------------------------------------------------
# The checks. Each returns (passed: bool, expected: str, observed: str).
# ---------------------------------------------------------------------------
@check("grounding_changes_decision", "M3")
def _c1():
    site = srv._load_sites()["site-c"]           # Quincy: long grid queue, overran precedent
    base = srv._score(site)
    f = srv.grounded_forecast(site)
    passed = (f["score"] < base["score"]
              and f["timeline_months"] > base["timeline_months"]
              and len(f["hits"]) >= 1)
    return passed, "grounded score/timeline worsen vs live-data baseline", \
        f"score {base['score']}->{f['score']}, timeline {base['timeline_months']}->{f['timeline_months']}, {len(f['hits'])} hit(s)"


@check("threshold_blocks_false_analogy", "M3")
def _c2():
    site = srv._load_sites()["site-b"]           # San Antonio: comparable build came in ON TIME
    base = srv._score(site)
    f = srv.grounded_forecast(site)
    passed = f["score"] == base["score"] and f["recommendation"] == "Strong candidate"
    return passed, "no overrun penalty; stays Strong", \
        f"grounded {f['score']} vs base {base['score']}, rec '{f['recommendation']}'"


@check("source_failure_flagged", "M2/M5")
def _c3():
    rec = agents.Orchestrator().evaluate_site("site-c", fail_source="hazard")
    passed = bool(rec["flags"]) and rec["verdict"] == "FLAG"
    return passed, "failure recorded + Critic FLAG", \
        f"flags={len(rec['flags'])}, verdict={rec['verdict']}"


@check("guardrail_downgrades_unsupported_strong", "M6")
def _c4():
    forged = {"recommendation": "Strong candidate", "hits": [], "risk_tier": "High"}
    g = guardrails.guard_output(forged, source_failed=False)
    passed = g["downgraded"] and g["recommendation"].startswith("Conditional")
    return passed, "Strong -> Conditional when unsupported", \
        f"downgraded={g['downgraded']}, rec='{g['recommendation']}'"


@check("high_capital_escalated_to_human", "M6")
def _c5():
    recs = agents.Orchestrator().evaluate_all()
    strongs = [r for r in recs if r["recommendation"] == "Strong candidate"]
    passed = len(strongs) >= 1 and all(r["human_review_required"] for r in strongs)
    return passed, ">=1 Strong, all require human review", \
        f"{len(strongs)} Strong; all escalated={all(r['human_review_required'] for r in strongs)}"


@check("ranking_deterministic", "M5")
def _c6():
    order1 = [r["site_id"] for r in agents.Orchestrator().evaluate_all()]
    order2 = [r["site_id"] for r in agents.Orchestrator().evaluate_all()]
    passed = order1 == order2
    return passed, "same order on repeat runs", f"stable={order1 == order2}"


@check("scores_in_range", "M1")
def _c7():
    recs = agents.Orchestrator().evaluate_all()
    bad = [r["site_id"] for r in recs if not (0 <= r["score"] <= 100)]
    return not bad, "every score within 0-100", f"out-of-range={bad or 'none'}"


@check("config_validation_catches_bad_weights", "M1/M6")
def _c8():
    default_ok = guardrails.validate_weights() == []
    tampered_caught = guardrails.validate_weights(
        {"power": 0.5, "connectivity": 0.2, "hazard": 0.2, "permitting": 0.2}) != []
    passed = default_ok and tampered_caught
    return passed, "default valid, bad weights rejected", \
        f"default_ok={default_ok}, tampered_caught={tampered_caught}"


@check("unknown_site_refused", "M2")
def _c9():
    out = srv.get_power("Atlantis")
    passed = out.lower().startswith("no site")
    return passed, "refuse unknown site, do not fabricate", f"'{out[:38]}...'"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run() -> int:
    rows, passed_count = [], 0
    for check_id, module, fn in _CHECKS:
        try:
            passed, expected, observed = fn()
        except Exception as e:
            passed, expected, observed = False, "no exception", f"ERROR: {e}"
        rows.append({"id": check_id, "module": module, "expected": expected,
                     "observed": observed, "result": "PASS" if passed else "FAIL"})
        passed_count += bool(passed)

    OUTPUT_DIR.mkdir(exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "module", "expected", "observed", "result"])
        writer.writeheader()
        writer.writerows(rows)

    width = max(len(r["id"]) for r in rows)
    print(f"{'CHECK':<{width}}  {'MOD':<7} RESULT   OBSERVED")
    print("-" * (width + 55))
    for r in rows:
        print(f"{r['id']:<{width}}  {r['module']:<7} {r['result']:<7}  {r['observed']}")
    total = len(rows)
    print("-" * (width + 55))
    print(f"{passed_count}/{total} checks passed. Wrote {CSV_PATH}")
    return 0 if passed_count == total else 1


if __name__ == "__main__":
    sys.exit(run())
