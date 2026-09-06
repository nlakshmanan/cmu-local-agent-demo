"""
main.py -- the AUTONOMOUS entrypoint (Module 1-6 as one cohesive system).

app.py is the interactive chat face. This is the other way to run the system:
head-less and end to end. It sweeps EVERY candidate site through the multi-agent
Orchestrator (Researcher -> Analyst -> Guardrails -> Critic -> escalation),
prints a ranked shortlist, runs the human-review gate on the sites that need a
sign-off, and writes a machine-readable outputs/run_report.json.

    python main.py              # run, then approve flagged sites interactively
    python main.py --no-input   # unattended: flagged sites are left 'pending'

This file is the thing to point at when someone asks "show me the whole system
working on its own."
"""

import argparse
import datetime
import json
from pathlib import Path

import config
import guardrails
from agents import Orchestrator, render_ranking

OUTPUT_DIR = Path(__file__).parent / "outputs"
REPORT_PATH = OUTPUT_DIR / "run_report.json"


def main():
    parser = argparse.ArgumentParser(description="Autonomous data-center site forecaster.")
    parser.add_argument("--no-input", action="store_true",
                        help="don't prompt for human review; leave flagged sites 'pending'.")
    args = parser.parse_args()

    # --- Input guardrail: refuse to run on an invalid configuration. --------
    issues = guardrails.validate_weights()
    if issues:
        raise SystemExit("Config guardrail failed: " + "; ".join(issues))

    print(f"Data-center site forecaster -- autonomous run ({datetime.date.today()})")
    print(f"Leadership weights: {config.CRITERION_WEIGHTS}\n")

    # --- Autonomous multi-agent sweep over every site. ----------------------
    orch = Orchestrator(verbose=False)
    records = orch.evaluate_all()
    print(render_ranking(records))

    # --- Human-in-the-loop gate on the sites that need a sign-off. ----------
    needing = [r for r in records if r["human_review_required"]]
    print()
    if needing and not args.no_input:
        print(f"{len(needing)} site(s) require human review before shipping:\n")
        orch.review_gate(records, interactive=True)
    else:
        orch.review_gate(records, interactive=False)
        if needing:
            print(f"{len(needing)} site(s) marked 'pending' (run without --no-input to approve).")

    # --- Write the machine-readable report. ---------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "weights": config.CRITERION_WEIGHTS,
        "sites": [{k: v for k, v in r.items() if k != "trace"} for r in records],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")

    # --- One-line executive summary. ----------------------------------------
    top = records[0]
    print(f"\nTop site: {top['site_name']} -- {top['score']}/100, "
          f"{top['recommendation']} "
          f"(decision: {top['human_decision']}).")


if __name__ == "__main__":
    main()
