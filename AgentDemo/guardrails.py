"""
guardrails.py -- MODULE 6: guardrails, enforced in CODE.

A prompt is a request. A guardrail is a guarantee. Everything in this file is a
hard rule the system obeys regardless of what a language model would prefer to
say. Three kinds:

  1. INPUT validation  -- refuse bad inputs before they reach the pipeline
                          (an unknown site, weights that don't sum to 1.0).
  2. OUTPUT rules      -- a recommendation must be EARNED. A 'Strong' call that
                          rests on a failed source, no precedent, or High risk
                          is downgraded automatically, with the reason recorded.
  3. ESCALATION        -- decide when a result must not ship without a human
                          (see the human-review gate in agents.py).

The Critic (agents.py) offers judgement and can be wrong. These rules do not
offer judgement; they hold the line. That separation -- soft reviewer, hard
guardrail -- is the point.
"""

import config

STRONG = "Strong candidate"
CONDITIONAL = "Conditional -- mitigate the top risk before committing capital"


# ---------------------------------------------------------------------------
# 1. INPUT VALIDATION
# ---------------------------------------------------------------------------
def validate_weights(weights: dict = None) -> list[str]:
    """Return a list of problems with the criterion weights (empty = valid)."""
    weights = weights if weights is not None else config.CRITERION_WEIGHTS
    issues = []
    total = round(sum(weights.values()), 6)
    if total != 1.0:
        issues.append(f"CRITERION_WEIGHTS sum to {total}, must be 1.0")
    for k, v in weights.items():
        if v < 0:
            issues.append(f"weight '{k}' is negative ({v})")
    return issues


def validate_site(site_id: str, known_ids) -> str | None:
    """Return an error string if the site is unknown, else None."""
    if site_id not in set(known_ids):
        return f"unknown site '{site_id}' (known: {', '.join(known_ids)})"
    return None


# ---------------------------------------------------------------------------
# 2. OUTPUT RULES -- a recommendation must be earned.
# ---------------------------------------------------------------------------
def guard_output(forecast: dict, source_failed: bool) -> dict:
    """Enforce the output rules on a grounded forecast.

    Returns {recommendation, actions, downgraded}. `recommendation` may be a
    downgraded version of the forecast's own; `actions` records every rule that
    fired, in plain language, so the change is auditable.
    """
    original = forecast["recommendation"]
    rec = original
    grounded = bool(forecast["hits"])
    high_risk = forecast["risk_tier"] == "High"
    actions = []

    # RULE 1: 'Strong' is only allowed when grounded, not high-risk, and no
    # source failed. Otherwise it is downgraded -- the model does not get to
    # keep an unsupported 'Strong'.
    if original == STRONG and (not grounded or source_failed or high_risk):
        why = []
        if not grounded:
            why.append("no precedent")
        if source_failed:
            why.append("a data source failed")
        if high_risk:
            why.append("risk is High")
        rec = CONDITIONAL
        actions.append(f"downgraded 'Strong' -> 'Conditional' ({', '.join(why)})")

    # RULE 2: an ungrounded forecast must say so.
    if not grounded:
        actions.append("forecast is live-data-only (no precedent); confidence reduced")

    # RULE 3: a source failure must be surfaced, never swallowed.
    if source_failed:
        actions.append("a data source failed on first fetch; result flagged")

    return {"recommendation": rec, "actions": actions, "downgraded": rec != original}


# ---------------------------------------------------------------------------
# 3. ESCALATION -- when must a human sign off?
# ---------------------------------------------------------------------------
def needs_human_review(recommendation: str, confidence: str, guard: dict) -> tuple[bool, list]:
    """Decide whether a result must not ship without a human, and why.

    A human is required for a high-capital 'Strong' call, whenever a guardrail
    had to downgrade something, whenever a source failed, or on low confidence.
    """
    reasons = []
    if recommendation == STRONG:
        reasons.append("high-capital 'Strong' recommendation")
    if guard["downgraded"]:
        reasons.append("a guardrail downgraded the recommendation")
    if any("failed" in a for a in guard["actions"]):
        reasons.append("a data source failed")
    if confidence == "low":
        reasons.append("low analyst/critic confidence")
    return (bool(reasons), reasons)
