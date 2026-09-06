"""
reasoning.py -- MODULE 4: advanced reasoning (Tree-of-Thoughts).

The rest of the agent answers in a straight line: gather facts, score, reply.
Some questions are not straight lines. "This site has a long grid queue -- what
do we DO about it?" has several possible answers, and the good one only shows
itself once you lay a few side by side and weigh them.

That is Tree-of-Thoughts: instead of committing to the first idea, you BRANCH
into several candidate strategies, EVALUATE each on more than one axis, then
SELECT the best (and prune the rest). This module does exactly that for site
risk mitigation.

    top risk  ->  branch into 2-3 mitigations  ->  score each on
                  (effectiveness, cost, months of relief)  ->  pick the best

WHY DETERMINISTIC?
------------------
The BRANCHES come from a small, auditable knowledge base rather than a 4B
model's imagination, and the SELECTION is explicit arithmetic. That makes the
reasoning reproducible and defensible in front of leadership -- you can point at
why one option won. An LLM is great at *proposing* an extra creative branch;
where it would slot in is marked  # LLM-HOOK.  The controller (branch, score,
select) is the part worth owning in code.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SITES_FILE = DATA_DIR / "candidate_sites.json"

_COST_PENALTY = {"Low": 0, "Medium": 15, "High": 30}


def _load_sites() -> dict:
    return {s["site_id"]: s for s in json.loads(SITES_FILE.read_text(encoding="utf-8"))}


def _find_site(sites: dict, name: str):
    name = (name or "").strip().lower()
    if not name:
        return None
    normalized = name.replace(" ", "-")
    for sid in sites:
        if normalized == sid.lower():
            return sid
    for sid, s in sites.items():
        if name == s["name"].lower() or name in s["name"].lower().split():
            return sid
    for sid, s in sites.items():
        if any(w in s["name"].lower() for w in name.split() if len(w) > 2):
            return sid
    return None


# ---------------------------------------------------------------------------
# STEP 1 -- find the ONE risk worth planning against.
# We score each risk category 0-1 from the site's raw facts and take the max.
# Planning against the biggest lever is itself a small reasoning step.
# ---------------------------------------------------------------------------
def identify_top_risk(site: dict) -> tuple[str, str]:
    p, h, g, pm = site["power"], site["hazard"], site["geo"], site["permitting"]
    risk_levels = {"low": 0, "medium": 1, "high": 2}

    severities = {
        "grid":         (min(p["interconnection_queue_months"] / 30, 1.0),
                         f"interconnection queue is {p['interconnection_queue_months']} months"),
        "water":        ({"low": 0.9, "medium": 0.4, "high": 0.0}[pm.get("water_availability", "high")],
                         f"water availability is {pm.get('water_availability', 'n/a')}"),
        "permitting":   ({"low": 0.0, "medium": 0.4, "high": 0.9}[pm["permit_complexity"]],
                         f"permit complexity is {pm['permit_complexity']}"),
        "hazard":       (sum(risk_levels[h[k]] for k in ("flood_risk", "seismic_risk", "storm_risk")) / 6,
                         f"natural-hazard exposure (flood {h['flood_risk']}, seismic {h['seismic_risk']}, storm {h['storm_risk']})"),
        "power_cost":   (max(0.0, min((p["cost_per_kwh"] - 0.03) / 0.05, 1.0)),
                         f"power cost is ${p['cost_per_kwh']}/kWh"),
        "connectivity": (min(g["latency_ms"] / 50, 1.0),
                         f"latency is {g['latency_ms']} ms, {g['km_to_major_metro']} km to metro"),
    }
    category = max(severities, key=lambda k: severities[k][0])
    return category, severities[category][1]


# ---------------------------------------------------------------------------
# STEP 2 -- the candidate BRANCHES, one small library per risk category.
# Each carries the three axes we will weigh. relief_months is roughly how much
# of the timeline (or how much risk, for hazard) the option buys back.
# ---------------------------------------------------------------------------
MITIGATIONS = {
    "grid": [
        {"name": "On-site generation bridge", "how": "temporary gas or solar-plus-storage to energize before the grid catches up", "effectiveness": 0.75, "cost": "High", "relief_months": 12},
        {"name": "Phased / staged energization", "how": "bring capacity online in blocks as the interconnection clears", "effectiveness": 0.60, "cost": "Medium", "relief_months": 8},
        {"name": "Secure an earlier queue slot / alternate substation", "how": "negotiate an alternate interconnection point already studied", "effectiveness": 0.65, "cost": "Medium", "relief_months": 9},
    ],
    "water": [
        {"name": "Closed-loop / air cooling redesign", "how": "remove reliance on evaporative water for cooling", "effectiveness": 0.80, "cost": "High", "relief_months": 6},
        {"name": "Reclaimed / greywater sourcing agreement", "how": "contract non-potable water to sidestep groundwater limits", "effectiveness": 0.60, "cost": "Medium", "relief_months": 5},
        {"name": "Warm-water cooling / lower rack density", "how": "raise supply temperature and cut per-rack water demand", "effectiveness": 0.50, "cost": "Medium", "relief_months": 4},
    ],
    "permitting": [
        {"name": "Front-load permitting with a dedicated expediter", "how": "start approvals well before groundbreaking with a local expediter", "effectiveness": 0.75, "cost": "Medium", "relief_months": 9},
        {"name": "Pre-application meetings + phased permits", "how": "de-risk approvals early and permit in stages", "effectiveness": 0.60, "cost": "Low", "relief_months": 6},
        {"name": "Choose a pre-zoned / entitled parcel", "how": "trade site optionality for a faster path through zoning", "effectiveness": 0.70, "cost": "High", "relief_months": 8},
    ],
    "hazard": [
        {"name": "Hardened design (wind / seismic / flood)", "how": "engineer the build to the specific hazard profile", "effectiveness": 0.80, "cost": "High", "relief_months": 3},
        {"name": "Elevate + flood barriers + site grading", "how": "raise critical plant and route water away", "effectiveness": 0.60, "cost": "Medium", "relief_months": 2},
        {"name": "Redundant power + N+1 cooling", "how": "add resilience so a storm event is not an outage", "effectiveness": 0.65, "cost": "High", "relief_months": 2},
    ],
    "power_cost": [
        {"name": "Lock a rate with a PPA / renewable procurement", "how": "sign a power-purchase agreement to fix the price", "effectiveness": 0.65, "cost": "Low", "relief_months": 4},
        {"name": "On-site solar + storage to shave peak", "how": "self-generate to cut peak-demand charges", "effectiveness": 0.60, "cost": "High", "relief_months": 3},
        {"name": "Efficiency + demand response", "how": "higher-ambient cooling and curtailment to lower the bill", "effectiveness": 0.50, "cost": "Medium", "relief_months": 2},
    ],
    "connectivity": [
        {"name": "Build a metro edge PoP / cache", "how": "place latency-sensitive capacity closer to users", "effectiveness": 0.65, "cost": "Medium", "relief_months": 4},
        {"name": "Contract additional dark-fiber routes", "how": "add diverse long-haul paths to the metro", "effectiveness": 0.65, "cost": "High", "relief_months": 4},
        {"name": "Place latency-sensitive workloads elsewhere", "how": "keep only tolerant workloads at this site", "effectiveness": 0.45, "cost": "Low", "relief_months": 3},
    ],
}


# ---------------------------------------------------------------------------
# STEP 3 -- EVALUATE each branch on its axes into one comparable utility, then
# SELECT the winner. Higher effectiveness and more months of relief help; cost
# hurts. This is the "search" of Tree-of-Thoughts, made explicit.
# ---------------------------------------------------------------------------
def _utility(branch: dict) -> float:
    return round(
        branch["effectiveness"] * 100          # how well it works (0-100)
        - _COST_PENALTY[branch["cost"]]         # what it costs us
        + min(branch["relief_months"], 18),     # time it buys back (capped)
        1,
    )


def tree_of_thoughts(site_name: str, llm_propose=None) -> dict:
    """Run the Tree-of-Thoughts mitigation search for one site.

    Returns a dict: {site_id, site_name, top_risk, why, branches (scored,
    ranked), chosen}. `llm_propose` is an optional callable(site, category) ->
    extra branch dict; left None, planning is fully deterministic.
    """
    sites = _load_sites()
    sid = _find_site(sites, site_name)
    if sid is None:
        return {"error": f"No site matching '{site_name}'."}
    site = sites[sid]

    category, why = identify_top_risk(site)
    branches = [dict(b) for b in MITIGATIONS[category]]

    # # LLM-HOOK: an LLM could add one site-specific branch of its own here.
    # We keep it optional so the demo is reproducible without a model.
    if llm_propose is not None:
        try:
            extra = llm_propose(site, category)
            if extra and {"name", "effectiveness", "cost", "relief_months"} <= set(extra):
                extra.setdefault("how", "")
                branches.append(dict(extra))
        except Exception:
            pass

    for b in branches:
        b["utility"] = _utility(b)
    branches.sort(key=lambda b: b["utility"], reverse=True)
    chosen = branches[0]

    return {
        "site_id": sid,
        "site_name": site["name"],
        "top_risk": category,
        "why": why,
        "branches": branches,
        "chosen": chosen,
    }


def render(result: dict) -> str:
    """Format a Tree-of-Thoughts result as text for the trace + the answer."""
    if "error" in result:
        return result["error"]
    lines = [
        f"Mitigation plan for {result['site_name']} ({result['site_id']}).",
        f"Top risk: {result['top_risk']} -- {result['why']}.",
        f"Tree-of-Thoughts explored {len(result['branches'])} options:",
    ]
    for i, b in enumerate(result["branches"], 1):
        mark = "CHOSEN " if b is result["chosen"] else "pruned "
        lines.append(
            f"  [{mark}] {b['name']} -- {b['how']} "
            f"(effectiveness {b['effectiveness']}, cost {b['cost']}, "
            f"~{b['relief_months']} mo relief; utility {b['utility']})"
        )
    c = result["chosen"]
    lines.append(
        f"Recommended mitigation: {c['name']}. It scored highest on the balance "
        f"of effectiveness, cost, and time recovered."
    )
    return "\n".join(lines)
