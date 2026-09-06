"""
course_server.py -- An MCP server (data-center SITING tools).

WHAT IS THIS FILE?
------------------
A completely separate PROGRAM. The agent does not import it. It launches this
file as a subprocess and talks to it over stdin/stdout using JSON-RPC. At
startup the agent asks "what can you do?" (list_tools) and builds its prompt
from the answer, so adding a @server.tool() here means the agent can use it
with ZERO changes to agent.py.

THE DIVISION OF LABOR (the actual lesson of this file)
------------------------------------------------------
Every tool below is deterministic Python: exact sub-scores, a weighted total,
a real chart rendered by matplotlib. An LLM cannot do any of that reliably. It
cannot see the site data and cannot be trusted to average four weighted
criteria in its head.

But look at what the tools DON'T do: none of them can answer "which site should
we actually commit capital to, and what's the one risk to fix first?" That
needs the numbers (tools) plus judgement and the leadership context (memory).
Gluing those into a recommendation is the LLM's half of the job.

    Tools  -> facts, math, artifacts.              (things code does perfectly)
    LLM    -> interpretation, judgement, language.  (things code can't do)

DOMAIN
------
The agent forecasts data-center SITE selection for a cloud region expansion.
Candidate sites live in data/candidate_sites.json; past build closeouts (used
later for retrieval grounding) live in data/past_projects.json. The model has
never seen either file. Tools are the only way in.

HOW A TOOL IS DESCRIBED TO THE MODEL
------------------------------------
The @server.tool() decorator reads each function's TYPE HINTS and DOCSTRING and
auto-generates the JSON schema sent to the agent. The docstring is not a
comment. It is the prompt the model reads to decide whether to call the tool.
Write docstrings like instructions, because that is what they are.

RUN IT STANDALONE (to prove it's a real server):
    python course_server.py
    (it waits for JSON-RPC on stdin -- Ctrl+C to quit)
"""

import json
import statistics
from pathlib import Path
from typing import Optional

from mcp.server import MCPServer

# The server's name. Shows up in client UIs.
server = MCPServer("siting-tools")

DATA_DIR = Path(__file__).parent / "data"
SITES_FILE = DATA_DIR / "candidate_sites.json"
CHARTS_DIR = Path(__file__).parent / "charts"


# ---------------------------------------------------------------------------
# LEADERSHIP PRIORITIES (Module 1)
# The weights and the build-timeline floor live in config.py, so the agent's
# persona and this scoring tool obey ONE source of truth. Re-weight them there
# and both the agent and this server pick up the change on restart.
#
# We import config defensively: if this server is ever run from a directory
# where config.py is not importable, it still works with the same defaults.
# ---------------------------------------------------------------------------
try:
    import config
    CRITERION_WEIGHTS = config.CRITERION_WEIGHTS
    BASE_BUILD_MONTHS = config.BASE_BUILD_MONTHS
except Exception:
    # ---- MODIFY HERE (fallback only; edit config.py, not this) ----
    CRITERION_WEIGHTS = {"power": 0.35, "connectivity": 0.20, "hazard": 0.25, "permitting": 0.20}
    BASE_BUILD_MONTHS = 18


# ---------------------------------------------------------------------------
# DATA LAYER
# ---------------------------------------------------------------------------
# We read a JSON file because it's zero-setup and you can open it in an editor
# to see exactly what the agent sees.
#
# ---- MODIFY HERE ----
# In a real project this is where you'd hit an internal siting database, an
# EIA / ISO power API, NOAA/FEMA hazard feeds, or a GIS service. NOTE: only this
# function changes. The @server.tool() functions below stay identical and the
# agent never knows the difference. That boundary is the whole point of MCP.
def _load_sites() -> dict:
    sites = json.loads(SITES_FILE.read_text(encoding="utf-8"))
    return {s["site_id"]: s for s in sites}


def _find_site(sites: dict, name: str) -> Optional[str]:
    """Forgiving lookup. Small models are sloppy, and a teacher-style user says
    'Quincy' or 'site c', not 'site-c'. Match the id, the full name, any word
    in the name, or a prefix. Returns the canonical site_id or None."""
    name = (name or "").strip().lower()
    if not name:
        return None
    # exact id (accept "site c" / "sitec" for "site-c")
    normalized = name.replace(" ", "-")
    for sid in sites:
        if normalized == sid.lower():
            return sid
    # match against the human name
    for sid, s in sites.items():
        full = s["name"].lower()
        if name == full or name in full.split():
            return sid
    for sid, s in sites.items():
        if s["name"].lower().startswith(name):
            return sid
    # last resort: any word of the query appears in the name
    for sid, s in sites.items():
        if any(w in s["name"].lower() for w in name.split() if len(w) > 2):
            return sid
    return None


def _short(name: str) -> str:
    """'Columbus, OH - Central US expansion' -> 'Columbus' for chart labels."""
    return name.split(",")[0].split(" - ")[0].strip()


# ---------------------------------------------------------------------------
# Deterministic 0-100 sub-scores. This is the math the model must NEVER do
# itself. Each turns one criterion's raw facts into a comparable 0-100 number.
# ---------------------------------------------------------------------------
def _clamp(x: float) -> float:
    return max(0.0, min(100.0, x))


_RISK_LEVEL = {"low": 0, "medium": 1, "high": 2}
_TIER_VALUE = {"low": 0, "medium": 1, "high": 2}


def _power_score(power: dict) -> float:
    # Cheaper $/kWh -> higher (0.03 -> 100, 0.08 -> 0).
    cost = _clamp(100 - (power["cost_per_kwh"] - 0.03) / 0.05 * 100)
    # More deliverable capacity -> higher (400 MW caps it).
    cap = _clamp(power["grid_capacity_mw"] / 400 * 100)
    # Shorter interconnection queue -> higher (30 mo -> 0).
    queue = _clamp(100 - power["interconnection_queue_months"] / 30 * 100)
    # Small bonus for a cleaner grid mix (nice-to-have, not a driver).
    renew = _clamp(power.get("renewable_pct", 0))
    return round(0.4 * cost + 0.2 * cap + 0.35 * queue + 0.05 * renew, 1)


def _hazard_score(hazard: dict) -> float:
    total = sum(_RISK_LEVEL[hazard[k]] for k in ("flood_risk", "seismic_risk", "storm_risk"))
    return round(_clamp(100 - total / 6 * 100), 1)   # 0 risk -> 100, all high -> 0


def _connectivity_score(geo: dict) -> float:
    # Latency dominates (0 ms -> 100, 50 ms -> 0); dense fiber is a small bonus.
    latency = _clamp(100 - geo["latency_ms"] / 50 * 100)
    fiber = _clamp(geo.get("fiber_routes", 0) / 6 * 100)
    return round(0.85 * latency + 0.15 * fiber, 1)


def _permitting_score(permitting: dict) -> float:
    incentive = {"low": 30, "medium": 60, "high": 90}[permitting["incentive_value"]]
    complexity_penalty = {"low": 0, "medium": 20, "high": 40}[permitting["permit_complexity"]]
    # Water is a real siting constraint for cooling. Scarce water is a penalty.
    water_penalty = {"low": 20, "medium": 8, "high": 0}[permitting.get("water_availability", "high")]
    return round(_clamp(incentive - complexity_penalty - water_penalty), 1)


def _score(site: dict) -> dict:
    """The whole deterministic scoring model for one site. Returned as a dict
    so both score_site (text) and chart_sites (bars) can reuse it."""
    subscores = {
        "power":        _power_score(site["power"]),
        "connectivity": _connectivity_score(site["geo"]),
        "hazard":       _hazard_score(site["hazard"]),
        "permitting":   _permitting_score(site["permitting"]),
    }
    wsum = sum(CRITERION_WEIGHTS.values()) or 1.0
    total = sum(subscores[k] * (CRITERION_WEIGHTS[k] / wsum) for k in subscores)

    # Timeline: build floor gated by the slowest dependency (the grid queue).
    timeline = BASE_BUILD_MONTHS + site["power"]["interconnection_queue_months"]

    # Risk tier straight from the weighted score.
    risk_tier = "Low" if total >= 75 else ("Medium" if total >= 55 else "High")

    if total >= 72 and risk_tier != "High":
        rec = "Strong candidate"
    elif total >= 55:
        rec = "Conditional -- mitigate the top risk before committing capital"
    else:
        rec = "Weak / deprioritize"

    return {
        "subscores": subscores,
        "score": round(total, 1),
        "timeline_months": timeline,
        "risk_tier": risk_tier,
        "recommendation": rec,
    }


# ===========================================================================
# DATA TOOLS -- simulate the live sources (EIA power, NOAA/FEMA hazard,
# Maps/Census geo, permitting portals). Here they read the JSON, but the agent
# treats each like a real call to a separate source.
# ===========================================================================
@server.tool()
def get_power(site: str) -> str:
    """Power profile for ONE candidate site: electricity cost, deliverable grid
    capacity, interconnection queue length, and renewable mix.

    Use this whenever the question is about power, electricity cost, grid
    capacity, or how long the utility interconnection will take for a specific
    site. For a full weighted assessment use score_site; for the list of every
    site use list_sites.
    """
    sites = _load_sites()
    sid = _find_site(sites, site)
    if sid is None:
        return f"No site matching '{site}'. Known sites: {', '.join(s['name'] for s in sites.values())}"
    p = sites[sid]["power"]
    return (
        f"Power for {sites[sid]['name']} ({sid}):\n"
        f"  - electricity cost: ${p['cost_per_kwh']}/kWh\n"
        f"  - deliverable grid capacity: {p['grid_capacity_mw']} MW\n"
        f"  - interconnection queue: {p['interconnection_queue_months']} months\n"
        f"  - renewable mix: {p.get('renewable_pct', 'n/a')}%"
    )


@server.tool()
def get_hazard(site: str) -> str:
    """Natural-hazard profile for ONE site: flood, seismic, and storm risk,
    each rated low / medium / high.

    Use this for questions about safety, natural hazards, flooding,
    earthquakes, or storm exposure at a specific site. For the overall
    assessment use score_site.
    """
    sites = _load_sites()
    sid = _find_site(sites, site)
    if sid is None:
        return f"No site matching '{site}'. Known sites: {', '.join(s['name'] for s in sites.values())}"
    h = sites[sid]["hazard"]
    return (
        f"Hazards for {sites[sid]['name']} ({sid}):\n"
        f"  - flood risk: {h['flood_risk']}\n"
        f"  - seismic risk: {h['seismic_risk']}\n"
        f"  - storm risk: {h['storm_risk']}"
    )


@server.tool()
def get_geo(site: str) -> str:
    """Connectivity profile for ONE site: network latency to the target metro,
    distance to that metro, and number of long-haul fiber routes.

    Use this for questions about latency, connectivity, distance to customers,
    or fiber for a specific site. For the overall assessment use score_site.
    """
    sites = _load_sites()
    sid = _find_site(sites, site)
    if sid is None:
        return f"No site matching '{site}'. Known sites: {', '.join(s['name'] for s in sites.values())}"
    g = sites[sid]["geo"]
    return (
        f"Connectivity for {sites[sid]['name']} ({sid}):\n"
        f"  - network latency: {g['latency_ms']} ms\n"
        f"  - distance to major metro: {g['km_to_major_metro']} km\n"
        f"  - long-haul fiber routes: {g.get('fiber_routes', 'n/a')}"
    )


@server.tool()
def get_permitting(site: str) -> str:
    """Permitting profile for ONE site: incentive value, permit complexity, and
    water availability for cooling, each rated low / medium / high.

    Use this for questions about incentives, tax breaks, permitting difficulty,
    regulatory friction, or water/cooling supply at a specific site. For the
    overall assessment use score_site.
    """
    sites = _load_sites()
    sid = _find_site(sites, site)
    if sid is None:
        return f"No site matching '{site}'. Known sites: {', '.join(s['name'] for s in sites.values())}"
    pm = sites[sid]["permitting"]
    return (
        f"Permitting for {sites[sid]['name']} ({sid}):\n"
        f"  - incentive value: {pm['incentive_value']}\n"
        f"  - permit complexity: {pm['permit_complexity']}\n"
        f"  - water availability: {pm.get('water_availability', 'n/a')}"
    )


# ===========================================================================
# COMPUTE TOOL -- the weighted score. Never let the model estimate this.
# ===========================================================================
@server.tool()
def score_site(site: str) -> str:
    """Full weighted assessment for ONE site: a 0-100 score across power,
    connectivity, hazard, and permitting, plus an estimated build timeline in
    months, a risk tier, and a recommendation.

    Use this whenever the teacher asks how good a site is, whether to build
    there, its score, its timeline, or its risk. Never estimate any of these
    numbers yourself -- this tool applies the exact leadership weights to every
    criterion. For one raw criterion use get_power / get_hazard / get_geo /
    get_permitting; to compare every site use list_sites.
    """
    sites = _load_sites()
    sid = _find_site(sites, site)
    if sid is None:
        return f"No site matching '{site}'. Known sites: {', '.join(s['name'] for s in sites.values())}"
    s = sites[sid]
    r = _score(s)
    sub = r["subscores"]
    return (
        f"Assessment for {s['name']} ({sid}):\n"
        f"  sub-scores (0-100): power {sub['power']}, connectivity {sub['connectivity']}, "
        f"hazard {sub['hazard']}, permitting {sub['permitting']}\n"
        f"  weighted score: {r['score']}/100\n"
        f"  estimated timeline: {r['timeline_months']} months "
        f"(18-month build floor + {s['power']['interconnection_queue_months']}-month grid queue)\n"
        f"  risk tier: {r['risk_tier']}\n"
        f"  recommendation: {r['recommendation']}\n"
        f"  weights used: {CRITERION_WEIGHTS}"
    )


# ===========================================================================
# ROSTER -- every site at once, ranked. This tool exists for the same reason
# the gradebook's list_students did: asked "compare all my sites", a model with
# no matching tool will confidently invent a ranking. When an agent fabricates,
# the fix is usually a missing tool, not a better prompt.
# ===========================================================================
@server.tool()
def list_sites() -> str:
    """List EVERY candidate site with its weighted score, timeline, and risk
    tier, ranked best-first -- one line each.

    Use this when the teacher asks for all the sites, the shortlist, the
    ranking, or wants to compare sites. For one site's full breakdown use
    score_site.
    """
    sites = _load_sites()
    ranked = sorted(
        ((sid, s, _score(s)) for sid, s in sites.items()),
        key=lambda t: t[2]["score"],
        reverse=True,
    )
    lines = [f"Candidate sites ranked ({len(sites)} total):"]
    for i, (sid, s, r) in enumerate(ranked, 1):
        lines.append(
            f"  {i}. {s['name']} ({sid}): score {r['score']}/100, "
            f"{r['timeline_months']} mo, {r['risk_tier']} risk -- {r['recommendation']}"
        )
    return "\n".join(lines)


# ===========================================================================
# ARTIFACT TOOL -- returns a PNG, not just text. The LLM cannot draw. It can
# DECIDE a chart is needed and delegate to code that draws one. We return a
# marker line the GUI uses to display the image, plus the plotted numbers as
# text so the model can talk about what's in it.
# ===========================================================================
@server.tool()
def chart_sites(target: str = "all") -> str:
    """Draw a bar chart as a PNG and display it to the teacher.

    Use this whenever the teacher asks for a chart, graph, plot, or any visual
    of the sites. Pass "all" (the default) to chart the weighted score of every
    site side by side (the ranking). Pass a single site's name to chart that
    site's four sub-scores (power, connectivity, hazard, permitting).

    The chart is shown automatically. Never describe the image file itself --
    just summarize what the numbers show.
    """
    import matplotlib
    matplotlib.use("Agg")   # no GUI window; render straight to a file
    import matplotlib.pyplot as plt

    sites = _load_sites()
    CHARTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    single = None if target.strip().lower() in ("all", "sites", "everything", "", "ranking") \
        else _find_site(sites, target)
    if target.strip().lower() not in ("all", "sites", "everything", "", "ranking") and single is None:
        return f"No site matching '{target}'. Known sites: {', '.join(s['name'] for s in sites.values())}"

    if single is None:
        # Ranking chart: weighted score per site.
        ranked = sorted(
            ((s, _score(s)) for s in sites.values()),
            key=lambda t: t[1]["score"], reverse=True,
        )
        labels = [_short(s["name"]) for s, _ in ranked]
        scores = [r["score"] for _, r in ranked]
        colors = ["#2e7d32" if v >= 72 else ("#f9a825" if v >= 55 else "#c62828") for v in scores]
        ax.bar(labels, scores, color=colors)
        ax.set_title("Candidate sites by weighted score")
        ax.set_ylabel("Weighted score /100")
        path = CHARTS_DIR / "site_scores.png"
        plotted = [f"  - {l}: {v}/100" for l, v in zip(labels, scores)]
        plotted.append(
            "Correct summary (repeat faithfully): ranked best-first -> "
            + ", ".join(f"{l} ({v})" for l, v in zip(labels, scores)) + "."
        )
    else:
        # Sub-score profile for one site.
        s = sites[single]
        r = _score(s)
        cats = ["power", "connectivity", "hazard", "permitting"]
        vals = [r["subscores"][c] for c in cats]
        ax.bar(cats, vals, color="#4c72b0")
        ax.set_title(f"{_short(s['name'])} -- sub-scores (weighted total {r['score']}/100)")
        ax.set_ylabel("Sub-score /100")
        path = CHARTS_DIR / f"{single}_subscores.png"
        plotted = [f"  - {c}: {v}/100" for c, v in zip(cats, vals)]
        plotted.append(
            f"Weighted total {r['score']}/100, {r['timeline_months']} months, "
            f"{r['risk_tier']} risk."
        )

    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

    # The GUI watches for this marker line and renders the PNG in the chat.
    return f"CHART_SAVED: {path}\nThe chart plots:\n" + "\n".join(plotted)


if __name__ == "__main__":
    # stdio transport: the client launches this file and pipes JSON-RPC over
    # stdin/stdout. MCP also supports HTTP for remote servers.
    server.run()
