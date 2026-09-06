"""
agents.py -- MODULE 5: multi-agent coordination.

The chat agent in agent.py is ONE generalist doing everything: gather, score,
answer. That is fine for a conversation. For an autonomous, auditable pipeline
it helps to split the work across specialists that hand structured results to
each other, so each has one job and the handoffs are inspectable.

    Researcher  -> gathers the raw facts for a site (calls the data tools),
                   and notices when a source fails.
    Analyst     -> turns those facts into a grounded forecast (score + precedent)
                   and a mitigation plan (Tree-of-Thoughts).
    Critic      -> reviews the Analyst's output and returns a verdict:
                   APPROVE, REVIEW, or FLAG, with reasons and a confidence.
    Orchestrator-> runs the three in order, passes results along, and can sweep
                   every site into a ranked shortlist.

This reuses everything already built: the MCP tools and helpers in
course_server.py, the retrieval in retrieval.py, and the reasoning in
reasoning.py. Coordination is the new idea here, not new domain logic.

The Critic is deliberately a separate role. A generator that also grades its
own work grades it generously. A separate reviewer with its own rules is how
you catch an over-confident recommendation before it reaches a human -- and it
is the natural place the guardrails (Task 9) and the human-review gate
(Task 10) plug in.

Run standalone:  python agents.py
"""

import course_server as srv     # reuse the MCP tools + helpers (M1-4)
import reasoning                 # Tree-of-Thoughts (M4)

_SOURCES = ("power", "hazard", "geo", "permitting")


def _trace(store: list, verbose: bool, msg: str):
    store.append(msg)
    if verbose:
        print(msg)


# ---------------------------------------------------------------------------
# ROLE 1 -- RESEARCHER: gather the facts, notice failures.
# ---------------------------------------------------------------------------
class Researcher:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def run(self, site_id: str, fail_source: str = None) -> dict:
        """Call each data tool for the site. `fail_source` simulates one source
        failing on the first attempt, so the pipeline can show recovery."""
        sites = srv._load_sites()
        site = sites[site_id]
        trace, flags, sources = [], [], {}
        _trace(trace, self.verbose, f"[Researcher] gathering facts for {site['name']} ({site_id})")

        tool_of = {"power": srv.get_power, "hazard": srv.get_hazard,
                   "geo": srv.get_geo, "permitting": srv.get_permitting}
        for src in _SOURCES:
            if fail_source == src:
                # Simulated outage: do NOT invent data. Flag it and retry once.
                _trace(trace, self.verbose, f"[Researcher] {src} source FAILED -- not guessing; retrying once")
                flags.append(f"{src} source failed on first attempt; used retry value")
            tool_of[src](site_id)   # the real MCP tool call (also proves it works)
            sources[src] = "ok"
            _trace(trace, self.verbose, f"[Researcher] {src}: ok")

        return {"site_id": site_id, "site_name": site["name"], "site": site,
                "sources": sources, "flags": flags, "trace": trace}


# ---------------------------------------------------------------------------
# ROLE 2 -- ANALYST: forecast + mitigation from the gathered facts.
# ---------------------------------------------------------------------------
class Analyst:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def run(self, research: dict) -> dict:
        trace = []
        site = research["site"]
        _trace(trace, self.verbose, f"[Analyst] scoring + grounding {research['site_name']}")
        forecast = srv.grounded_forecast(site)          # M1-3: score + precedent
        _trace(trace, self.verbose,
               f"[Analyst] score {forecast['base']['score']} -> {forecast['score']}/100, "
               f"{forecast['timeline_months']} mo, {forecast['risk_tier']} risk "
               f"({len(forecast['hits'])} precedent(s))")
        mitigation = reasoning.tree_of_thoughts(research["site_id"])   # M4
        _trace(trace, self.verbose,
               f"[Analyst] top risk '{mitigation['top_risk']}' -> "
               f"mitigation '{mitigation['chosen']['name']}'")
        return {"forecast": forecast, "mitigation": mitigation, "trace": trace}


# ---------------------------------------------------------------------------
# ROLE 3 -- CRITIC: an independent reviewer. Its rules are the seed of the
# guardrail layer (Task 9) and the human-review gate (Task 10).
# ---------------------------------------------------------------------------
class Critic:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def run(self, research: dict, analysis: dict) -> dict:
        trace, reasons, serious = [], [], False
        forecast = analysis["forecast"]

        if research["flags"]:
            reasons.append("a data source failed on first fetch (used a retry)")
            serious = True

        if not forecast["hits"]:
            reasons.append("no comparable precedent -- timeline/risk are live-data-only")

        # A 'Strong' label must be earned. It should never survive High risk or
        # a missing precedent; if it somehow does, that is a serious defect.
        if forecast["recommendation"] == "Strong candidate" and \
                (forecast["risk_tier"] == "High" or not forecast["hits"]):
            reasons.append("'Strong candidate' not fully supported (high risk or ungrounded)")
            serious = True

        # A score sitting on a recommendation boundary is fragile.
        if any(abs(forecast["score"] - edge) <= 2 for edge in (55, 72)):
            reasons.append(f"score {forecast['score']} sits on a recommendation boundary")

        if not reasons:
            verdict, confidence = "APPROVE", "high"
        elif serious:
            verdict, confidence = "FLAG", "low"
        else:
            verdict, confidence = "REVIEW", "medium"

        _trace(trace, self.verbose,
               f"[Critic] verdict {verdict} (confidence {confidence})"
               + (f" -- {'; '.join(reasons)}" if reasons else ""))
        return {"verdict": verdict, "confidence": confidence, "reasons": reasons, "trace": trace}


# ---------------------------------------------------------------------------
# ORCHESTRATOR -- coordinates the three roles and sweeps every site.
# ---------------------------------------------------------------------------
class Orchestrator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.researcher = Researcher(verbose)
        self.analyst = Analyst(verbose)
        self.critic = Critic(verbose)

    def evaluate_site(self, site_id: str, fail_source: str = None) -> dict:
        research = self.researcher.run(site_id, fail_source=fail_source)
        analysis = self.analyst.run(research)
        review = self.critic.run(research, analysis)
        f = analysis["forecast"]
        return {
            "site_id": site_id,
            "site_name": research["site_name"],
            "score": f["score"],
            "timeline_months": f["timeline_months"],
            "risk_tier": f["risk_tier"],
            "recommendation": f["recommendation"],
            "top_risk": analysis["mitigation"]["top_risk"],
            "mitigation": analysis["mitigation"]["chosen"]["name"],
            "verdict": review["verdict"],
            "confidence": review["confidence"],
            "reasons": review["reasons"],
            "flags": research["flags"],
            "trace": research["trace"] + analysis["trace"] + review["trace"],
        }

    def evaluate_all(self) -> list[dict]:
        records = [self.evaluate_site(sid) for sid in srv._load_sites()]
        records.sort(key=lambda r: r["score"], reverse=True)
        return records


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_site(rec: dict) -> str:
    lines = [
        f"{rec['site_name']} ({rec['site_id']})",
        f"  score {rec['score']}/100, {rec['timeline_months']} mo, {rec['risk_tier']} risk -- {rec['recommendation']}",
        f"  top risk: {rec['top_risk']}; recommended mitigation: {rec['mitigation']}",
        f"  critic: {rec['verdict']} (confidence {rec['confidence']})"
        + (f" -- {'; '.join(rec['reasons'])}" if rec["reasons"] else ""),
    ]
    return "\n".join(lines)


def render_ranking(records: list[dict]) -> str:
    lines = [f"Ranked shortlist ({len(records)} sites):"]
    for i, r in enumerate(records, 1):
        lines.append(f"  {i}. {r['site_name']} ({r['site_id']}): {r['score']}/100, "
                     f"{r['risk_tier']} risk, critic {r['verdict']} -- {r['recommendation']}")
    return "\n".join(lines)


if __name__ == "__main__":
    orch = Orchestrator(verbose=True)
    print("\n=== One site end-to-end (with a simulated hazard-source failure) ===")
    rec = orch.evaluate_site("site-c", fail_source="hazard")
    print("\n" + render_site(rec))
    print("\n=== Full autonomous sweep ===")
    print(render_ranking(orch.evaluate_all()))
