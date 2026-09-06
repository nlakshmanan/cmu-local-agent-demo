"""
map_view.py -- the MAP view (presentation only).

Separation of concerns:
  data/candidate_sites.json  -> WHERE the sites are (lat/lon).
  agents.py (Orchestrator)   -> the ANALYSIS (score, timeline, risk, verdict).
  map_view.py (this file)    -> turns those two into a Plotly figure.
  app.py                     -> only wires the figure into a Gradio tab.

This module knows nothing about Gradio and does no analysis of its own. It asks
the agent for records, then draws them. `figure_from_records()` is kept pure (it
takes records in) so it can be tested without running the agent or a model.

Offline by design: Plotly's scatter_geo with scope="usa" draws the base map
from built-in geometry, so there are no map tiles, no Mapbox token, and no
network call. It stays consistent with the project's "100% local" promise.
"""

import json
from pathlib import Path

import plotly.graph_objects as go

_SITES_FILE = Path(__file__).parent / "data" / "candidate_sites.json"
_RISK_COLOR = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}


def _load_sites() -> dict:
    return {s["site_id"]: s for s in json.loads(_SITES_FILE.read_text(encoding="utf-8"))}


def figure_from_records(records: list[dict]):
    """Build the map figure from agent records. Pure: no agent, no model.

    Each record needs site_id, site_name, score, timeline_months, risk_tier,
    recommendation, and verdict. Coordinates are joined in from the data file.
    """
    sites = _load_sites()
    lats, lons, colors, sizes, text = [], [], [], [], []
    for r in records:
        s = sites.get(r["site_id"])
        if not s or "lat" not in s or "lon" not in s:
            continue
        lats.append(s["lat"])
        lons.append(s["lon"])
        colors.append(_RISK_COLOR.get(r["risk_tier"], "#888888"))
        sizes.append(12 + s["power"]["grid_capacity_mw"] / 20)  # bigger = more capacity
        text.append(
            f"<b>{r['site_name']}</b><br>"
            f"score {r['score']}/100 · {r['timeline_months']} mo · {r['risk_tier']} risk<br>"
            f"{r['recommendation']}<br>"
            f"critic: {r['verdict']}"
        )

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, text=text, hoverinfo="text", mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="white"),
                    sizemode="diameter"),
    ))
    fig.update_layout(
        geo=dict(scope="usa", bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        title=dict(text="Candidate sites (color = risk tier, size = grid capacity)",
                   x=0.5, font=dict(size=12)),
    )
    return fig


def build_map():
    """Run the agent over every site, then draw the map. This is what the UI calls."""
    from agents import Orchestrator          # imported here so tests can skip the agent
    records = Orchestrator().evaluate_all()
    return figure_from_records(records)
