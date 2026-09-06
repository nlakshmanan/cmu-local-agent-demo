"""
retrieval.py -- MODULE 3: RAG / retrieval grounding (long-term DOMAIN memory).

Long-term MEMORY (memory.py) is what the USER told us. This file is different:
it is the company's own history -- past data-center build closeouts in
data/past_projects.json -- retrieved by similarity to ground a NEW site's
forecast in what actually happened on comparable builds.

The whole RAG pipeline, in one small module:

    documents -> CHUNK -> EMBED -> SCORE by similarity -> top-k above threshold

Then precedent_signal_from() turns the retrieved text into a NUMBER (extra
months, a score penalty, a risk bump). That number is HOW retrieval changes the
decision, not just decorates it with a citation.

EMBEDDINGS
----------
Default is a dependency-free LEXICAL retriever (a tiny TF-IDF cosine in pure
Python) so the project runs offline and is easy to read. TF-IDF is *lexical* --
it matches shared words. For true *semantic* search, set USE_EMBEDDINGS = True
and it reuses the same nomic-embed-text model memory.py already uses. Only the
Embedder changes; the rest of the pipeline is identical. That swap is the whole
Module-3 point: RAG is modular.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PAST_FILE = DATA_DIR / "past_projects.json"

# ---- MODIFY HERE ----------------------------------------------------------
# How many precedents to retrieve, and the similarity floor below which a match
# counts as "no evidence". The two retrievers live on different scales, so each
# gets its own floor. Set a floor to 0.0 to watch the failure mode: irrelevant
# precedents get dragged into every forecast.
TOP_K = 3
LEXICAL_THRESHOLD = 0.25
EMBED_THRESHOLD = 0.55

# Flip to True to use real semantic embeddings (needs Ollama running with
# nomic-embed-text pulled). False = offline lexical TF-IDF, no dependencies.
USE_EMBEDDINGS = False
EMBED_MODEL = "nomic-embed-text"
# ---------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "was", "were", "is", "are", "be", "been", "it", "its", "as",
    "that", "this", "these", "those", "but", "not", "no", "than", "then", "so",
    "up", "out", "off", "about", "into", "over", "under", "per", "we", "our",
}


def _load_corpus() -> list[dict]:
    return json.loads(PAST_FILE.read_text(encoding="utf-8"))


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 2 and w not in _STOP]


# ---------------------------------------------------------------------------
# CHUNKING. Each closeout report is only a few sentences, so one report = one
# chunk. (If these documents were long, we'd split them into overlapping
# windows here so one chunk carries one coherent lesson with its reason.)
# ---------------------------------------------------------------------------
def _chunk_of(doc: dict) -> str:
    # Region is part of the searchable text so same-region builds rank higher.
    return f"{doc['title']} {doc['region']} {doc['text']}"


# ---------------------------------------------------------------------------
# EMBEDDERS (pluggable). Default = lexical TF-IDF; optional = Ollama embeddings.
# ---------------------------------------------------------------------------
class _LexicalEmbedder:
    """A tiny TF-IDF cosine in pure Python. No numpy, no sklearn, no network."""

    def __init__(self, corpus: list[str]):
        self.docs_tokens = [_tokens(c) for c in corpus]
        n = len(self.docs_tokens)
        df = Counter()
        for toks in self.docs_tokens:
            for w in set(toks):
                df[w] += 1
        # Smoothed inverse document frequency.
        self.idf = {w: math.log((n + 1) / (df_w + 1)) + 1.0 for w, df_w in df.items()}
        self.doc_vecs = [self._vec(toks) for toks in self.docs_tokens]

    def _vec(self, toks: list[str]) -> dict:
        tf = Counter(toks)
        total = sum(tf.values()) or 1
        return {w: (c / total) * self.idf.get(w, 0.0) for w, c in tf.items()}

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[w] * b[w] for w in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb + 1e-9)

    def similarities(self, query_text: str) -> list[float]:
        q = self._vec(_tokens(query_text))
        return [self._cosine(q, d) for d in self.doc_vecs]


class _OllamaEmbedder:
    """Real semantic embeddings. Reuses the same model memory.py uses."""

    def __init__(self, corpus: list[str]):
        import ollama  # lazy import so lexical mode needs nothing
        self._ollama = ollama
        self.matrix = [self._embed(c) for c in corpus]

    def _embed(self, text: str) -> list[float]:
        return self._ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-9)

    def similarities(self, query_text: str) -> list[float]:
        q = self._embed(query_text)
        return [self._cosine(q, d) for d in self.matrix]


# ---------------------------------------------------------------------------
# THE RETRIEVER
# ---------------------------------------------------------------------------
class PrecedentRetriever:
    def __init__(self, use_embeddings: bool = USE_EMBEDDINGS):
        self.docs = _load_corpus()
        corpus = [_chunk_of(d) for d in self.docs]
        if use_embeddings:
            try:
                self.embedder = _OllamaEmbedder(corpus)
                self.threshold = EMBED_THRESHOLD
                self.mode = "semantic"
                return
            except Exception:
                # Ollama not reachable -> fall back to lexical rather than crash.
                pass
        self.embedder = _LexicalEmbedder(corpus)
        self.threshold = LEXICAL_THRESHOLD
        self.mode = "lexical"

    def query(self, query_text: str, top_k: int = TOP_K, threshold: float = None) -> list[dict]:
        """Rank every past project against the query, keep top_k above the floor."""
        threshold = self.threshold if threshold is None else threshold
        sims = self.embedder.similarities(query_text)
        order = sorted(range(len(self.docs)), key=lambda i: sims[i], reverse=True)
        hits = []
        for i in order[:top_k]:
            if sims[i] < threshold:      # FAILURE GUARD: weak match == no evidence
                continue
            hit = dict(self.docs[i])
            hit["similarity"] = round(float(sims[i]), 3)
            hits.append(hit)
        return hits


# ---------------------------------------------------------------------------
# Turn retrieved precedents into a NUMBER the forecast uses.
# This is the bridge from "retrieved text" to "changed decision".
# ---------------------------------------------------------------------------
def precedent_signal_from(hits: list[dict]) -> dict:
    """
    If comparable builds OVERRAN, produce a penalty:
      - add the average grid-slip months to the timeline
      - dock the score
      - bump the risk tier
    If the only comparable builds came in ON TIME, return a small reassurance
    note with no numeric change. If nothing was retrieved, return {} and the
    caller reports 'no comparable precedent' and scores on live data only.
    """
    if not hits:
        return {}

    overruns = [h for h in hits if h.get("outcome") == "overran"]
    if not overruns:
        sources = ", ".join(h["title"] for h in hits)
        return {
            "timeline_add_months": 0,
            "score_penalty": 0,
            "risk_bump": 0,
            "reason": (f"{len(hits)} comparable build(s) came in on time "
                       f"(source: {sources}) -- no timeline penalty applied."),
        }

    avg_slip = round(sum(h["grid_slip_months"] for h in overruns) / len(overruns))
    sources = ", ".join(sorted({h["title"] for h in overruns}))
    return {
        "timeline_add_months": avg_slip,
        "score_penalty": min(20, 6 + avg_slip),   # bounded penalty
        "risk_bump": 1,
        "reason": (f"{len(overruns)} comparable build(s) overran on grid "
                   f"energization by ~{avg_slip} months "
                   f"(source: {sources})."),
    }
