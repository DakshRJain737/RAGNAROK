from __future__ import annotations

import re
import json
import math
import pickle
import logging
import collections
from pathlib import Path
from typing import Optional

import config
from pipeline.schemas import CandidateFeatureVector

logger = logging.getLogger(__name__)


# ── Ontology loader ───────────────────────────────────────────────────────────

def _load_ontology(path: Path) -> dict[str, list[str]]:
    """
    Load skill_map.json as a clean synonym map.

    Expects the JSON to be structured as:
        { "python": ["py", "python3"], "faiss": ["facebook ai similarity search"], ... }

    If the file is missing or malformed, returns an empty dict (graceful degradation).
    """
    if not path.exists():
        logger.warning("skill_map.json not found at '%s'. Ontology expansion disabled.", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Only keep entries where value is a list of strings — skip malformed keys
        ontology = {}
        for key, val in raw.items():
            if isinstance(val, list):
                ontology[key.lower().strip()] = [
                    v.lower().strip() for v in val if isinstance(v, str)
                ]
        logger.info("Loaded ontology: %d skill entries from '%s'", len(ontology), path)
        return ontology
    except Exception as e:
        logger.warning("Failed to parse skill_map.json: %s. Ontology expansion disabled.", e)
        return {}


ONTOLOGY: dict[str, list[str]] = _load_ontology(config.SKILL_MAP_PATH)


# ── Text utilities ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Lowercase and split on word boundaries.
    Keeps alphanumeric tokens plus common tech punctuation (c++, .net, node.js).
    """
    if not text:
        return []
    return re.findall(r'\b[a-z0-9][a-z0-9\+\#\.]*\b', text.lower())


def _expand_query(tokens: list[str]) -> list[str]:
    """
    Expand query tokens using the ontology synonym map.
    Each original token keeps weight 1.0; synonyms are added once (no duplication).

    e.g. ["faiss"] → ["faiss", "facebook ai similarity search", "vector index"]
    """
    expanded = list(tokens)  # start with originals
    seen = set(tokens)

    for token in tokens:
        # Exact key match in ontology
        if token in ONTOLOGY:
            for synonym in ONTOLOGY[token]:
                if synonym not in seen:
                    expanded.append(synonym)
                    seen.add(synonym)

    return expanded


def _build_candidate_text(c: CandidateFeatureVector) -> str:
    """
    Build a single text document per candidate for BM25 indexing.
    Mirrors FaissIndex._build_embedding_text but optimised for keyword matching:
    - Skills are repeated by proficiency weight so expert skills score higher
    - Career titles and industries are emphasised
    - No hard char limit (BM25 handles length via doc_len normalisation)
    """
    parts: list[str] = []

    # Current role — high signal for title matching
    parts.append(c.current_title)
    parts.append(c.current_title)   # repeat for weight
    parts.append(c.current_company)
    parts.append(c.current_industry)

    # Headline + summary
    if c.headline:
        parts.append(c.headline)
    if c.summary:
        parts.append(c.summary)

    # Skills — repeat by proficiency so expert > advanced > intermediate > beginner
    repeat_map = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    for skill in c.skills:
        repeats = repeat_map.get(skill.proficiency, 1)
        parts.extend([skill.name_raw] * repeats)

    # Career history — titles, companies, industries, descriptions
    for job in c.career_history:
        parts.append(job.title)
        parts.append(job.company)
        parts.append(job.industry)
        if job.description:
            parts.append(job.description)

    # Education
    for edu in c.education:
        parts.append(f"{edu.degree} {edu.field_of_study} {edu.institution}")

    # Location
    parts.append(c.location)

    return " ".join(p for p in parts if p and p.strip())


# ── BM25 core ─────────────────────────────────────────────────────────────────

class _BM25Core:
    """
    Pure Okapi BM25 over a pre-tokenized corpus.
    Separated from BM25Index so it can be pickled cleanly.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(corpus)
        self.avg_dl = sum(len(d) for d in corpus) / self.n if self.n else 1.0
        self.dl = [len(d) for d in corpus]

        # df[token] = number of documents containing token
        self.df: dict[str, int] = collections.defaultdict(int)
        # inverted_index[token][doc_idx] = term frequency
        self.inv: dict[str, dict[int, int]] = collections.defaultdict(dict)

        self._build(corpus)

    def _build(self, corpus: list[list[str]]) -> None:
        for idx, doc in enumerate(corpus):
            counts = collections.Counter(doc)
            for token, freq in counts.items():
                self.inv[token][idx] = freq
                self.df[token] += 1

    def score(self, query_tokens: list[str]) -> dict[int, float]:
        scores: dict[int, float] = collections.defaultdict(float)
        for token in query_tokens:
            if token not in self.inv:
                continue
            df = self.df[token]
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)
            for doc_idx, tf in self.inv[token].items():
                dl = self.dl[doc_idx]
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_dl))
                )
                scores[doc_idx] += idf * tf_norm
        return scores


# ── Public class ──────────────────────────────────────────────────────────────

class BM25Index:
    """
    Keyword retrieval index over CandidateFeatureVector.

    API mirrors FaissIndex:
        .build(candidates, save=True)
        .load()
        .search(query_text, top_k) → list[tuple[str, float]]
    """

    def __init__(
        self,
        index_path: Path = config.BM25_INDEX_PATH,
    ) -> None:
        self.index_path = index_path
        self._core: Optional[_BM25Core] = None
        self._id_map: Optional[list[str]] = None   # position → candidate_id

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        candidates: list[CandidateFeatureVector],
        save: bool = True,
    ) -> None:
        """
        Tokenize all candidates and build the BM25 inverted index.

        Args:
            candidates: parsed CandidateFeatureVector list
            save:       persist to disk for reuse
        """
        if not candidates:
            raise ValueError("candidates list is empty — nothing to index.")

        logger.info("Building BM25 index for %d candidates...", len(candidates))

        corpus = [_tokenize(_build_candidate_text(c)) for c in candidates]
        id_map = [c.candidate_id for c in candidates]

        self._core   = _BM25Core(corpus)
        self._id_map = id_map

        logger.info(
            "BM25 index built: %d candidates, %d unique tokens, avg_dl=%.1f",
            len(candidates),
            len(self._core.df),
            self._core.avg_dl,
        )

        if save:
            self._save()

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load pre-built index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at '{self.index_path}'. "
                "Run .build() first."
            )
        with open(self.index_path, "rb") as f:
            payload = pickle.load(f)
        self._core   = payload["core"]
        self._id_map = payload["id_map"]
        logger.info(
            "Loaded BM25 index: %d candidates from '%s'",
            len(self._id_map), self.index_path,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_text: str,
        top_k: int = config.KEYWORD_PATH_TOP_K,
        expand: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Keyword search with optional ontology expansion.

        Args:
            query_text: raw JD text or skill query string
            top_k:      number of results to return
            expand:     whether to expand query via ontology synonyms

        Returns:
            list of (candidate_id, bm25_score) sorted descending
        """
        self._require_loaded()

        tokens = _tokenize(query_text)
        if expand:
            tokens = _expand_query(tokens)

        raw_scores = self._core.score(tokens)

        # Sort by score, map idx → candidate_id, return top_k
        ranked = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = [(self._id_map[idx], float(score)) for idx, score in ranked]

        logger.debug("BM25 top result: %s", results[0] if results else None)
        return results

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._core is not None and self._id_map is not None

    @property
    def vocab_size(self) -> int:
        self._require_loaded()
        return len(self._core.df)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"core": self._core, "id_map": self._id_map}, f)
        logger.info("Saved BM25 index → %s", self.index_path)

    # ── Guards ────────────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                "BM25Index not loaded. Call .build() or .load() first."
            )

    def __repr__(self) -> str:
        status = (
            f"{len(self._id_map)} candidates, {self.vocab_size} tokens"
            if self.is_loaded else "not loaded"
        )
        return f"BM25Index({status})"