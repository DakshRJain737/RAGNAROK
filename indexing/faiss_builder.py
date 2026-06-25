from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import config

from pipeline.schemas import CandidateFeatureVector

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = config.BI_ENCODER_MODEL
EMBEDDING_DIM = config.EMBEDDING_DIM
N_CLUSTERS    = config.FAISS_NLIST
N_PROBE       = config.FAISS_NPROBE
BATCH_SIZE    = 128
MAX_TEXT_CHARS = 2000

INDEX_PATH  = config.FAISS_INDEX_PATH
ID_MAP_PATH = config.FAISS_ID_MAP_PATH

# Proficiency buckets as frozensets — O(1) lookup vs tuple/list `in`
_EXPERT_SET       = frozenset(("advanced", "expert"))
_INTERMEDIATE_SET = frozenset(("intermediate",))


class FaissIndex:
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        index_path: Path = INDEX_PATH,
        id_map_path: Path = ID_MAP_PATH,
    ) -> None:
        self.model_name  = model_name
        self.index_path  = index_path
        self.id_map_path = id_map_path

        self._model: Optional[SentenceTransformer] = None  # lazy loaded
        self._index: Optional[faiss.Index] = None
        self._id_map: Optional[list[str]] = None           # position → candidate_id

    def build(self, candidates: list[CandidateFeatureVector], save: bool = True) -> None:
      
        if not candidates:
            raise ValueError("candidates list is empty — nothing to index.")

        logger.info("Building FAISS index for %d candidates...", len(candidates))

        texts  = [c.embedding_text for c in candidates]
        id_map = [c.candidate_id for c in candidates]

        embeddings = self._encode_batch(texts)

        if len(candidates) >= N_CLUSTERS:
            index = self._build_ivf_index(embeddings)
            logger.info("Built IVF256 index (%d vectors)", index.ntotal)
        else:
            index = self._build_flat_index(embeddings)
            logger.warning(
                "Candidate pool (%d) < N_CLUSTERS (%d). "
                "Using IndexFlatIP (exact search). Switch to IVF for production.",
                len(candidates), N_CLUSTERS,
            )

        self._index  = index
        self._id_map = id_map

        if save:
            self._save(index, id_map)

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{self.index_path}'. Run .build() first."
            )
        self._index = faiss.read_index(str(self.index_path))
        self._index.nprobe = N_PROBE
        with open(self.id_map_path, "rb") as f:
            self._id_map = pickle.load(f)
        logger.info(
            "Loaded FAISS index: %d vectors from '%s'",
            self._index.ntotal, self.index_path,
        )

    def search(self, query_text: str, top_k: int = 100) -> list[tuple[str, float]]:
       
        self._require_loaded()

        # Bind model to local — avoids repeated _get_model() attr chain in hot path
        model = self._get_model()
        query_vec = model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # encode() already returns np.ndarray — reshape in-place, no copy
        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        scores, indices = self._index.search(query_vec, top_k)

        results = []
        id_map = self._id_map
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 when fewer results exist
                continue
            results.append((id_map[idx], float(score)))

        logger.debug("semantic_search top result: %s", results[0] if results else None)
        return results

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and self._id_map is not None

    @property
    def total_vectors(self) -> int:
        self._require_loaded()
        return self._index.ntotal

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        logger.info("Encoding %d candidates (batch_size=%d)...", len(texts), BATCH_SIZE)
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,   # ensures ndarray directly — skips internal tensor copy
        )
        # encode() with convert_to_numpy=True already returns float32 ndarray
        return embeddings.astype(np.float32, copy=False)

    # ── Index constructors ────────────────────────────────────────────────────

    @staticmethod
    def _build_ivf_index(embeddings: np.ndarray) -> faiss.IndexIVFFlat:
        quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
        index = faiss.IndexIVFFlat(
            quantizer, EMBEDDING_DIM, N_CLUSTERS, faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = N_PROBE
        return index

    @staticmethod
    def _build_flat_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)
        return index

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, index: faiss.Index, id_map: list[str]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        np.save(str(self.id_map_path), np.array(id_map, dtype=object))
        logger.info("Saved index → %s  |  id_map → %s", self.index_path, self.id_map_path)

    # ── Model cache ───────────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading sentence transformer: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
        return self._model

    # ── Guards ────────────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("Index not loaded. Call .build() or .load() first.")

    def __repr__(self) -> str:
        status = f"{self._index.ntotal} vectors" if self.is_loaded else "not loaded"
        return f"FaissIndex(model={self.model_name}, status={status})"