"""
Proposal Corpus Service
=======================

Lightweight retrieval over the historical (Job Post -> Proposal) corpus.
Used by the "Find Similar Past Proposal" feature on the main screen.

Storage backend (in priority order):
  1. MongoDB collection `proposal_corpus`  -> production / shared (Option B)
  2. data/proposal_corpus.json             -> local / dev fallback

Embeddings are stored alongside each item (in Mongo or in a sidecar JSON file)
so they are computed exactly once per row. Cosine similarity is done in-memory
(corpus is small).
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CORPUS_PATH = os.path.join("data", "proposal_corpus.json")
EMB_CACHE_PATH = os.path.join("data", "proposal_corpus.embeddings.json")
EMBEDDING_DIM = 3072  # text-embedding-3-large
MONGO_COLLECTION = "proposal_corpus"


# Confidence thresholds tuned for cosine similarity on text-embedding-3-large.
# We deliberately keep these HIGH so only near-exact matches surface—
# anything below WEAK_THRESHOLD is treated as "no usable match".
STRONG_THRESHOLD = 0.72   # >= this => "strong" (near-exact requirement match)
WEAK_THRESHOLD = 0.62     # >= this and < strong => "weak" (related but not exact)
# Below WEAK_THRESHOLD => no usable match, return empty


class ProposalCorpusService:
    _instance: Optional["ProposalCorpusService"] = None
    _lock = threading.Lock()

    def __init__(self, openai_service):
        self.openai_service = openai_service
        self.items: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []  # aligned with self.items
        self.backend: str = "none"  # "mongo" or "file"
        self._loaded = False

    # ------------------------------------------------------------------ public

    @classmethod
    def get_instance(cls, openai_service) -> "ProposalCorpusService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(openai_service)
            return cls._instance

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self._try_load_from_mongo():
                self.backend = "mongo"
            else:
                self._load_from_file()
                self.backend = "file"
            self._loaded = True
            logger.info(f"[ProposalCorpus] Ready: {len(self.items)} items (backend={self.backend})")

    def find_similar(
        self,
        title: str,
        description: str,
        skills: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Return the most similar past (job_post, proposal) records for the
        provided new job description. Matches are scored with cosine similarity.
        """
        self.ensure_loaded()

        if not self.items:
            return {"count": 0, "best_match": None, "matches": [], "message": "Corpus empty"}

        query_text = self._build_query_text(title, description, skills or [])
        if not query_text.strip():
            return {"count": 0, "best_match": None, "matches": [], "message": "Empty query"}

        try:
            q_vec = self.openai_service.get_embedding(query_text, dimensions=EMBEDDING_DIM)
        except Exception as e:
            logger.error(f"[ProposalCorpus] Embedding failed: {e}")
            return {"count": 0, "best_match": None, "matches": [], "message": "Embedding failed"}

        scored: List[Tuple[float, int]] = []
        # Lowercased query skills for soft re-ranking (secondary signal)
        q_skills = {s.strip().lower() for s in (skills or []) if s and s.strip()}
        for idx, vec in enumerate(self.embeddings):
            if not vec:
                continue
            sim = self._cosine(q_vec, vec)
            # Skills act ONLY as a tiny tie-breaker boost. They never push a
            # weak semantic match into the "usable" zone on their own.
            if q_skills:
                doc_skills = {s.strip().lower() for s in (self.items[idx].get("skills") or [])}
                overlap = len(q_skills & doc_skills)
                if overlap:
                    sim = min(1.0, sim + 0.005 * overlap)
            scored.append((sim, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(1, top_k)]

        matches: List[Dict[str, Any]] = []
        for score, idx in top:
            item = self.items[idx]
            matches.append({
                "similarity_score": round(float(score), 4),
                "confidence": self._confidence_label(score),
                "id": item.get("id"),
                "title": item.get("title"),
                "industry": item.get("industry"),
                "platform": item.get("platform"),
                "task_type": item.get("task_type"),
                "skills": item.get("skills", []),
                "job_post": item.get("job_post"),
                "proposal": item.get("proposal"),
                "feedback": item.get("feedback"),
            })

        best = matches[0] if matches else None
        usable = best and best["similarity_score"] >= WEAK_THRESHOLD

        return {
            "count": len(matches),
            "best_match": best if usable else None,
            "matches": matches,
            "threshold": {"strong": STRONG_THRESHOLD, "weak": WEAK_THRESHOLD},
            "has_match": bool(usable),
        }

    # ----------------------------------------------------------------- internal

    def _build_query_text(self, title: str, description: str, skills: List[str]) -> str:
        # Description-primary matching: we embed ONLY the job description so
        # the comparison axis matches the corpus side (which embeds only the
        # historical job_post). Title and skills are excluded from embedding
        # text; skills act as a soft tie-breaker downstream.
        return (description or "").strip()

    def _build_doc_text(self, item: Dict[str, Any]) -> str:
        return (item.get("job_post") or "").strip()

    # ------ Mongo backend (Option B - primary) ------------------------------

    def _try_load_from_mongo(self) -> bool:
        try:
            from app.db import get_db
            db_mgr = get_db()
            if db_mgr is None or db_mgr.db is None:
                return False
            coll = db_mgr.db[MONGO_COLLECTION]
            count = coll.count_documents({})
            if count == 0:
                logger.info(f"[ProposalCorpus] Mongo collection '{MONGO_COLLECTION}' empty; falling back to file")
                return False
        except Exception as e:
            logger.info(f"[ProposalCorpus] Mongo unavailable, using file fallback: {e}")
            return False

        # Load all items
        docs = list(coll.find({}, {"_id": 0}))
        items: List[Dict[str, Any]] = []
        embeddings: List[List[float]] = []
        to_embed: List[Tuple[int, str]] = []  # (index, text)

        for doc in docs:
            items.append({
                "id": doc.get("id"),
                "title": doc.get("title", ""),
                "company": doc.get("company", ""),
                "platform": doc.get("platform", ""),
                "task_type": doc.get("task_type", ""),
                "industry": doc.get("industry", ""),
                "skills": doc.get("skills", []) or [],
                "job_post": doc.get("job_post", ""),
                "proposal": doc.get("proposal", ""),
                "feedback": doc.get("feedback", ""),
                "content_hash": doc.get("content_hash", ""),
            })
            vec = doc.get("embedding")
            embeddings.append(vec if vec else None)
            if not vec:
                to_embed.append((len(items) - 1, self._build_doc_text(items[-1])))

        # Embed any missing rows on demand and write back to Mongo
        if to_embed:
            logger.info(f"[ProposalCorpus] Embedding {len(to_embed)} new/changed rows in Mongo...")
            texts = [t for _, t in to_embed]
            try:
                vecs = self.openai_service.get_embeddings_batch(texts, dimensions=EMBEDDING_DIM)
            except Exception as e:
                logger.error(f"[ProposalCorpus] Batch embed failed: {e}")
                vecs = [None] * len(texts)

            for (idx, _), vec in zip(to_embed, vecs):
                if not vec:
                    continue
                embeddings[idx] = vec
                item_id = items[idx].get("id")
                if item_id:
                    try:
                        coll.update_one(
                            {"id": item_id},
                            {"$set": {"embedding": vec, "embedded_at_dim": EMBEDDING_DIM}},
                        )
                    except Exception as e:
                        logger.warning(f"[ProposalCorpus] Failed writing embedding for {item_id}: {e}")

        self.items = items
        self.embeddings = [v if v else [0.0] * EMBEDDING_DIM for v in embeddings]
        return True

    # ------ File backend (dev / fallback) ----------------------------------

    def _load_from_file(self) -> None:
        if not os.path.exists(CORPUS_PATH):
            logger.warning(f"[ProposalCorpus] Missing corpus file: {CORPUS_PATH}")
            self.items = []
            self.embeddings = []
            return
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = data.get("items", []) or []
        logger.info(f"[ProposalCorpus] Loaded {len(self.items)} items from {CORPUS_PATH}")

        cache: Dict[str, List[float]] = {}
        if os.path.exists(EMB_CACHE_PATH):
            try:
                with open(EMB_CACHE_PATH, "r", encoding="utf-8") as f:
                    cache = json.load(f).get("vectors", {}) or {}
            except Exception as e:
                logger.warning(f"[ProposalCorpus] Cache read failed: {e}")
                cache = {}

        missing_indices: List[int] = []
        missing_texts: List[str] = []
        embeddings: List[Optional[List[float]]] = [None] * len(self.items)

        for i, item in enumerate(self.items):
            h = item.get("content_hash")
            if h and h in cache:
                embeddings[i] = cache[h]
            else:
                missing_indices.append(i)
                missing_texts.append(self._build_doc_text(item))

        if missing_texts:
            logger.info(f"[ProposalCorpus] Embedding {len(missing_texts)} new items (file backend)...")
            new_vecs = self.openai_service.get_embeddings_batch(
                missing_texts, dimensions=EMBEDDING_DIM
            )
            for idx, vec in zip(missing_indices, new_vecs):
                if vec:
                    embeddings[idx] = vec
                    h = self.items[idx].get("content_hash")
                    if h:
                        cache[h] = vec
            self._save_file_cache(cache)

        self.embeddings = [v if v else [0.0] * EMBEDDING_DIM for v in embeddings]

    def _save_file_cache(self, cache: Dict[str, List[float]]) -> None:
        try:
            os.makedirs(os.path.dirname(EMB_CACHE_PATH), exist_ok=True)
            with open(EMB_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "vectors": cache}, f)
            logger.info(f"[ProposalCorpus] Saved {len(cache)} vectors -> {EMB_CACHE_PATH}")
        except Exception as e:
            logger.error(f"[ProposalCorpus] Cache save failed: {e}")

    # ----------------------------------------------------------- math helpers

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= STRONG_THRESHOLD:
            return "strong"
        if score >= WEAK_THRESHOLD:
            return "weak"
        return "none"
