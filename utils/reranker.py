"""
Re-Ranker
=========
Uses a Cross-Encoder model to re-rank retrieved documents,
improving precision over bi-encoder retrieval alone.

Pipeline:
  Bi-encoder retrieval (fast, approximate)  →  Cross-encoder re-ranking (slow, precise)
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-Encoder re-ranker using sentence-transformers.
    
    The cross-encoder jointly encodes the (query, document) pair and 
    outputs a relevance score, making it more accurate than bi-encoder 
    cosine similarity alone.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model     = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info(f"Re-ranker loaded: {self.model_name}")
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            logger.warning(f"Re-ranker load failed ({e}), will use score pass-through")
            self._model = None

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents by cross-encoder relevance score.

        Args:
            query:     The user query string.
            documents: List of dicts with at least a 'text' key.
            top_k:     Number of top documents to return.

        Returns:
            Sorted list of documents with added 'rerank_score' key.
        """
        if not documents:
            return []

        if self._model is None:
            # Fallback: return original order
            logger.warning("Re-ranker unavailable, returning original ranking")
            return documents[:top_k]

        pairs  = [(query, doc["text"]) for doc in documents]
        scores = self._model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = round(float(score), 4)

        ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        logger.info(
            f"Re-ranked {len(documents)} docs → top {top_k} "
            f"| best_score={ranked[0]['rerank_score'] if ranked else 'N/A'}"
        )
        return ranked[:top_k]

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model":  self.model_name,
            "status": "loaded" if self._model else "fallback"
        }
