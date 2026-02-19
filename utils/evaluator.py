"""
RAG Evaluator
=============
Implements RAGAS-inspired evaluation metrics without requiring the full RAGAS library:

  1. Faithfulness       – Does the answer stick to the retrieved context?
  2. Answer Relevancy   – How relevant is the answer to the question?
  3. Context Precision  – Are retrieved chunks actually useful?
  4. Context Recall     – Does context cover the ground truth?
  5. Answer Correctness – How close is the answer to the ground truth? (if provided)
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Lightweight RAGAS-style evaluator using lexical + heuristic metrics.
    Can optionally use an LLM for more nuanced faithfulness scoring.
    """

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run all evaluation metrics.

        Returns:
            Dict with individual metric scores (0–1) and an aggregate score.
        """
        scores = {}

        scores["faithfulness"]      = self._faithfulness(answer, contexts)
        scores["answer_relevancy"]  = self._answer_relevancy(question, answer)
        scores["context_precision"] = self._context_precision(question, contexts)
        scores["context_recall"]    = self._context_recall(contexts, ground_truth) if ground_truth else None

        if ground_truth:
            scores["answer_correctness"] = self._answer_correctness(answer, ground_truth)

        # Aggregate (exclude None values)
        valid = [v for v in scores.values() if v is not None]
        scores["aggregate"] = round(sum(valid) / len(valid), 4) if valid else 0.0

        # Qualitative label
        agg = scores["aggregate"]
        if agg >= 0.8:
            scores["quality_label"] = "Excellent ✅"
        elif agg >= 0.6:
            scores["quality_label"] = "Good 🟡"
        elif agg >= 0.4:
            scores["quality_label"] = "Fair 🟠"
        else:
            scores["quality_label"] = "Poor 🔴"

        logger.info(f"Evaluation complete: {scores}")
        return scores

    # ── Metric Implementations ─────────────────────────────────────────────────

    def _faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Measures how much of the answer is grounded in the contexts.
        Heuristic: overlap of answer sentences with any context chunk.
        """
        if not contexts or not answer:
            return 0.0

        answer_sentences = self._split_sentences(answer)
        if not answer_sentences:
            return 0.0

        combined_context = " ".join(contexts).lower()
        supported = 0
        for sent in answer_sentences:
            tokens = set(self._tokenize(sent))
            # Check if key tokens appear in context
            if tokens and len(tokens & set(self._tokenize(combined_context))) / len(tokens) > 0.5:
                supported += 1

        return round(supported / len(answer_sentences), 4)

    def _answer_relevancy(self, question: str, answer: str) -> float:
        """
        Measures how relevant the answer is to the question.
        Heuristic: token overlap between question and answer.
        """
        if not question or not answer:
            return 0.0

        q_tokens = set(self._tokenize(question))
        a_tokens = set(self._tokenize(answer))

        # Stop words that don't count
        stopwords = {"what", "is", "the", "a", "an", "of", "in", "to", "how",
                     "does", "do", "can", "who", "when", "where", "why", "which"}
        q_tokens -= stopwords

        if not q_tokens:
            return 0.5

        overlap = len(q_tokens & a_tokens)
        score   = min(overlap / len(q_tokens), 1.0)

        # Penalty for very short answers (< 20 chars)
        if len(answer) < 20:
            score *= 0.5

        return round(score, 4)

    def _context_precision(self, question: str, contexts: List[str]) -> float:
        """
        Measures what fraction of retrieved chunks are relevant to the question.
        """
        if not contexts:
            return 0.0

        q_tokens = set(self._tokenize(question))
        if not q_tokens:
            return 0.5

        relevant = 0
        for ctx in contexts:
            ctx_tokens = set(self._tokenize(ctx))
            overlap    = len(q_tokens & ctx_tokens) / len(q_tokens)
            if overlap > 0.3:
                relevant += 1

        return round(relevant / len(contexts), 4)

    def _context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """
        Measures whether the context contains enough info to answer the ground truth.
        """
        if not ground_truth or not contexts:
            return 0.0

        gt_tokens      = set(self._tokenize(ground_truth))
        combined_ctx   = " ".join(contexts)
        ctx_tokens     = set(self._tokenize(combined_ctx))

        if not gt_tokens:
            return 0.5

        recall = len(gt_tokens & ctx_tokens) / len(gt_tokens)
        return round(min(recall, 1.0), 4)

    def _answer_correctness(self, answer: str, ground_truth: str) -> float:
        """
        F1-based token overlap between answer and ground truth.
        """
        if not ground_truth or not answer:
            return 0.0

        a_tokens  = Counter(self._tokenize(answer))
        gt_tokens = Counter(self._tokenize(ground_truth))

        common = sum((a_tokens & gt_tokens).values())
        if common == 0:
            return 0.0

        precision = common / sum(a_tokens.values())
        recall    = common / sum(gt_tokens.values())
        f1        = 2 * precision * recall / (precision + recall)
        return round(f1, 4)

    # ── Utility ────────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
