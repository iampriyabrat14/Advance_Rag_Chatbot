"""
RAG Chain
=========
Orchestrates the full Retrieve → Re-rank → Generate pipeline.

Supports:
- OpenAI GPT (default)
- Ollama (local models)
- HuggingFace Inference API
"""

import os
import logging
from typing import Dict, Any, List

from utils.vector_store import VectorStore
from utils.reranker import Reranker
from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an intelligent assistant with access to a knowledge base.
Answer questions using ONLY the provided context. If the context doesn't contain enough 
information to answer, say so clearly rather than making things up.

Guidelines:
- Be concise and accurate
- Cite which document you used when relevant
- If asked a follow-up, use the conversation history for context
- Format code blocks properly if needed
"""


class RAGChain:
    """End-to-end RAG pipeline: Retrieve → Re-rank → Generate."""

    def __init__(
        self,
        vector_store: VectorStore,
        reranker: Reranker,
        memory: ConversationMemory,
        llm_provider: str = None
    ):
        self.vector_store = vector_store
        self.reranker     = reranker
        self.memory       = memory
        self.llm_provider = llm_provider or os.environ.get("LLM_PROVIDER", "openai")
        self._llm_client  = None
        self._init_llm()

    # ── LLM Initialisation ─────────────────────────────────────────────────────

    def _init_llm(self):
        if self.llm_provider == "openai":
            try:
                import openai
                self._llm_client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY", "")
                )
                logger.info("LLM: OpenAI client initialised")
            except ImportError:
                logger.warning("openai package not installed, LLM calls will use mock")
        elif self.llm_provider == "ollama":
            logger.info("LLM: Using Ollama (local)")
        else:
            logger.info(f"LLM: Unknown provider '{self.llm_provider}', using mock")

    # ── Main Pipeline ──────────────────────────────────────────────────────────

    def chat(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        rerank_top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline:
        1. Retrieve top-k candidates from vector store
        2. Re-rank with cross-encoder
        3. Build prompt with memory + context
        4. Generate answer with LLM
        5. Update memory
        """

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        candidates = self.vector_store.similarity_search(query=query, top_k=top_k)
        logger.info(f"Retrieved {len(candidates)} candidates for query: '{query[:60]}...'")

        # ── Step 2: Re-rank ───────────────────────────────────────────────────
        if candidates:
            reranked = self.reranker.rerank(query, candidates, top_k=rerank_top_k)
        else:
            reranked = []

        # ── Step 3: Build context ─────────────────────────────────────────────
        context_text   = self._build_context(reranked)
        history_text   = self.memory.get_formatted_context(session_id)
        recent_messages = self.memory.get_recent_messages(session_id, n=6)

        # ── Step 4: Generate ──────────────────────────────────────────────────
        answer = self._generate(
            query=query,
            context=context_text,
            history=history_text,
            recent_messages=recent_messages
        )

        # ── Step 5: Update memory ─────────────────────────────────────────────
        self.memory.add_turn(session_id, "user", query)
        self.memory.add_turn(session_id, "assistant", answer)

        # ── Build response ────────────────────────────────────────────────────
        sources = list({
            c["metadata"].get("source", "unknown")
            for c in reranked
        })

        return {
            "answer":       answer,
            "sources":      sources,
            "context_used": [c["text"][:200] + "…" for c in reranked],
            "reranked":     [
                {
                    "text":          c["text"][:150] + "…",
                    "source":        c["metadata"].get("source"),
                    "score":         c.get("score"),
                    "rerank_score":  c.get("rerank_score")
                }
                for c in reranked
            ]
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_context(self, documents: List[Dict]) -> str:
        if not documents:
            return "No relevant documents found in the knowledge base."
        parts = []
        for i, doc in enumerate(documents, 1):
            src = doc["metadata"].get("source", "Unknown")
            parts.append(f"[{i}] Source: {src}\n{doc['text']}")
        return "\n\n---\n\n".join(parts)

    def _generate(
        self,
        query: str,
        context: str,
        history: str,
        recent_messages: List[Dict]
    ) -> str:
        """Generate an answer using the configured LLM provider."""

        user_prompt = f"""Context from knowledge base:
{context}

{"Conversation so far:" + chr(10) + history if history else ""}

Question: {query}

Answer based strictly on the context above:"""

        if self.llm_provider == "openai" and self._llm_client:
            return self._call_openai(user_prompt, recent_messages)
        elif self.llm_provider == "ollama":
            return self._call_ollama(user_prompt)
        else:
            return self._mock_response(query, context)

    def _call_openai(self, user_prompt: str, recent_messages: List[Dict]) -> str:
        model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(recent_messages)
        messages.append({"role": "user", "content": user_prompt})
        try:
            response = self._llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return f"LLM error: {e}"

    def _call_ollama(self, user_prompt: str) -> str:
        import requests
        model = os.environ.get("OLLAMA_MODEL", "llama3")
        url   = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        try:
            resp = requests.post(url, json={
                "model":  model,
                "prompt": SYSTEM_PROMPT + "\n\n" + user_prompt,
                "stream": False
            }, timeout=60)
            return resp.json().get("response", "No response from Ollama").strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return f"Ollama error: {e}"

    def _mock_response(self, query: str, context: str) -> str:
        """Demo response when no LLM is configured."""
        if "no relevant" in context.lower():
            return (
                "I couldn't find relevant information in the knowledge base. "
                "Please upload documents first and then ask your question."
            )
        preview = context[:400].replace("\n", " ")
        return (
            f"[DEMO MODE – configure OPENAI_API_KEY or use Ollama]\n\n"
            f"Based on the retrieved context, here is a summary:\n\n{preview}…\n\n"
            f"To get real AI-generated answers, set your LLM provider in .env."
        )
