"""
Vector Store
============
ChromaDB-backed vector store with SentenceTransformer embeddings.
Supports multi-collection management, CRUD, and similarity search.
"""

import logging
from typing import List, Dict, Any, Optional
from utils.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wraps ChromaDB with a SentenceTransformer embedding function.
    Uses a single 'rag_documents' collection with source-level metadata.
    """

    COLLECTION_NAME = "rag_documents"

    def __init__(
        self,
        persist_directory: str = "./vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory
        self.embedding_model   = embedding_model
        self._client     = None
        self._collection = None
        self._ef         = None
        self._init()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init(self):
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(
                f"VectorStore ready: {self._collection.count()} docs | "
                f"model={self.embedding_model}"
            )
        except ImportError:
            raise ImportError("Install chromadb & sentence-transformers: "
                              "pip install chromadb sentence-transformers")

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_documents(self, chunks: List[DocumentChunk], source: str):
        """Embed and store document chunks."""
        if not chunks:
            return

        # Build unique IDs (avoid collisions on re-ingestion)
        ids       = [f"{source}_chunk_{c.metadata['chunk_idx']}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Upsert to handle re-ingestion gracefully
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Upserted {len(chunks)} chunks from '{source}'")

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return top-k similar chunks with distance scores."""
        where = {"source": source_filter} if source_filter else None

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            hits.append({
                "text":     doc,
                "metadata": meta,
                "score":    round(1 - dist, 4)   # cosine similarity
            })
        return hits

    def get_doc_count(self) -> int:
        return self._collection.count()

    def get_collections(self) -> List[str]:
        return [c.name for c in self._client.list_collections()]

    def list_sources(self) -> List[str]:
        """Return unique source filenames stored in the DB."""
        if self._collection.count() == 0:
            return []
        results = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in results["metadatas"]}
        return sorted(sources)

    def delete_source(self, source: str):
        """Delete all chunks belonging to a source document."""
        results = self._collection.get(
            where={"source": source},
            include=["metadatas"]
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for '{source}'")
        else:
            raise ValueError(f"Source '{source}' not found in vector store")
