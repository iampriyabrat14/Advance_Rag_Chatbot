# Advanced RAG Chatbot 📚

> **Production-ready RAG system** with two-stage retrieval (bi-encoder + cross-encoder re-ranking), ChromaDB vector store, RAGAS-inspired evaluation, and persistent conversation memory.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## What It Does

Upload PDF or TXT documents. Ask questions. Get accurate, grounded answers — with conversation memory across turns and evaluation metrics to prove it's working.

---

## RAG Pipeline Architecture

```
Document Upload (PDF / TXT)
        │
        ▼
  Document Processor
  (chunking + metadata)
        │
        ▼
  Bi-Encoder Embeddings
  (sentence-transformers)
        │
        ▼
   ChromaDB Vector Store
   (persistent storage)
        │
   User Query
        │
        ▼
  Stage 1: Bi-Encoder Retrieval
  (top-K candidate chunks)
        │
        ▼
  Stage 2: Cross-Encoder Re-Ranking
  (precision re-scoring)
        │
        ▼
  Top-N Relevant Chunks
        +
  Conversation Memory
        │
        ▼
  LLM (OpenAI GPT / Ollama)
        │
        ▼
  Grounded Answer + Evaluation Metrics
```

---

## Features

- **Two-Stage Retrieval** — Bi-encoder for recall, cross-encoder re-ranker for precision
- **ChromaDB** — Persistent vector storage across sessions
- **Conversation Memory** — Full multi-turn dialogue with context retention
- **RAGAS-Inspired Eval** — Faithfulness, relevancy, precision, recall scored per query
- **Local LLM Support** — Works with Ollama (offline) or OpenAI GPT
- **Dockerized** — One-command deployment
- **Dark UI** — Tech-themed drag-and-drop chat interface
- **PDF + TXT Ingestion** — Supports mixed document uploads

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (bi-encoder) |
| Re-ranking | cross-encoder (sentence-transformers) |
| LLM | OpenAI GPT-4o or Ollama (local) |
| Backend | Flask, Python |
| Evaluation | Custom RAGAS-inspired metrics |
| Deployment | Docker |

---

## Quick Start

```bash
git clone https://github.com/iampriyabrat14/Advance_Rag_Chatbot.git
cd Advance_Rag_Chatbot
cp .env.example .env   # Add OPENAI_API_KEY (or set Ollama endpoint)

# With Docker
docker build -t rag-chatbot .
docker run -p 5000:5000 rag-chatbot

# Without Docker
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`, upload a document, and start chatting.

---

## Evaluation Metrics

Every response is evaluated automatically:

| Metric | What It Measures |
|--------|-----------------|
| Faithfulness | Is the answer grounded in retrieved context? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Are retrieved chunks precise? |
| Context Recall | Were all relevant chunks retrieved? |

---

## Project Structure

```
Advance_Rag_Chatbot/
├── app.py                    # Flask entrypoint
├── utils/
│   ├── document_processor.py # PDF/TXT parsing + chunking
│   ├── vector_store.py       # ChromaDB operations
│   ├── reranker.py           # Cross-encoder re-ranking
│   ├── memory.py             # Conversation history
│   ├── rag_orchestrator.py   # Pipeline coordination
│   └── evaluation.py        # RAGAS-inspired metrics
├── templates/index.html      # Single-page chat UI
├── static/                   # CSS + JS
├── Dockerfile
└── requirements.txt
```

---

## License

MIT © 2026 Priyabrat Dalbehera
