# 🧠 RAG ChatBot — Advanced Conversational AI Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)](https://flask.palletsprojects.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

A **production-ready, full-stack RAG (Retrieval-Augmented Generation) chatbot** built from scratch — designed to showcase advanced AI engineering skills for interviews and portfolio demonstrations.

---

## ✨ Key Features

| Feature | Implementation |
|---|---|
| 📄 **Document Ingestion** | PDF & TXT files with sentence-aware chunking |
| 🗄️ **Vector Database** | ChromaDB with persistent storage + cosine similarity |
| 🏆 **Re-Ranking** | Cross-Encoder (ms-marco-MiniLM) for precision boost |
| 🧠 **Conversation Memory** | Sliding-window session memory with LLM context injection |
| 📈 **RAG Evaluation** | RAGAS-inspired metrics: Faithfulness, Relevancy, Precision, Recall |
| 🌐 **Backend** | Flask REST API with session management |
| 💻 **Frontend** | Pure HTML/CSS/JS — no framework dependencies |
| 🤖 **LLM Support** | OpenAI GPT or Ollama (local LLMs) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER QUERY                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Flask API  │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
   │  Bi-Encoder │  │  Conversation│  │  Evaluator │
   │  Retrieval  │  │   Memory     │  │  (RAGAS)   │
   │  (ChromaDB) │  │              │  │            │
   └──────┬──────┘  └──────┬──────┘  └────────────┘
          │                │
   ┌──────▼──────┐         │
   │ Cross-Encoder│        │
   │  Re-Ranker  │         │
   └──────┬──────┘         │
          │                │
   ┌──────▼────────────────▼──────┐
   │          LLM (GPT / Ollama)  │
   │     Context-Aware Generation │
   └──────────────┬───────────────┘
                  │
           ┌──────▼──────┐
           │   ANSWER    │
           │ + Sources   │
           └─────────────┘
```

### RAG Pipeline Explained

```
1. RETRIEVE  → Query ChromaDB with bi-encoder embeddings (top-K candidates)
2. RE-RANK   → Cross-encoder scores each (query, chunk) pair → re-sort by relevance
3. GENERATE  → LLM synthesises answer from re-ranked context + conversation history
4. MEMORY    → Store turn in sliding-window memory for multi-turn coherence
```

---

## 📁 Project Structure

```
rag-chatbot/
│
├── app.py                      # Flask application & API routes
│
├── utils/
│   ├── document_processor.py   # PDF/TXT reader + sentence-aware chunker
│   ├── vector_store.py         # ChromaDB wrapper with CRUD operations
│   ├── reranker.py             # Cross-Encoder re-ranking (ms-marco)
│   ├── memory.py               # Per-session conversation memory
│   ├── rag_chain.py            # End-to-end RAG pipeline orchestrator
│   └── evaluator.py            # RAGAS-style evaluation metrics
│
├── templates/
│   └── index.html              # Single-page UI
│
├── static/
│   ├── css/style.css           # Dark tech theme stylesheet
│   └── js/app.js               # Frontend logic (vanilla JS)
│
├── uploads/                    # Temporary uploaded files
├── vector_db/                  # ChromaDB persistent storage
├── flask_sessions/             # Server-side session files
├── evaluations/                # Saved evaluation logs (JSON)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

```env
# Minimum required:
FLASK_SECRET_KEY=your-secret-key
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
python app.py
```

Open **http://localhost:5000** 🎉

---

## 🔧 Configuration

### Using OpenAI (Default)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
LLM_MODEL=gpt-3.5-turbo      # or gpt-4, gpt-4o
```

### Using Ollama (Free, Local)

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3
```

```env
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3
```

### Demo Mode (No LLM required)

Leave `OPENAI_API_KEY` empty. The chatbot will return retrieved context directly — great for testing the retrieval pipeline.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the chat UI |
| `POST` | `/api/upload` | Upload PDF/TXT files |
| `POST` | `/api/chat` | Send a chat message |
| `GET` | `/api/history` | Get conversation history |
| `POST` | `/api/clear` | Clear session memory |
| `GET` | `/api/stats` | Vector store statistics |
| `GET` | `/api/documents` | List ingested documents |
| `DELETE` | `/api/documents/<name>` | Delete a document |
| `POST` | `/api/evaluate` | Run RAGAS evaluation |

### Chat Request / Response

```json
// POST /api/chat
{
  "message": "What is retrieval-augmented generation?",
  "top_k": 5,
  "rerank_top_k": 3
}

// Response
{
  "answer": "RAG is a technique that combines...",
  "sources": ["rag_paper.pdf"],
  "context_used": ["RAG combines dense retrieval..."],
  "reranked": [
    {
      "text": "RAG combines...",
      "source": "rag_paper.pdf",
      "score": 0.87,
      "rerank_score": 2.43
    }
  ],
  "elapsed_sec": 1.24
}
```

### Evaluation Request

```json
// POST /api/evaluate
{
  "question": "What is RAG?",
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "contexts": ["RAG is a method that...", "The technique combines..."],
  "ground_truth": "RAG is a technique combining retrieval and generation."
}

// Response
{
  "scores": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.92,
    "context_precision": 0.75,
    "context_recall": 0.80,
    "answer_correctness": 0.78,
    "aggregate": 0.82,
    "quality_label": "Excellent ✅"
  }
}
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|---|---|
| **Faithfulness** | Does the answer stay grounded in retrieved context? |
| **Answer Relevancy** | Is the answer on-topic with the question? |
| **Context Precision** | What fraction of retrieved chunks are useful? |
| **Context Recall** | Does the context cover the ground truth? |
| **Answer Correctness** | F1 token overlap with ground truth (if provided) |
| **Aggregate** | Mean of all available metrics |

These are computed using lightweight lexical heuristics (no external API needed).  
For production use, integrate the full [RAGAS library](https://github.com/explodinggradients/ragas).

---

## 🧩 Technical Deep Dive

### Chunking Strategy

```
Input document → Sentence splitting → Sliding window (500 chars, 100 overlap)
                                    → Preserves sentence boundaries
                                    → Metadata: source, chunk_idx, char_count
```

### Embedding Model

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Distance**: Cosine similarity
- **Storage**: ChromaDB HNSW index (approximate nearest neighbour)

### Re-Ranking

```
Bi-encoder retrieval (fast)     →   top-K candidates (K=5 default)
Cross-encoder scoring (precise) →   top-N re-ranked  (N=3 default)

Cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
Input: (query, document_chunk) pair
Output: scalar relevance score → sort descending
```

### Memory Architecture

```python
session_id → deque(maxlen=max_turns * 2)
           → {"role": "user"|"assistant", "content": "...", "ts": "..."}

# Injected into LLM prompt as:
"Conversation so far:
Human: <question>
Assistant: <answer>
..."
```

---

## 🖥️ UI Features

- **Dark tech theme** with gradient accents
- **Pipeline visualiser**: Retrieve → Re-rank → Generate → Memory (animated)
- **Drag & drop** file upload with progress bar
- **Source inspector modal**: view re-ranked chunks with scores
- **Evaluation panel**: in-UI RAGAS evaluation with score bars
- **Document manager**: list and delete ingested documents
- **Retrieval sliders**: adjust top-K and re-rank-K live
- **Auto-resizing** textarea, keyboard shortcuts (Enter to send)
- **Responsive**: works on mobile and desktop

---

## 🔬 Interview Talking Points

### "Why RAG over fine-tuning?"
RAG keeps the knowledge base updatable without retraining. You can add new documents at runtime, while fine-tuning bakes knowledge into weights permanently and requires expensive retraining cycles.

### "Why re-ranking?"
Bi-encoders (used for retrieval) encode query and document *independently* — fast but approximate. Cross-encoders see the *joint* (query, document) pair, capturing interaction signals missed by bi-encoders. The two-stage pipeline balances speed and precision.

### "How does memory work?"
Each session maintains a sliding window of N recent turns. The formatted history is injected into the LLM system prompt, giving the model conversation context without exceeding token limits via truncation from the oldest end.

### "How would you scale this?"
- Replace ChromaDB with Pinecone/Weaviate/Qdrant for billion-scale search
- Async ingestion pipeline with Celery + Redis
- Cache frequent queries with Redis
- Deploy Flask behind Gunicorn + Nginx
- Containerise with Docker, orchestrate with Kubernetes

---

## 🛣️ Roadmap

- [ ] Streaming responses (SSE)
- [ ] HyDE (Hypothetical Document Embeddings) query expansion
- [ ] Parent-child chunk retrieval
- [ ] Conversation summarisation for long sessions
- [ ] LangChain / LlamaIndex integration example
- [ ] Docker Compose deployment
- [ ] Full RAGAS integration with LLM-based metrics

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web framework & REST API |
| `flask-session` | Server-side session storage |
| `chromadb` | Vector database |
| `sentence-transformers` | Bi-encoder embeddings + cross-encoder |
| `pypdf` | PDF text extraction |
| `openai` | GPT model client |

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Built with ❤️ to demonstrate production-grade RAG engineering.

> **Tip for interviewers**: Every component in this system is modular and swappable — embedding model, vector store, LLM, and evaluator are all plug-in replaceable through the environment configuration.
