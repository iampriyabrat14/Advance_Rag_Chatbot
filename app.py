"""
RAG-Based Advanced Conversational Chatbot
=========================================
Author: Your Name
Description: A production-ready RAG chatbot with PDF/Text ingestion,
             ChromaDB vector store, cross-encoder re-ranking, 
             conversation memory, and RAGAS evaluation.
"""

import os
import json
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from werkzeug.utils import secure_filename

from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.reranker import Reranker
from utils.rag_chain import RAGChain
from utils.memory import ConversationMemory
from utils.evaluator import RAGEvaluator

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── App Configuration ────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "rag-chatbot-secret-2024")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_sessions"
app.config["UPLOAD_FOLDER"] = "./uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
app.config["ALLOWED_EXTENSIONS"] = {"pdf", "txt"}
Session(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("./flask_sessions", exist_ok=True)
os.makedirs("./evaluations", exist_ok=True)

# ─── Global Components ────────────────────────────────────────────────────────
doc_processor = DocumentProcessor()
vector_store  = VectorStore(persist_directory="./vector_db")
reranker      = Reranker()
memory        = ConversationMemory(max_turns=10)
rag_chain     = RAGChain(vector_store=vector_store, reranker=reranker, memory=memory)
evaluator     = RAGEvaluator()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = f"sess_{int(time.time())}"
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_documents():
    """Upload and ingest PDF / TXT files into the vector store."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    results = []

    for file in files:
        if not file or not allowed_file(file.filename):
            results.append({"filename": file.filename, "status": "skipped – unsupported type"})
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            chunks = doc_processor.process_file(filepath)
            vector_store.add_documents(chunks, source=filename)
            results.append({
                "filename": filename,
                "status": "success",
                "chunks": len(chunks)
            })
            logger.info(f"Ingested {filename}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            results.append({"filename": filename, "status": f"error: {str(e)}"})

    return jsonify({"results": results, "total_docs": vector_store.get_doc_count()})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint with RAG pipeline."""
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"].strip()
    session_id   = session.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        start_time = time.time()
        result = rag_chain.chat(
            query=user_message,
            session_id=session_id,
            top_k=int(data.get("top_k", 5)),
            rerank_top_k=int(data.get("rerank_top_k", 3))
        )
        elapsed = round(time.time() - start_time, 2)

        response = {
            "answer":       result["answer"],
            "sources":      result["sources"],
            "context_used": result["context_used"],
            "reranked":     result["reranked"],
            "elapsed_sec":  elapsed,
            "session_id":   session_id,
            "timestamp":    datetime.now().isoformat()
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """Run RAGAS-style evaluation on a Q&A pair."""
    data = request.get_json()
    required = ["question", "answer", "contexts"]
    if not all(k in data for k in required):
        return jsonify({"error": f"Required fields: {required}"}), 400

    try:
        scores = evaluator.evaluate(
            question=data["question"],
            answer=data["answer"],
            contexts=data["contexts"],
            ground_truth=data.get("ground_truth", "")
        )
        # Persist evaluation log
        log_path = f"./evaluations/eval_{int(time.time())}.json"
        with open(log_path, "w") as f:
            json.dump({"input": data, "scores": scores, "ts": datetime.now().isoformat()}, f, indent=2)

        return jsonify({"scores": scores, "log": log_path})
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return conversation history for the current session."""
    session_id = session.get("session_id", "default")
    history = memory.get_history(session_id)
    return jsonify({"history": history, "session_id": session_id})


@app.route("/api/clear", methods=["POST"])
def clear_session():
    """Clear conversation memory for current session."""
    session_id = session.get("session_id", "default")
    memory.clear(session_id)
    return jsonify({"message": "Conversation cleared", "session_id": session_id})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return vector store statistics."""
    return jsonify({
        "total_documents": vector_store.get_doc_count(),
        "collections":     vector_store.get_collections(),
        "model_info": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker_model":  "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "llm":             os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        }
    })


@app.route("/api/documents", methods=["GET"])
def list_documents():
    """List all ingested documents."""
    docs = vector_store.list_sources()
    return jsonify({"documents": docs})


@app.route("/api/documents/<source>", methods=["DELETE"])
def delete_document(source):
    """Delete a document from the vector store."""
    try:
        vector_store.delete_source(source)
        return jsonify({"message": f"Deleted: {source}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
