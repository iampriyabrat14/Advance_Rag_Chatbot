/**
 * RAG ChatBot – app.js
 * Handles: chat, file upload, evaluation, UI state
 */

"use strict";

// ─── State ──────────────────────────────────────────────────────────────────
const state = {
  loading:  false,
  lastSources: []
};

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const els = {
  messages:     $("messages"),
  userInput:    $("userInput"),
  sendBtn:      $("sendBtn"),
  clearBtn:     $("clearBtn"),
  fileInput:    $("fileInput"),
  uploadZone:   $("uploadZone"),
  uploadProg:   $("uploadProgress"),
  progressFill: $("progressFill"),
  uploadStatus: $("uploadStatus"),
  docList:      $("docList"),
  docCount:     $("docCount"),
  chunkCount:   $("chunkCount"),
  statusDot:    $("statusDot"),
  topK:         $("topK"),
  topKVal:      $("topKVal"),
  reRankK:      $("reRankK"),
  reRankVal:    $("reRankVal"),
  charCount:    $("charCount"),
  sessionInfo:  $("sessionInfo"),
  sidebar:      $("sidebar"),
  sidebarToggle:$("sidebarToggle"),
  mobileMenu:   $("mobileMenu"),
  evalPanel:    $("evalPanel"),
  evalPanelBtn: $("evalPanelBtn"),
  closeEvalPanel:$("closeEvalPanel"),
  runEval:      $("runEval"),
  evalQ:        $("evalQ"),
  evalA:        $("evalA"),
  evalCtx:      $("evalCtx"),
  evalGT:       $("evalGT"),
  evalResults:  $("evalResults"),
  overlay:      $("overlay"),
  sourcesModal: $("sourcesModal"),
  closeModal:   $("closeModal"),
  modalBody:    $("modalBody"),
  refreshDocs:  $("refreshDocs"),
  // pipeline steps
  psRetrieve:   $("ps-retrieve"),
  psRerank:     $("ps-rerank"),
  psGenerate:   $("ps-generate"),
  psMemory:     $("ps-memory"),
};

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadStats();
  loadDocuments();
  bindEvents();
  autoResizeTextarea();
});

// ─── Event Bindings ───────────────────────────────────────────────────────────
function bindEvents() {
  // Send message
  els.sendBtn.addEventListener("click", sendMessage);
  els.userInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  els.userInput.addEventListener("input", () => {
    const len = els.userInput.value.length;
    els.charCount.textContent = `${len} / 2000`;
    autoResizeTextarea();
  });

  // Clear
  els.clearBtn.addEventListener("click", clearConversation);

  // File upload
  els.uploadZone.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", e => uploadFiles(e.target.files));

  // Drag & drop
  els.uploadZone.addEventListener("dragover", e => { e.preventDefault(); els.uploadZone.classList.add("drag-over"); });
  els.uploadZone.addEventListener("dragleave", () => els.uploadZone.classList.remove("drag-over"));
  els.uploadZone.addEventListener("drop", e => {
    e.preventDefault();
    els.uploadZone.classList.remove("drag-over");
    uploadFiles(e.dataTransfer.files);
  });

  // Sliders
  els.topK.addEventListener("input", () => els.topKVal.textContent = els.topK.value);
  els.reRankK.addEventListener("input", () => els.reRankVal.textContent = els.reRankK.value);

  // Sidebar
  els.sidebarToggle.addEventListener("click", () => {
    els.sidebar.classList.toggle("collapsed");
    els.sidebarToggle.textContent = els.sidebar.classList.contains("collapsed") ? "▶" : "◀";
  });

  els.mobileMenu.addEventListener("click", () => {
    els.sidebar.classList.toggle("mobile-open");
    els.overlay.classList.toggle("show");
  });

  // Evaluation panel
  els.evalPanelBtn.addEventListener("click", () => els.evalPanel.classList.toggle("open"));
  els.closeEvalPanel.addEventListener("click", () => els.evalPanel.classList.remove("open"));
  els.runEval.addEventListener("click", runEvaluation);

  // Modal
  els.closeModal.addEventListener("click", closeModal);
  els.overlay.addEventListener("click", () => {
    closeModal();
    els.sidebar.classList.remove("mobile-open");
    els.overlay.classList.remove("show");
  });

  // Refresh docs
  els.refreshDocs.addEventListener("click", loadDocuments);
}

// ─── Chat ─────────────────────────────────────────────────────────────────────
async function sendMessage() {
  const text = els.userInput.value.trim();
  if (!text || state.loading) return;

  appendMessage("user", text);
  els.userInput.value = "";
  els.charCount.textContent = "0 / 2000";
  autoResizeTextarea();

  state.loading = true;
  els.sendBtn.disabled = true;

  const typingId = appendTyping();
  animatePipeline();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message:      text,
        top_k:        parseInt(els.topK.value),
        rerank_top_k: parseInt(els.reRankK.value)
      })
    });

    const data = await res.json();
    removeTyping(typingId);
    resetPipeline();

    if (data.error) {
      appendMessage("ai", `⚠️ Error: ${data.error}`);
    } else {
      state.lastSources = data.reranked || [];
      appendAIMessage(data);
      // Auto-fill eval panel
      els.evalQ.value   = text;
      els.evalA.value   = data.answer;
      els.evalCtx.value = (data.context_used || []).join("\n---\n");
    }
  } catch (err) {
    removeTyping(typingId);
    resetPipeline();
    appendMessage("ai", `⚠️ Network error: ${err.message}`);
  } finally {
    state.loading = false;
    els.sendBtn.disabled = false;
    els.userInput.focus();
  }
}

function appendMessage(role, text) {
  const el = createBubble(role, text);
  els.messages.appendChild(el);
  scrollToBottom();
  return el;
}

function appendAIMessage(data) {
  const div = document.createElement("div");
  div.className = "message ai";
  div.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="bubble">
      ${formatText(data.answer)}
      ${data.sources?.length ? `
        <div class="source-chips">
          ${data.sources.map(s => `<span class="chip">📄 ${s}</span>`).join("")}
        </div>` : ""}
      <div class="bubble-meta">
        <button class="sources-btn" onclick="openSourcesModal()">
          🔍 View Sources (${data.reranked?.length || 0})
        </button>
        <span class="elapsed-badge">⏱ ${data.elapsed_sec}s</span>
      </div>
    </div>`;
  els.messages.appendChild(div);
  scrollToBottom();
}

function createBubble(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const avatar = role === "user" ? "👤" : "🤖";
  div.innerHTML = `
    <div class="avatar">${avatar}</div>
    <div class="bubble">${formatText(text)}</div>`;
  return div;
}

function appendTyping() {
  const id  = "typing_" + Date.now();
  const div = document.createElement("div");
  div.className = "message ai typing-indicator";
  div.id = id;
  div.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>`;
  els.messages.appendChild(div);
  scrollToBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function formatText(text) {
  if (!text) return "";
  // Bold
  text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  // Inline code
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  // Code blocks
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code>${escHtml(code.trim())}</code></pre>`);
  // Line breaks
  text = text.replace(/\n/g, "<br>");
  return text;
}

function escHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ─── Pipeline Animation ───────────────────────────────────────────────────────
let pipelineTimer = null;

function animatePipeline() {
  const steps = [els.psRetrieve, els.psRerank, els.psGenerate, els.psMemory];
  let i = 0;
  steps.forEach(s => s.className = "pipeline-step");

  pipelineTimer = setInterval(() => {
    if (i > 0) steps[i-1].className = "pipeline-step done";
    if (i < steps.length) { steps[i].className = "pipeline-step active"; i++; }
    else { clearInterval(pipelineTimer); }
  }, 700);
}

function resetPipeline() {
  clearInterval(pipelineTimer);
  const steps = [els.psRetrieve, els.psRerank, els.psGenerate, els.psMemory];
  steps.forEach(s => { s.className = "pipeline-step done"; });
  setTimeout(() => steps.forEach(s => s.className = "pipeline-step"), 1500);
}

// ─── Sources Modal ────────────────────────────────────────────────────────────
function openSourcesModal() {
  if (!state.lastSources.length) return;

  els.modalBody.innerHTML = state.lastSources.map((src, i) => `
    <div class="source-card">
      <div class="source-card-header">
        <span class="source-rank">#${i+1} — Re-ranked</span>
        <div class="source-scores">
          ${src.score != null ? `<span class="score-badge embed">embed: ${src.score}</span>` : ""}
          ${src.rerank_score != null ? `<span class="score-badge rerank">rerank: ${src.rerank_score}</span>` : ""}
        </div>
      </div>
      <p class="source-text">${escHtml(src.text || "")}</p>
      <p class="source-name">📄 ${src.source || "Unknown"}</p>
    </div>
  `).join("");

  els.sourcesModal.classList.add("show");
  els.overlay.classList.add("show");
}

function closeModal() {
  els.sourcesModal.classList.remove("show");
  els.overlay.classList.remove("show");
}

// ─── Upload ───────────────────────────────────────────────────────────────────
async function uploadFiles(fileList) {
  if (!fileList || fileList.length === 0) return;

  const formData = new FormData();
  [...fileList].forEach(f => formData.append("files", f));

  els.uploadProg.style.display = "block";
  els.progressFill.style.width = "20%";
  els.uploadStatus.textContent = "Uploading & processing…";

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();

    els.progressFill.style.width = "100%";
    els.uploadStatus.textContent = `✅ ${data.results?.length} file(s) processed`;

    setTimeout(() => { els.uploadProg.style.display = "none"; els.progressFill.style.width = "0%"; }, 2500);

    loadDocuments();
    loadStats();

    const summary = (data.results || []).map(r => `${r.filename}: ${r.status}`).join("\n");
    appendMessage("ai", `📁 **Documents ingested:**\n${summary}`);
  } catch (err) {
    els.uploadStatus.textContent = `❌ Error: ${err.message}`;
  }

  els.fileInput.value = "";
}

// ─── Documents ────────────────────────────────────────────────────────────────
async function loadDocuments() {
  try {
    const res  = await fetch("/api/documents");
    const data = await res.json();
    const docs = data.documents || [];

    els.docCount.textContent = docs.length;

    if (docs.length === 0) {
      els.docList.innerHTML = '<li class="empty-hint">No documents yet</li>';
      return;
    }

    els.docList.innerHTML = docs.map(d => `
      <li class="doc-item">
        <span>📄</span>
        <span class="doc-name" title="${d}">${d}</span>
        <button class="del-btn" onclick="deleteDoc('${d}')" title="Delete">✕</button>
      </li>`).join("");
  } catch (e) { console.error("loadDocuments error:", e); }
}

async function deleteDoc(source) {
  if (!confirm(`Delete "${source}" from knowledge base?`)) return;
  try {
    await fetch(`/api/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
    loadDocuments();
    loadStats();
  } catch (e) { alert("Delete failed: " + e.message); }
}

async function loadStats() {
  try {
    const res  = await fetch("/api/stats");
    const data = await res.json();
    els.chunkCount.textContent = data.total_documents ?? "—";
    els.statusDot.className = "status-dot active";
  } catch (e) {
    els.statusDot.className = "status-dot";
  }
}

// ─── Clear conversation ───────────────────────────────────────────────────────
async function clearConversation() {
  if (!confirm("Clear conversation history?")) return;
  await fetch("/api/clear", { method: "POST" });
  els.messages.innerHTML = `
    <div class="welcome-card">
      <div class="welcome-icon">✨</div>
      <h3>Conversation cleared</h3>
      <p>Start a new conversation by asking a question.</p>
    </div>`;
}

// ─── Evaluation ───────────────────────────────────────────────────────────────
async function runEvaluation() {
  const q   = els.evalQ.value.trim();
  const a   = els.evalA.value.trim();
  const ctx = els.evalCtx.value.trim();

  if (!q || !a || !ctx) { alert("Fill in Question, Answer and Context."); return; }

  els.runEval.textContent = "Running…";
  els.runEval.disabled    = true;

  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question:     q,
        answer:       a,
        contexts:     ctx.split("---").map(s => s.trim()).filter(Boolean),
        ground_truth: els.evalGT.value.trim()
      })
    });

    const data = await res.json();
    renderEvalResults(data.scores || {});
  } catch (e) {
    alert("Evaluation failed: " + e.message);
  } finally {
    els.runEval.textContent = "▶ Run Evaluation";
    els.runEval.disabled    = false;
  }
}

function renderEvalResults(scores) {
  const metrics = [
    ["faithfulness",       "Faithfulness"],
    ["answer_relevancy",   "Answer Relevancy"],
    ["context_precision",  "Context Precision"],
    ["context_recall",     "Context Recall"],
    ["answer_correctness", "Answer Correctness"],
    ["aggregate",          "Aggregate Score"]
  ];

  const scoreColor = v => {
    if (v >= 0.8) return "#10b981";
    if (v >= 0.6) return "#f59e0b";
    if (v >= 0.4) return "#f97316";
    return "#ef4444";
  };

  els.evalResults.style.display = "block";
  els.evalResults.innerHTML = `
    ${metrics.filter(([k]) => scores[k] != null).map(([k, label]) => {
      const v   = scores[k];
      const pct = Math.round(v * 100);
      const col = scoreColor(v);
      return `
        <div class="metric-row">
          <span class="metric-name">${label}</span>
          <div>
            <span class="metric-score" style="color:${col}">${pct}%</span>
            <div class="score-bar">
              <div class="score-fill" style="width:${pct}%;background:${col}"></div>
            </div>
          </div>
        </div>`;
    }).join("")}
    <div class="quality-label">${scores.quality_label || ""}</div>`;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function autoResizeTextarea() {
  const ta = els.userInput;
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
}
