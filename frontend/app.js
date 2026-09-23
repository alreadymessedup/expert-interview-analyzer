// Expert Interview Analyzer -- vanilla JS frontend, no build step.
// Talks to the FastAPI backend in server.py via the /api/* routes below.

const state = {
  analysisId: null,
  experts: [],       // [{expert_label, display_name, results:[...]}]
  crossAnalysis: [],
  transcripts: [],    // [{expert_label, source_filename, has_timestamps, segments:[...]}]
  chatHistory: [],    // [{question, answer, citations, quotes}]
  activeTabIndex: 0,
};

const el = (id) => document.getElementById(id);

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function setStatus(text) {
  el("header-status").textContent = text || "";
}

function showError(message) {
  const box = el("error-box");
  box.textContent = message;
  box.classList.remove("hidden");
}

function clearError() {
  el("error-box").classList.add("hidden");
  el("error-box").textContent = "";
}

// --------------------------------------------------------------------------
// Boot: load default interview guide + available models
// --------------------------------------------------------------------------
async function boot() {
  try {
    const [guideRes, modelsRes] = await Promise.all([
      fetch("/api/default-guide").then((r) => r.json()),
      fetch("/api/models").then((r) => r.json()),
    ]);

    el("objective").value = guideRes.objective || "";
    el("questions").value = (guideRes.questions || []).join("\n");

    const select = el("model-select");
    select.innerHTML = "";
    modelsRes.options.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      select.appendChild(opt);
    });
    select.value = modelsRes.default;
  } catch (e) {
    setStatus("Could not reach backend");
  }

  loadPreviousAnalyses();
}

// --------------------------------------------------------------------------
// Previous analyses (persisted via ChromaDB -- see core/store.py)
// --------------------------------------------------------------------------
async function loadPreviousAnalyses() {
  const container = el("previous-analyses-list");
  try {
    const res = await fetch("/api/analyses");
    const list = await res.json();
    if (!res.ok) throw new Error("Could not load previous analyses.");

    if (!list.length) {
      container.innerHTML = `<p class="text-xs text-slate">No previous analyses saved yet -- run one below.</p>`;
      return;
    }
    container.innerHTML = list.map(renderPreviousAnalysisRow).join("");
    container.querySelectorAll("[data-resume-id]").forEach((btn) => {
      btn.addEventListener("click", () => resumeAnalysis(btn.dataset.resumeId));
    });
    container.querySelectorAll("[data-delete-id]").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        btn.textContent = "Deleting...";
        try {
          await fetch(`/api/analysis/${btn.dataset.deleteId}`, { method: "DELETE" });
        } finally {
          loadPreviousAnalyses();
        }
      });
    });
  } catch (e) {
    container.innerHTML = `<p class="text-xs text-warn">Could not load previous analyses -- is the backend running?</p>`;
  }
}

function renderPreviousAnalysisRow(a) {
  const when = new Date(a.created_at).toLocaleString();
  const objectivePreview =
    a.project_objective.length > 90 ? a.project_objective.slice(0, 90) + "..." : a.project_objective;
  return `
  <div class="flex items-center justify-between border border-line rounded px-3 py-2 bg-white">
    <div class="min-w-0">
      <p class="text-sm font-medium truncate">${escapeHtml(a.expert_names.join(", "))}</p>
      <p class="text-xs text-slate truncate">${escapeHtml(objectivePreview)} -- ${escapeHtml(when)}</p>
    </div>
    <div class="flex items-center gap-3 shrink-0 ml-3">
      <button data-resume-id="${a.analysis_id}" class="text-xs font-medium text-accent hover:underline">Resume</button>
      <button data-delete-id="${a.analysis_id}" class="text-xs text-slate hover:text-warn">Delete</button>
    </div>
  </div>`;
}

async function resumeAnalysis(analysisId) {
  clearError();
  setStatus("Loading saved analysis...");
  try {
    const res = await fetch(`/api/analysis/${analysisId}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Could not load that analysis.");

    state.analysisId = data.analysis_id;
    state.experts = data.experts;
    state.crossAnalysis = data.cross_analysis;
    state.transcripts = data.transcripts;
    state.chatHistory = data.chat_history || [];
    state.activeTabIndex = 0;

    if (data.project_objective) el("objective").value = data.project_objective;
    if (data.questions && data.questions.length) el("questions").value = data.questions.join("\n");
    if (data.model) el("model-select").value = data.model;

    el("upload-panel").classList.add("hidden");
    el("results-panel").classList.remove("hidden");
    renderTabs();
    setStatus("Resumed saved analysis");
  } catch (e) {
    showError(e.message || "Could not resume that analysis.");
    setStatus("");
  }
}

// --------------------------------------------------------------------------
// File selection UI feedback
// --------------------------------------------------------------------------
document.querySelectorAll(".expert-file").forEach((input) => {
  input.addEventListener("change", () => {
    const card = input.closest("[data-expert-card]");
    const status = card.querySelector(".expert-file-status");
    status.textContent = input.files.length ? input.files[0].name : "No file selected";
  });
});

// --------------------------------------------------------------------------
// Analyze
// --------------------------------------------------------------------------
el("analyze-btn").addEventListener("click", async () => {
  clearError();

  const cards = [...document.querySelectorAll("[data-expert-card]")];
  const files = cards.map((c) => c.querySelector(".expert-file").files[0]);
  const names = cards.map((c) => c.querySelector(".expert-name").value.trim() || "Expert");

  if (files.some((f) => !f)) {
    showError("Please choose a transcript file for all three experts.");
    return;
  }

  const formData = new FormData();
  formData.append("file1", files[0]);
  formData.append("file2", files[1]);
  formData.append("file3", files[2]);
  formData.append("name1", names[0]);
  formData.append("name2", names[1]);
  formData.append("name3", names[2]);
  formData.append("objective", el("objective").value);
  formData.append("questions", el("questions").value);
  formData.append("model", el("model-select").value);
  // No api_key field appended -- server falls back to GROQ_API_KEY from its
  // own environment/.env, which is the expected setup for this deployment.

  const btn = el("analyze-btn");
  btn.disabled = true;
  el("analyze-status").textContent = "Analyzing -- answering each expert's questions, then comparing across experts. This can take a minute...";
  setStatus("Analyzing...");

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Analysis failed.");
    }

    state.analysisId = data.analysis_id;
    state.experts = data.experts;
    state.crossAnalysis = data.cross_analysis;
    state.transcripts = data.transcripts;
    state.chatHistory = [];
    state.activeTabIndex = 0;

    el("upload-panel").classList.add("hidden");
    el("results-panel").classList.remove("hidden");
    renderTabs();
    setStatus("Analysis ready");
    loadPreviousAnalyses(); // refresh so the just-saved analysis shows up when the user goes back
  } catch (e) {
    showError(e.message || "Something went wrong contacting the backend.");
    setStatus("");
  } finally {
    btn.disabled = false;
    el("analyze-status").textContent = "";
  }
});

el("new-analysis-btn").addEventListener("click", () => {
  state.analysisId = null;
  state.experts = [];
  state.crossAnalysis = [];
  state.transcripts = [];
  state.chatHistory = [];
  el("results-panel").classList.add("hidden");
  el("upload-panel").classList.remove("hidden");
  clearError();
  setStatus("");
  loadPreviousAnalyses();
});

// --------------------------------------------------------------------------
// Tabs
// --------------------------------------------------------------------------
function tabLabels() {
  return [...state.experts.map((e) => e.display_name), "Cross-expert analysis", "Ask a question", "Raw transcripts"];
}

function renderTabs() {
  const labels = tabLabels();
  const bar = el("tab-bar");
  bar.innerHTML = labels
    .map(
      (label, i) => `
      <button class="tab-btn px-3 py-2 text-sm border-b-2 -mb-px" data-active="${i === state.activeTabIndex}" data-tab-index="${i}">
        ${escapeHtml(label)}
      </button>`
    )
    .join("");

  bar.querySelectorAll("[data-tab-index]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.activeTabIndex = parseInt(btn.dataset.tabIndex, 10);
      renderTabs();
    });
  });

  renderTabContent();
}

function renderTabContent() {
  const container = el("tab-content");
  const nExperts = state.experts.length;
  const idx = state.activeTabIndex;

  if (idx < nExperts) {
    container.innerHTML = renderExpertTab(state.experts[idx]);
  } else if (idx === nExperts) {
    container.innerHTML = renderCrossAnalysisTab();
  } else if (idx === nExperts + 1) {
    container.innerHTML = renderChatTab();
    wireChatTab();
  } else {
    container.innerHTML = renderRawTranscriptsTab();
  }
}

// --------------------------------------------------------------------------
// Per-expert tab
// --------------------------------------------------------------------------
const STATUS_LABEL = {
  answered: `<span class="inline-flex items-center gap-1.5 text-verified text-xs font-medium"><span class="status-dot" style="background:#3C6E52"></span>Answered</span>`,
  partial: `<span class="inline-flex items-center gap-1.5 text-warn text-xs font-medium"><span class="status-dot" style="background:#9C6B33"></span>Partially discussed</span>`,
  not_discussed: `<span class="inline-flex items-center gap-1.5 text-slate text-xs font-medium"><span class="status-dot" style="background:#A9B7C4"></span>Not discussed</span>`,
};

function renderExpertTab(ea) {
  return `<div class="space-y-4">${ea.results.map(renderQuestionCard).join("")}</div>`;
}

function renderQuestionCard(r) {
  const citations = (r.citations || [])
    .map((c) => escapeHtml(c.label) + (c.exists === false ? " (unverified id)" : ""))
    .join(", ");

  const quotes = (r.quotes || [])
    .map((q) =>
      q.verified
        ? `<blockquote class="quote text-sm italic text-ink/80 my-2">&ldquo;${escapeHtml(q.text)}&rdquo; <span class="not-italic text-slate">-- ${escapeHtml(q.label)}</span></blockquote>`
        : `<p class="text-xs text-warn my-2">A quote here could not be verified against the transcript and was hidden.</p>`
    )
    .join("");

  return `
  <div class="border border-line rounded-md p-4 bg-white/60">
    <div class="flex items-start justify-between gap-4 mb-1">
      <p class="font-medium text-sm">${escapeHtml(r.question)}</p>
      ${STATUS_LABEL[r.status] || ""}
    </div>
    ${r.answer ? `<p class="text-sm text-ink/90 mt-2">${escapeHtml(r.answer)}</p>` : ""}
    ${citations ? `<p class="text-xs text-slate mt-2">Source: ${citations}</p>` : ""}
    ${quotes}
  </div>`;
}

// --------------------------------------------------------------------------
// Cross-expert analysis tab
// --------------------------------------------------------------------------
function renderCrossAnalysisTab() {
  if (!state.crossAnalysis || !state.crossAnalysis.length) {
    return `<p class="text-sm text-slate">No cross-expert analysis available.</p>`;
  }
  return `<div class="space-y-4">${state.crossAnalysis.map(renderCrossItem).join("")}</div>`;
}

function renderCrossItem(item) {
  const themes = item.themes || [];
  const disagreements = item.disagreements || [];

  const themesHtml = themes.length
    ? `<div class="mt-3">
        <p class="text-xs font-semibold uppercase tracking-wide text-verified mb-1.5">Common themes</p>
        <ul class="space-y-1.5">
          ${themes
            .map((t) => {
              const experts = (t.supporting_experts || []).map((s) => `${escapeHtml(s.expert || "?")} [${escapeHtml(s.segment_id || "?")}]`).join(", ");
              return `<li class="text-sm"><span>${escapeHtml(t.summary || "")}</span><br/><span class="text-xs text-slate">${experts}</span></li>`;
            })
            .join("")}
        </ul>
      </div>`
    : "";

  const disagreementsHtml = disagreements.length
    ? `<div class="mt-3">
        <p class="text-xs font-semibold uppercase tracking-wide text-warn mb-1.5">Disagreements</p>
        <ul class="space-y-1.5">
          ${disagreements
            .map((d) => {
              const positions = (d.positions || [])
                .map((p) => `${escapeHtml(p.expert || "?")}: ${escapeHtml(p.position || "")} [${escapeHtml(p.segment_id || "?")}]`)
                .join("; ");
              return `<li class="text-sm"><span>${escapeHtml(d.summary || "")}</span><br/><span class="text-xs text-slate">${positions}</span></li>`;
            })
            .join("")}
        </ul>
      </div>`
    : "";

  const empty = !themes.length && !disagreements.length
    ? `<p class="text-xs text-slate mt-2">No cross-expert theme or disagreement identified for this question.</p>`
    : "";

  return `
  <div class="border border-line rounded-md p-4 bg-white/60">
    <p class="font-medium text-sm">${escapeHtml(item.question || "")}</p>
    ${themesHtml}
    ${disagreementsHtml}
    ${empty}
  </div>`;
}

// --------------------------------------------------------------------------
// Chat tab
// --------------------------------------------------------------------------
function renderChatTab() {
  const history = state.chatHistory
    .map(
      (turn) => `
      <div class="mb-4">
        <p class="text-sm font-medium">${escapeHtml(turn.question)}</p>
        <div class="mt-1.5 border border-line rounded-md p-3 bg-white/60">
          <p class="text-sm">${escapeHtml(turn.answer)}</p>
          ${
            turn.citations && turn.citations.length
              ? `<p class="text-xs text-slate mt-2">Source: ${turn.citations
                  .map((c) => `${escapeHtml(c.expert_label || "")} ${escapeHtml(c.label || "")}`)
                  .join(", ")}</p>`
              : ""
          }
          ${
            (turn.quotes || [])
              .filter((q) => q.verified)
              .map((q) => `<blockquote class="quote text-sm italic mt-2">&ldquo;${escapeHtml(q.text)}&rdquo; <span class="not-italic text-slate">-- ${escapeHtml(q.expert_label || "")} ${escapeHtml(q.label || "")}</span></blockquote>`)
              .join("")
          }
        </div>
      </div>`
    )
    .join("");

  return `
  <div>
    <p class="text-sm text-slate mb-4">Ask anything across all three transcripts. Answers are grounded only in what was said.</p>
    <div id="chat-history">${history}</div>
    <div class="flex gap-2 mt-4">
      <input id="chat-input" type="text" placeholder="Ask a question across all transcripts..."
             class="flex-1 border border-line rounded px-3 py-2 text-sm bg-white" />
      <button id="chat-send" class="bg-accent text-white text-sm font-medium px-4 py-2 rounded hover:opacity-90">Ask</button>
    </div>
    <p id="chat-error" class="text-xs text-warn mt-2 hidden"></p>
  </div>`;
}

function wireChatTab() {
  const input = el("chat-input");
  const sendBtn = el("chat-send");
  const errorP = el("chat-error");

  const send = async () => {
    const question = input.value.trim();
    if (!question) return;
    errorP.classList.add("hidden");
    sendBtn.disabled = true;
    sendBtn.textContent = "Asking...";

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis_id: state.analysisId,
          question,
          model: el("model-select").value,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Could not answer.");

      state.chatHistory.push({
        question,
        answer: data.answer || "",
        citations: data.citations || [],
        quotes: data.quotes || [],
      });
      input.value = "";
      renderTabContent();
      wireChatTab(); // re-wire after re-render
      el("chat-input").focus();
    } catch (e) {
      errorP.textContent = e.message || "Something went wrong.";
      errorP.classList.remove("hidden");
    } finally {
      sendBtn.disabled = false;
      sendBtn.textContent = "Ask";
    }
  };

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") send();
  });
}

// --------------------------------------------------------------------------
// Raw transcripts tab
// --------------------------------------------------------------------------
function renderRawTranscriptsTab() {
  return `<div class="space-y-3">${state.transcripts.map(renderTranscriptAccordion).join("")}</div>`;
}

function renderTranscriptAccordion(t, idx) {
  const segments = t.segments
    .map(
      (s) => `
      <p class="text-sm mb-2">
        <code class="text-xs bg-accentSoft text-accent px-1 py-0.5 rounded">${escapeHtml(s.label)}</code>
        ${s.speaker ? `<span class="font-medium">${escapeHtml(s.speaker)}:</span>` : ""}
        ${escapeHtml(s.text)}
      </p>`
    )
    .join("");

  return `
  <details class="border border-line rounded-md bg-white/60">
    <summary class="cursor-pointer px-4 py-3 text-sm font-medium">
      ${escapeHtml(t.expert_label)} -- ${escapeHtml(t.source_filename)}
      <span class="text-xs text-slate font-normal">(${t.has_timestamps ? "timestamped" : "no timestamps detected"})</span>
    </summary>
    <div class="px-4 pb-4 max-h-96 overflow-y-auto">${segments}</div>
  </details>`;
}

boot();
