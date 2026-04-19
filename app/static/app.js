/* ============================================================
   CardioRisk AI — Frontend Logic
   Handles form → API → rich animated results
   ============================================================ */

const API_BASE = "http://localhost:8000";

// ── DOM refs ──────────────────────────────────────────────
const form        = document.getElementById("predict-form");
const predictBtn  = document.getElementById("predict-btn");
const resetBtn    = document.getElementById("reset-btn");

const idleState     = document.getElementById("idle-state");
const loadingState  = document.getElementById("loading-state");
const resultState   = document.getElementById("result-state");

// ── Slider ↔ Number input sync ────────────────────────────
const sliders = document.querySelectorAll(".slider");
sliders.forEach(slider => {
  const numId  = slider.id.replace("-slider", "");
  const numInput = document.getElementById(numId);
  if (!numInput) return;

  // Sync slider fill (CSS custom property)
  const updateFill = (val, min, max) => {
    const pct = ((val - min) / (max - min)) * 100;
    slider.style.setProperty("--val", pct + "%");
    slider.style.background =
      `linear-gradient(90deg, var(--accent-blue) ${pct}%, rgba(255,255,255,0.08) ${pct}%)`;
  };

  slider.addEventListener("input", () => {
    numInput.value = slider.value;
    updateFill(slider.value, slider.min, slider.max);
  });
  numInput.addEventListener("input", () => {
    const v = Math.min(Math.max(numInput.value, slider.min), slider.max);
    slider.value = v;
    updateFill(v, slider.min, slider.max);
  });

  // Init
  updateFill(slider.value, slider.min, slider.max);
});

// ── Show/hide states ──────────────────────────────────────
function showState(state) {
  [idleState, loadingState, resultState].forEach(el => el.classList.add("hidden"));
  state.classList.remove("hidden");
}

// ── Collect form values ───────────────────────────────────
function collectPayload() {
  const data = {};
  const radio = (name) => {
    const el = form.querySelector(`input[name="${name}"]:checked`);
    return el ? parseFloat(el.value) : null;
  };
  const num = (id) => {
    const el = document.getElementById(id);
    return el ? parseFloat(el.value) : null;
  };
  const sel = (id) => {
    const el = document.getElementById(id);
    return el && el.value ? parseFloat(el.value) : null;
  };

  data.male            = radio("male");
  data.age             = num("age");
  data.education       = sel("education");
  data.currentSmoker   = radio("currentSmoker");
  data.cigsPerDay      = num("cigsPerDay");
  data.BPMeds          = radio("BPMeds");
  data.prevalentStroke = radio("prevalentStroke");
  data.prevalentHyp    = radio("prevalentHyp");
  data.diabetes        = radio("diabetes");
  data.totChol         = num("totChol");
  data.sysBP           = num("sysBP");
  data.diaBP           = num("diaBP");
  data.BMI             = num("BMI");
  data.heartRate       = num("heartRate");
  data.glucose         = num("glucose");

  return data;
}

// ── Validate ──────────────────────────────────────────────
function validate(data) {
  const missing = Object.entries(data)
    .filter(([, v]) => v === null || isNaN(v))
    .map(([k]) => k);
  return missing;
}

// ── Animate probability bars ──────────────────────────────
function animateBars(probChd, probNoChd) {
  setTimeout(() => {
    document.getElementById("bar-no-chd").style.width = `${probNoChd * 100}%`;
    document.getElementById("bar-chd").style.width    = `${probChd * 100}%`;
    document.getElementById("val-no-chd").textContent = `${(probNoChd * 100).toFixed(1)}%`;
    document.getElementById("val-chd").textContent    = `${(probChd * 100).toFixed(1)}%`;
  }, 100);
}

// ── Animate Gauge ─────────────────────────────────────────
function animateGauge(score) {
  const arc    = document.getElementById("gauge-arc");
  const needle = document.getElementById("gauge-needle");
  const scoreEl= document.getElementById("gauge-score");
  const levelEl= document.getElementById("gauge-level");

  const totalLen = 267;                  // arc length
  const pct    = Math.min(score / 100, 1);
  const offset = totalLen * (1 - pct);

  const angleDeg = -90 + pct * 180;     // -90° (left) → +90° (right)

  // Determine colours
  let colour, level;
  if (score < 25)       { colour = "#22c55e"; level = "🟢 Low Risk"; }
  else if (score < 50)  { colour = "#eab308"; level = "🟡 Moderate Risk"; }
  else if (score < 75)  { colour = "#f97316"; level = "🟠 High Risk"; }
  else                   { colour = "#ef4444"; level = "🔴 Very High Risk"; }

  setTimeout(() => {
    arc.style.strokeDashoffset = offset;
    needle.setAttribute("transform", `rotate(${angleDeg} 100 95)`);
    levelEl.textContent = level;
    levelEl.style.color = colour;

    // Count-up animation
    let start = 0;
    const end = Math.round(score);
    const step = end / 40;
    const timer = setInterval(() => {
      start = Math.min(start + step, end);
      scoreEl.textContent = Math.round(start);
      if (start >= end) clearInterval(timer);
    }, 20);
  }, 200);
}

// ── Render Feature Bars ───────────────────────────────────
function renderFeatureBars(features) {
  const container = document.getElementById("feature-bars");
  container.innerHTML = "";

  // Take top 8
  const top = features.slice(0, 8);
  const maxPct = top[0]?.contribution_pct || 1;

  top.forEach((f, i) => {
    const barPct = (f.contribution_pct / maxPct) * 100;
    const rankClass = i === 0 ? "rank-1" : i === 1 ? "rank-2" : i === 2 ? "rank-3" : "rank-n";
    const delay = `${i * 0.07}s`;

    const row = document.createElement("div");
    row.className = "feat-row";
    row.innerHTML = `
      <div class="feat-meta">
        <span class="feat-name">${escapeHtml(f.label)} <span style="color:var(--text-muted);font-size:.72rem">(${f.value}${f.unit ? " " + f.unit : ""})</span></span>
        <span class="feat-pct">${f.contribution_pct.toFixed(1)}%</span>
      </div>
      <div class="feat-bar-wrap">
        <div class="feat-bar-fill ${rankClass}" style="width:0%; --delay:${delay}"></div>
      </div>
    `;
    container.appendChild(row);
    // Trigger animation
    requestAnimationFrame(() => {
      setTimeout(() => {
        row.querySelector(".feat-bar-fill").style.width = `${barPct}%`;
        row.querySelector(".feat-bar-fill").style.transition =
          `width 0.8s cubic-bezier(0.4,0,0.2,1) ${delay}`;
      }, 50);
    });
  });
}

// ── Render Verdict ────────────────────────────────────────
function renderVerdict(result) {
  const card    = document.getElementById("verdict-card");
  const icon    = document.getElementById("verdict-icon");
  const label   = document.getElementById("verdict-label");
  const sub     = document.getElementById("verdict-sublabel");
  const badge   = document.getElementById("verdict-badge");

  if (result.prediction === 1) {
    card.className  = "verdict-card risk";
    icon.textContent= "⚠️";
    label.textContent = result.label;
    sub.textContent   = `${(result.probability_chd * 100).toFixed(1)}% probability · ${result.risk_level} Risk`;
    badge.textContent = `CHD Risk`;
  } else {
    card.className  = "verdict-card safe";
    icon.textContent= "✅";
    label.textContent = result.label;
    sub.textContent   = `${(result.probability_no_chd * 100).toFixed(1)}% probability of no CHD · ${result.risk_level} Risk`;
    badge.textContent = `Clear`;
  }
}

// ── Main Submit ───────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const payload = collectPayload();
  const missing = validate(payload);

  if (missing.length > 0) {
    alert(`Please fill in all fields.\nMissing: ${missing.join(", ")}`);
    return;
  }

  // Show loading
  showState(loadingState);
  predictBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Server error");
    }

    const result = await res.json();

    // Populate results
    renderVerdict(result);
    animateBars(result.probability_chd, result.probability_no_chd);
    animateGauge(result.risk_score);
    renderFeatureBars(result.top_features);
    document.getElementById("gauge-level").textContent = result.risk_level + " Risk";
    document.getElementById("ai-message-text").textContent = result.message;

    showState(resultState);

    // Scroll results into view on mobile
    document.getElementById("results-panel").scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    showState(idleState);
    alert(`❌ Prediction failed: ${err.message}\n\nMake sure the FastAPI server is running on http://localhost:8000`);
  } finally {
    predictBtn.disabled = false;
  }
});

// ── Reset ─────────────────────────────────────────────────
resetBtn.addEventListener("click", () => {
  form.reset();
  // Re-init sliders
  sliders.forEach(slider => {
    const numId   = slider.id.replace("-slider", "");
    const numInput= document.getElementById(numId);
    if (numInput) numInput.value = slider.defaultValue;
    const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.background =
      `linear-gradient(90deg, var(--accent-blue) ${pct}%, rgba(255,255,255,0.08) ${pct}%)`;
  });
  showState(idleState);
});

// ── Health check on load ──────────────────────────────────
(async () => {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      const badge = document.getElementById("model-badge");
      badge.textContent = `⚡ ${data.model} Ready`;
      badge.style.background = "rgba(34,197,94,0.15)";
    }
  } catch {
    const badge = document.getElementById("model-badge");
    badge.textContent = "⚠️ Server Offline";
    badge.style.background = "rgba(239,68,68,0.15)";
    badge.style.color = "#ef4444";
    badge.style.borderColor = "rgba(239,68,68,0.3)";
  }
})();

// ── Utility ───────────────────────────────────────────────
function escapeHtml(str) {
  return str.replace(/[&<>"']/g, c =>
    ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c])
  );
}
