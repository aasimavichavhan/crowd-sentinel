// =========================================================================
// CROWD SENTINEL - Real-Time Dashboard Client Controller
// =========================================================================

let ws = null;
let chart = null;
let soundEnabled = false;
let audioCtx = null;
let alertCount = 0;
const seenAlertIds = new Set();

// Chart rolling history
const MAX_CHART_POINTS = 40;
const chartLabels = [];
const chartData = [];

document.addEventListener("DOMContentLoaded", () => {
  initChart();
  fetchVideoList();
  connectWebSocket();
  bindUIEvents();
});

// =========================================================================
// 1. Chart.js Real-Time Trajectory Initialization
// =========================================================================
function initChart() {
  const ctx = document.getElementById("riskChart").getContext("2d");
  
  // Custom gradient for risk line
  const gradient = ctx.createLinearGradient(0, 0, 0, 180);
  gradient.addColorStop(0, "rgba(239, 68, 68, 0.4)");
  gradient.addColorStop(0.5, "rgba(245, 158, 11, 0.2)");
  gradient.addColorStop(1, "rgba(16, 185, 129, 0.05)");

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: chartLabels,
      datasets: [
        {
          label: "Average Risk Score",
          data: chartData,
          borderColor: "#38bdf8",
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.35,
          fill: true,
          backgroundColor: gradient,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: "index",
          intersect: false,
          callbacks: {
            label: (ctx) => `Overall Risk: ${ctx.parsed.y}/100`
          }
        }
      },
      scales: {
        x: {
          display: true,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b", font: { size: 9 }, maxTicksLimit: 6 }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: "rgba(255, 255, 255, 0.08)" },
          ticks: {
            stepSize: 25,
            color: "#64748b",
            font: { size: 9 },
            callback: (v) => v
          }
        }
      }
    }
  });
}

function updateChart(timeLabel, score, tier) {
  if (!chart) return;

  if (chartLabels.length >= MAX_CHART_POINTS) {
    chartLabels.shift();
    chartData.shift();
  }

  chartLabels.push(timeLabel);
  chartData.push(score);

  // Dynamically color line based on overall risk
  if (tier === "Risky") {
    chart.data.datasets[0].borderColor = "#ef4444";
  } else if (tier === "Dense") {
    chart.data.datasets[0].borderColor = "#f97316";
  } else if (tier === "Moderate") {
    chart.data.datasets[0].borderColor = "#f59e0b";
  } else {
    chart.data.datasets[0].borderColor = "#10b981";
  }

  chart.update();

  const tag = document.getElementById("chart-latest-score");
  if (tag) {
    tag.textContent = `Risk: ${score.toFixed(1)}/100`;
  }
}

// =========================================================================
// 2. WebSocket Telemetry Streaming
// =========================================================================
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

  ws = new WebSocket(wsUrl);

  const statusIndicator = document.getElementById("status-indicator");
  const statusLabel = document.getElementById("status-label");
  const idleOverlay = document.getElementById("stream-idle");

  ws.onopen = () => {
    statusIndicator.className = "status-indicator live";
    statusLabel.textContent = "STREAM ACTIVE";
    if (idleOverlay) idleOverlay.style.display = "none";
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "frame") {
        handleFrameUpdate(data);
      }
    } catch (e) {
      console.error("WS Parse error:", e);
    }
  };

  ws.onclose = () => {
    statusIndicator.className = "status-indicator";
    statusLabel.textContent = "RECONNECTING...";
    if (idleOverlay) idleOverlay.style.display = "flex";
    setTimeout(connectWebSocket, 1500);
  };

  ws.onerror = (err) => {
    console.error("WS Error:", err);
    ws.close();
  };
}

// =========================================================================
// 3. Process Live Frame Telemetry
// =========================================================================
function handleFrameUpdate(payload) {
  // 1. Update Video Frame Image
  const liveImg = document.getElementById("live-feed");
  if (liveImg && payload.frame) {
    liveImg.src = payload.frame;
  }

  // 2. Floating HUD Telemetry
  document.getElementById("hud-total-people").textContent = payload.total_people || 0;
  document.getElementById("hud-fps").textContent = `${payload.fps || 0} FPS`;

  const tier = payload.highest_risk || "Normal";
  const tierEl = document.getElementById("hud-risk-tier");
  tierEl.textContent = tier.toUpperCase();
  tierEl.className = `pill-val risk-val ${tier.toLowerCase()}`;

  // 3. Chart Update
  const timeStr = `${payload.current_time_sec.toFixed(1)}s`;
  updateChart(timeStr, payload.average_risk || 0, tier);

  // 4. Update Zone Matrix
  if (payload.zones) {
    renderZoneMatrix(payload.zones);
  }

  // 5. Handle Alerts (including automated email dispatch events)
  if (payload.new_alerts && payload.new_alerts.length > 0) {
    payload.new_alerts.forEach(alert => {
      if (!seenAlertIds.has(alert.id)) {
        seenAlertIds.add(alert.id);
        appendAlertToLog(alert);
        triggerAlarmChime();

        // Check if this is an email dispatch event from the backend
        if (alert.zone_id === "SYSTEM" && alert.message.includes("Email alert sent")) {
          addDispatchLogEntry(alert.message, true);
          // Pulse the email status pill
          const emailPill = document.getElementById("email-status-pill");
          if (emailPill) {
            emailPill.classList.add("dispatching");
            setTimeout(() => emailPill.classList.remove("dispatching"), 3000);
          }
          const emailStatus = document.getElementById("email-dispatch-status");
          if (emailStatus) {
            emailStatus.textContent = "EMAIL SENT";
            emailStatus.className = "status-tag dispatched";
            setTimeout(() => {
              emailStatus.textContent = "AUTO-ACTIVE";
              emailStatus.className = "status-tag active";
            }, 5000);
          }
        }
      }
    });
  }
}

// =========================================================================
// 4. Render Zone Matrix Cards
// =========================================================================
function renderZoneMatrix(zones) {
  const container = document.getElementById("zone-grid-container");
  if (!container) return;

  // Build or update cards
  let html = "";
  zones.forEach(z => {
    const tier = z.risk_level.toLowerCase();
    const isPredictive = z.predicted_risk_30s === "Risky" || z.predicted_risk_30s === "Dense";

    html += `
      <div class="zone-card tier-${tier}">
        <div class="zone-card-top">
          <span class="zone-card-id">${z.zone_id}</span>
          <span class="zone-risk-tag ${tier}">${z.risk_level}</span>
        </div>
        <div class="zone-stats-row">
          <span>Count: <strong class="zone-stat-val">${z.count} pax</strong></span>
          <span>Turbulence: <strong class="zone-stat-val">${z.turbulence_score}%</strong></span>
        </div>
        <div class="zone-stats-row">
          <span>Density: <strong class="zone-stat-val">${z.density_score}/100</strong></span>
          <span>Risk: <strong class="zone-stat-val">${z.risk_score}</strong></span>
        </div>
        ${isPredictive ? `
          <div class="trend-badge">
            <span>⚠️ Surge ~30s (+${z.trend_slope}/s)</span>
          </div>
        ` : ''}
      </div>
    `;
  });

  container.innerHTML = html;
}

// =========================================================================
// 5. Alert Management & Feed
// =========================================================================
function appendAlertToLog(alert) {
  const feed = document.getElementById("alert-feed-list");
  const emptyMsg = document.getElementById("empty-alert-msg");
  if (emptyMsg) emptyMsg.style.display = "none";

  alertCount++;
  document.getElementById("alert-counter").textContent = `${alertCount} ALERTS`;

  const item = document.createElement("div");
  item.className = "alert-item";
  item.innerHTML = `
    <div class="alert-top">
      <span class="alert-zone">${alert.zone_id} — ${alert.zone_name}</span>
      <span class="alert-time">${alert.timestamp}</span>
    </div>
    <div class="alert-message">${alert.message}</div>
    <div class="alert-stats-tag">
      <span>Headcount: ${alert.people_count} pax</span>
      <span>Turbulence: ${alert.turbulence_score}%</span>
      <span>Score: ${alert.risk_score}/100</span>
    </div>
  `;

  feed.insertBefore(item, feed.firstChild);
}

// Synthesized Audio Warning Chime (Web Audio API)
function triggerAlarmChime() {
  if (!soundEnabled) return;
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(880, audioCtx.currentTime); // High pitch alert
    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.3);
    gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.3);
  } catch (e) {
    console.warn("Audio chime prevented:", e);
  }
}

// =========================================================================
// 6. Video Selector & Stream Controls
// =========================================================================
async function fetchVideoList() {
  try {
    const res = await fetch("/api/videos");
    const data = await res.json();
    const select = document.getElementById("video-select");
    select.innerHTML = "";

    data.videos.forEach(v => {
      const opt = document.createElement("option");
      opt.value = v.filename;
      opt.textContent = `${v.label} [${v.tag}] (${v.duration_sec}s)`;
      if (data.current_video === v.filename) {
        opt.selected = true;
      }
      select.appendChild(opt);
    });

    select.addEventListener("change", async (e) => {
      const chosen = e.target.value;
      if (chosen) {
        await fetch("/api/control/play", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video: chosen })
        });
        resetFeedUI();
      }
    });
  } catch (e) {
    console.error("Failed to load video list:", e);
  }
}

function resetFeedUI() {
  chartLabels.length = 0;
  chartData.length = 0;
  if (chart) chart.update();
  seenAlertIds.clear();
  alertCount = 0;
  document.getElementById("alert-counter").textContent = "0 ALERTS";
  const feed = document.getElementById("alert-feed-list");
  feed.innerHTML = `
    <div class="empty-alerts" id="empty-alert-msg">
      <div class="shield-check">✓</div>
      <p>All zones within safe parameters.</p>
      <span class="empty-sub">Continuous optical flow & density tracking active.</span>
    </div>
  `;
}

function bindUIEvents() {
  // Play / Pause / Restart
  document.getElementById("btn-play").addEventListener("click", async () => {
    const videoSelect = document.getElementById("video-select");
    await fetch("/api/control/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video: videoSelect.value })
    });
  });

  document.getElementById("btn-pause").addEventListener("click", async () => {
    await fetch("/api/control/pause", { method: "POST" });
  });

  document.getElementById("btn-restart").addEventListener("click", async () => {
    await fetch("/api/control/restart", { method: "POST" });
    resetFeedUI();
  });

  // Audio Toggle
  const audioBtn = document.getElementById("audio-toggle");
  const audioIcon = document.getElementById("audio-icon");
  audioBtn.addEventListener("click", () => {
    soundEnabled = !soundEnabled;
    audioIcon.textContent = soundEnabled ? "🔊" : "🔇";
    audioBtn.title = soundEnabled ? "Mute Alarm Chime" : "Enable Alarm Chime";
    if (soundEnabled && !audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
  });

  // Sensitivity Presets
  const segments = document.querySelectorAll(".segmented-control .segment");
  segments.forEach(btn => {
    btn.addEventListener("click", async () => {
      segments.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const preset = btn.getAttribute("data-preset");
      await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset })
      });
      addDispatchLogEntry(`Sensitivity preset adjusted to: ${preset.toUpperCase()}`);
    });
  });
}

function addDispatchLogEntry(msg, isDispatched = false) {
  const log = document.getElementById("dispatch-log");
  if (!log) return;
  const time = new Date().toLocaleTimeString();
  const entry = document.createElement("div");
  entry.className = `log-entry ${isDispatched ? 'dispatched' : ''}`;
  entry.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-msg">${msg}</span>`;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}
