"""
Agent OS — Dashboard Server
FastAPI server that exposes kernel telemetry over WebSocket.
The frontend connects and receives live updates every second.
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logger = logging.getLogger("agent_os.dashboard")


class DashboardServer:
    """
    Attaches to a running Kernel and broadcasts state over WebSocket.
    Multiple browser tabs can connect simultaneously.
    """

    def __init__(self, kernel, broadcast_interval: float = 0.5):
        self._kernel = kernel
        self._interval = broadcast_interval
        self._connections: list[WebSocket] = []
        self._event_log: list[dict] = []   # last 100 events
        self._app = FastAPI(title="Agent OS Dashboard")
        self._setup_routes()

    @property
    def app(self):
        return self._app

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _setup_routes(self):

        @self._app.get("/", response_class=HTMLResponse)
        async def index():
            return HTMLResponse(DASHBOARD_HTML)

        @self._app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self._connections.append(ws)
            logger.info("Dashboard client connected (%d total)", len(self._connections))
            try:
                while True:
                    await asyncio.sleep(30)   # keep alive
            except WebSocketDisconnect:
                self._connections.remove(ws)
                logger.info("Dashboard client disconnected")

        @self._app.get("/api/state")
        async def get_state():
            return self._collect_state()

    # ------------------------------------------------------------------
    # Broadcast loop — call this as a background task
    # ------------------------------------------------------------------

    async def broadcast_loop(self):
        while True:
            if self._connections:
                state = self._collect_state()
                msg = json.dumps(state)
                dead = []
                for ws in self._connections:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self._connections.remove(ws)
            await asyncio.sleep(self._interval)

    def log_event(self, event_type: str, message: str, pid: str = ""):
        self._event_log.append({
            "time": time.strftime("%H:%M:%S"),
            "type": event_type,
            "message": message,
            "pid": pid,
        })
        if len(self._event_log) > 100:
            self._event_log.pop(0)

    # ------------------------------------------------------------------
    # State collection
    # ------------------------------------------------------------------

    def _collect_state(self) -> dict:
        kernel = self._kernel
        stats = kernel.stats
        ipc = kernel.ipc_stats
        mem = kernel.memory_stats

        procs = []
        for p in kernel.ps():
            procs.append({
                "pid": p.pid,
                "name": p.name,
                "state": p.state.value,
                "priority": p.priority.name,
                "tokens_used": p.budget.used,
                "tokens_total": p.budget.total_allocated,
                "tokens_pct": round(p.budget.utilization * 100, 1),
                "cpu_time": round(p.cpu_time, 2),
                "switches": p.context_switches,
                "result": str(p.result)[:80] if p.result else None,
                "error": p.error,
            })

        return {
            "ts": time.time(),
            "processes": procs,
            "stats": {
                "ticks": stats.tick,
                "switches": stats.context_switches,
                "tokens_total": stats.total_tokens_used,
                "llm_calls": stats.total_llm_calls,
                "uptime": round(stats.uptime, 1),
                "agents_spawned": stats.total_agents_spawned,
                "agents_dead": stats.total_agents_dead,
                "ipc_sent": ipc.get("messages_sent", 0),
                "ipc_dropped": ipc.get("messages_dropped", 0),
            },
            "memory": mem,
            "events": self._event_log[-20:],
        }


# ------------------------------------------------------------------
# Dashboard HTML — single file, no build step needed
# ------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent OS — Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #0a0e1a;
    color: #c9d1d9;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 13px;
    min-height: 100vh;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    border-bottom: 1px solid #1e2d40;
    background: #0d1117;
  }

  .logo {
    font-size: 16px;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: 2px;
  }

  .logo span { color: #3fb950; }

  .status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
  }

  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #3fb950;
    animation: pulse 2s infinite;
  }

  .dot.offline { background: #f85149; animation: none; }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 1px;
    background: #1e2d40;
    border-bottom: 1px solid #1e2d40;
  }

  .stat-card {
    background: #0d1117;
    padding: 16px 20px;
    text-align: center;
  }

  .stat-value {
    font-size: 28px;
    font-weight: 700;
    color: #58a6ff;
    line-height: 1;
    margin-bottom: 4px;
  }

  .stat-label {
    font-size: 10px;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .main {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 0;
    height: calc(100vh - 120px);
  }

  .panel {
    background: #0d1117;
    border-right: 1px solid #1e2d40;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .panel-header {
    padding: 10px 16px;
    border-bottom: 1px solid #1e2d40;
    font-size: 11px;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: #0a0e1a;
    flex-shrink: 0;
  }

  .panel-body {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }

  /* Process table */
  .proc-row {
    display: grid;
    grid-template-columns: 80px 140px 90px 80px 1fr 60px;
    gap: 8px;
    align-items: center;
    padding: 8px 16px;
    border-bottom: 1px solid #0d1421;
    transition: background 0.15s;
  }

  .proc-row:hover { background: #161b22; }

  .proc-row.header {
    font-size: 10px;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid #1e2d40;
    padding: 6px 16px;
  }

  .pid { color: #6e7681; font-size: 11px; }
  .name { color: #e6edf3; font-weight: 600; }

  .state-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .state-running  { background: #1a4a1a; color: #3fb950; border: 1px solid #3fb950; }
  .state-ready    { background: #1a3a4a; color: #58a6ff; border: 1px solid #58a6ff; }
  .state-blocked  { background: #4a3a1a; color: #d29922; border: 1px solid #d29922; }
  .state-sleeping { background: #2a1a4a; color: #bc8cff; border: 1px solid #bc8cff; }
  .state-zombie   { background: #3a1a1a; color: #f85149; border: 1px solid #f85149; }
  .state-dead     { background: #1a1a1a; color: #484f58; border: 1px solid #30363d; }

  .priority-HIGH       { color: #f85149; }
  .priority-NORMAL     { color: #58a6ff; }
  .priority-LOW        { color: #6e7681; }
  .priority-BACKGROUND { color: #484f58; }

  /* Token bar */
  .token-bar-wrap { position: relative; }
  .token-bar-bg {
    height: 6px;
    background: #1e2d40;
    border-radius: 3px;
    overflow: hidden;
  }
  .token-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
    transition: width 0.4s ease;
  }
  .token-bar-fill.high { background: linear-gradient(90deg, #b91c1c, #f85149); }
  .token-label {
    font-size: 10px;
    color: #6e7681;
    margin-top: 2px;
  }

  .switches { color: #6e7681; font-size: 11px; text-align: center; }

  /* Right panel — event log */
  .right-panels {
    display: flex;
    flex-direction: column;
    background: #0a0e1a;
  }

  .event-row {
    display: grid;
    grid-template-columns: 60px 70px 1fr;
    gap: 6px;
    padding: 5px 12px;
    border-bottom: 1px solid #0d1421;
    font-size: 11px;
    line-height: 1.4;
  }

  .event-time { color: #484f58; }
  .event-type-SPAWN  { color: #3fb950; }
  .event-type-DONE   { color: #58a6ff; }
  .event-type-KILL   { color: #f85149; }
  .event-type-TOOL   { color: #d29922; }
  .event-type-IPC    { color: #bc8cff; }
  .event-type-INFO   { color: #6e7681; }
  .event-msg { color: #8b949e; word-break: break-all; }

  /* Uptime ticker */
  .uptime { color: #3fb950; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: #1e2d40; border-radius: 2px; }

  .empty-state {
    padding: 40px 16px;
    text-align: center;
    color: #484f58;
    font-size: 12px;
  }

  /* Token chart */
  .chart-wrap {
    padding: 12px 16px;
    flex-shrink: 0;
    border-top: 1px solid #1e2d40;
  }

  .chart-label {
    font-size: 10px;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }

  .sparkline {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 40px;
  }

  .spark-bar {
    flex: 1;
    background: #1f6feb;
    border-radius: 1px 1px 0 0;
    min-height: 2px;
    transition: height 0.3s ease;
    opacity: 0.8;
  }
</style>
</head>
<body>

<header>
  <div class="logo">AGENT<span>OS</span> <span style="color:#6e7681;font-size:12px;font-weight:400">// kernel dashboard</span></div>
  <div class="status">
    <div class="dot" id="conn-dot"></div>
    <span id="conn-label">Connecting...</span>
    &nbsp;&nbsp;
    <span style="color:#6e7681">uptime</span>&nbsp;
    <span class="uptime" id="uptime">0s</span>
  </div>
</header>

<div class="grid">
  <div class="stat-card">
    <div class="stat-value" id="s-ticks">0</div>
    <div class="stat-label">Scheduler Ticks</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" id="s-tokens" style="color:#3fb950">0</div>
    <div class="stat-label">Total Tokens</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" id="s-llm" style="color:#d29922">0</div>
    <div class="stat-label">LLM Calls</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" id="s-switches" style="color:#bc8cff">0</div>
    <div class="stat-label">Context Switches</div>
  </div>
</div>

<div class="main">
  <!-- Process Table -->
  <div class="panel">
    <div class="panel-header">⬡ Process Table</div>
    <div class="proc-row header">
      <span>PID</span>
      <span>Name</span>
      <span>State</span>
      <span>Priority</span>
      <span>Token Budget</span>
      <span>Switches</span>
    </div>
    <div class="panel-body" id="proc-table">
      <div class="empty-state">Waiting for agents...</div>
    </div>

    <div class="chart-wrap">
      <div class="chart-label">Token burn (last 30 ticks)</div>
      <div class="sparkline" id="sparkline"></div>
    </div>
  </div>

  <!-- Right: Event Log -->
  <div class="right-panels">
    <div class="panel" style="flex:1;border-right:none;">
      <div class="panel-header">⬡ Event Log</div>
      <div class="panel-body" id="event-log">
        <div class="empty-state">No events yet...</div>
      </div>
    </div>
  </div>
</div>

<script>
  const tokenHistory = new Array(30).fill(0);
  let lastTokenTotal = 0;
  let ws;

  function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);

    ws.onopen = () => {
      document.getElementById('conn-dot').classList.remove('offline');
      document.getElementById('conn-label').textContent = 'Connected';
    };

    ws.onclose = () => {
      document.getElementById('conn-dot').classList.add('offline');
      document.getElementById('conn-label').textContent = 'Reconnecting...';
      setTimeout(connect, 2000);
    };

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      updateDashboard(data);
    };
  }

  // Also poll REST for initial state
  function pollRest() {
    fetch('/api/state').then(r => r.json()).then(updateDashboard).catch(() => {});
    setTimeout(pollRest, 1000);
  }

  function updateDashboard(data) {
    const s = data.stats;

    // Stats bar
    document.getElementById('s-ticks').textContent = s.ticks;
    document.getElementById('s-tokens').textContent = s.tokens_total;
    document.getElementById('s-llm').textContent = s.llm_calls;
    document.getElementById('s-switches').textContent = s.switches;
    document.getElementById('uptime').textContent = s.uptime + 's';

    // Token sparkline
    const delta = s.tokens_total - lastTokenTotal;
    lastTokenTotal = s.tokens_total;
    tokenHistory.push(delta);
    tokenHistory.shift();
    renderSparkline();

    // Process table
    renderProcessTable(data.processes);

    // Event log
    renderEventLog(data.events);
  }

  function renderProcessTable(procs) {
    const el = document.getElementById('proc-table');
    if (!procs || procs.length === 0) {
      el.innerHTML = '<div class="empty-state">No processes running</div>';
      return;
    }
    el.innerHTML = procs.map(p => {
      const pct = p.tokens_pct;
      const isHigh = pct > 75;
      return `
        <div class="proc-row">
          <span class="pid">${p.pid}</span>
          <span class="name">${p.name}</span>
          <span><span class="state-badge state-${p.state}">${p.state}</span></span>
          <span class="priority-${p.priority}">${p.priority}</span>
          <span class="token-bar-wrap">
            <div class="token-bar-bg">
              <div class="token-bar-fill ${isHigh ? 'high' : ''}" style="width:${pct}%"></div>
            </div>
            <div class="token-label">${p.tokens_used} / ${p.tokens_total} (${pct}%)</div>
          </span>
          <span class="switches">${p.switches}</span>
        </div>
      `;
    }).join('');
  }

  function renderEventLog(events) {
    const el = document.getElementById('event-log');
    if (!events || events.length === 0) {
      el.innerHTML = '<div class="empty-state">No events yet...</div>';
      return;
    }
    const reversed = [...events].reverse();
    el.innerHTML = reversed.map(e => `
      <div class="event-row">
        <span class="event-time">${e.time}</span>
        <span class="event-type-${e.type}">${e.type}</span>
        <span class="event-msg">${e.message}</span>
      </div>
    `).join('');
  }

  function renderSparkline() {
    const el = document.getElementById('sparkline');
    const max = Math.max(...tokenHistory, 1);
    el.innerHTML = tokenHistory.map(v => {
      const h = Math.round((v / max) * 38) + 2;
      return `<div class="spark-bar" style="height:${h}px"></div>`;
    }).join('');
  }

  connect();
  pollRest();
</script>
</body>
</html>
"""
