"use strict";
// ============================================================================
// Orbit Wars — shared viewer core (game animation only, NO info panels).
// Ported verbatim from replay_view_experiments.html so visualize_local renders
// the game identically to the project viewer. The page-specific HTML wires
// the selection UI and calls loadReplayObject({meta, static, steps}).
//
// Expected replay shape (both official adapter + local replays produce this):
//   replay.static.initial_planets : [[pid, owner, x, y, r, ships, prod], ...]
//   replay.steps[t] = { t, planets:[[pid,owner,x,y,r,ships,prod],...],
//                          fleets:[[fid,owner,x,y,angle,ships,src],...] }
//   replay.meta = { winner, scores, n_turns, agent_a, agent_b, seed }
// ============================================================================

// ---- constants ----
const BOARD = 100.0, SUN_X = 50, SUN_Y = 50, SUN_R = 10;
const COLORS = {
  "0": "#3b82f6", "1": "#ef4444", "2": "#10b981", "3": "#f59e0b",
  "-1": "#94a3b8",
};
const COMET_COLOR = "#a78bfa";

// ---- state ----
let replay = null;
let curTurn = 0;
let playing = false;
let speedMultiplier = 0.25;

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const W = canvas.width, H = canvas.height;
const PAD = 4;
const scale = W / (BOARD + PAD * 2);

function worldToCanvas(x, y) {
  return [(x + PAD) * scale, (y + PAD) * scale];
}

// ============================================================================
// Game canvas (identical to the experiment viewer)
// ============================================================================
function clear() {
  ctx.fillStyle = "#050a15";
  ctx.fillRect(0, 0, W, H);
}
function drawSun() {
  const [cx, cy] = worldToCanvas(SUN_X, SUN_Y);
  const r = SUN_R * scale;
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
  g.addColorStop(0, "#fde047"); g.addColorStop(0.6, "#fbbf24");
  g.addColorStop(1, "rgba(252,211,77,0)");
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "rgba(252,165,165,0.35)";
  ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]);
}
function isComet(p) {
  if (!replay.static || !replay.static.initial_planets) return false;
  const initIds = new Set(replay.static.initial_planets.map(ip => ip[0]));
  return !initIds.has(p[0]);
}
function drawPlanets() {
  const step = replay.steps[curTurn];
  for (const p of step.planets) {
    const [pid, owner, x, y, radius, ships, prod] = p;
    const [cx, cy] = worldToCanvas(x, y);
    const r = radius * scale;
    const comet = isComet(p);
    const color = comet ? COMET_COLOR : (COLORS[String(owner)] || COLORS["-1"]);
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.3)"; ctx.lineWidth = 0.5;
    ctx.stroke();
    ctx.fillStyle = "white";
    ctx.font = "bold 13px monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(String(ships), cx, cy);
    if (prod > 0 && !comet) {
      ctx.fillStyle = "#fbbf24";
      ctx.font = "bold 11px monospace";
      ctx.fillText(`+${prod}`, cx, cy + r + 9);
    }
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.font = "bold 11px monospace";
    ctx.fillText(`p${pid}`, cx, cy - r - 7);
  }
}
function drawFleets() {
  const step = replay.steps[curTurn];
  for (const ft of step.fleets) {
    // env schema: [id, owner, x, y, angle, from_planet_id, ships]
    const [fid, owner, x, y, angle, src, ships] = ft;
    const [cx, cy] = worldToCanvas(x, y);
    const color = COLORS[String(owner)] || COLORS["-1"];
    const size = Math.max(4, Math.min(10, 4 + Math.log(Math.max(ships, 1)) * 0.7)) * (scale / 8);
    ctx.save();
    ctx.translate(cx, cy); ctx.rotate(angle);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(size * 1.5, 0);
    ctx.lineTo(-size * 0.7, -size * 0.7);
    ctx.lineTo(-size * 0.7, size * 0.7);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.4)"; ctx.lineWidth = 0.5;
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = color;
    ctx.font = "bold 9px monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(String(ships), cx + size + 2, cy - size);
  }
}
// HUD removed: turn + per-player totals (incl. all 4 players + overage) already
// live at the top of the side panel via renderSidePanel, so we no longer draw
// them on the game canvas — nothing overlaps the fleets/planets anymore.

// ============================================================================
// Per-turn series (ships / production / overage) + side panel
// Computed once per replay (not per frame); charts only redraw the marker.
// ============================================================================
let _series = null;        // {nAgents, ships:[[..]xA], prod, overage, N}
let _fleetLaunch = {};      // fid -> turn it first appeared (for "in flight Nt")

function _inferAgents() {
  let mx = 1;
  for (const st of replay.steps) {
    for (const p of st.planets) if (p[1] > mx) mx = p[1];
    for (const f of st.fleets) if (f[1] > mx) mx = f[1];
  }
  return mx + 1;
}
function _computeFleetLaunch() {
  _fleetLaunch = {};
  for (let t = 0; t < replay.steps.length; t++) {
    for (const f of replay.steps[t].fleets) {
      if (!(f[0] in _fleetLaunch)) _fleetLaunch[f[0]] = t;
    }
  }
}
const ACT_TIMEOUT = 1.0;     // env per-turn budget (s); time over it draws from the overage bank
const OVERAGE_BANK_S = 60.0; // starting overage bank (s)
function _computeSeries() {
  const steps = replay.steps, N = steps.length;
  const nA = replay.meta.n_agents || 2;
  const ships = Array.from({ length: nA }, () => new Array(N).fill(0));
  const prod = Array.from({ length: nA }, () => new Array(N).fill(0));
  const overage = Array.from({ length: nA }, () => new Array(N).fill(null));
  // per-turn agent time: local replays carry step.time = {"0":sec,"1":sec} (steps 1+);
  // official replays lack it. time is charted in ms. The remaining overage bank is
  // DERIVED here (60s − running Σ max(0, time−1s), clamped ≥0): the env's own
  // remainingOverageTime is a constant 60 in our local runner (it never times our
  // pre-computed-action agents), so the bank must come from our measured per-turn time.
  const hasTime = steps.some(st => st.time);
  const time = hasTime ? Array.from({ length: nA }, () => new Array(N).fill(null)) : null;
  const overageBank = hasTime ? Array.from({ length: nA }, () => new Array(N).fill(null)) : null;
  const bank = new Array(nA).fill(OVERAGE_BANK_S);
  for (let t = 0; t < N; t++) {
    const st = steps[t];
    for (const p of st.planets) { const o = p[1]; if (o >= 0 && o < nA) { ships[o][t] += p[5]; prod[o][t] += p[6]; } }
    for (const f of st.fleets) { const o = f[1]; if (o >= 0 && o < nA) ships[o][t] += f[6]; }  // f[6]=ships
    if (st.overage) for (let a = 0; a < nA; a++) if (st.overage[a] != null) overage[a][t] = st.overage[a];
    if (hasTime && st.time) for (let a = 0; a < nA; a++) {
      const v = st.time[String(a)];   // seconds
      if (v != null) { time[a][t] = v * 1000; bank[a] = Math.max(0, bank[a] - Math.max(0, v - ACT_TIMEOUT)); }
    }
    if (hasTime) for (let a = 0; a < nA; a++) overageBank[a][t] = bank[a];  // seconds, running
  }
  const timeStats = hasTime ? time.map(arr => {
    const vals = arr.filter(v => v != null);
    if (!vals.length) return null;
    return { avg: vals.reduce((s, x) => s + x, 0) / vals.length, min: Math.min(...vals), max: Math.max(...vals) };
  }) : null;
  _series = { nAgents: nA, ships, prod, overage, time, overageBank, timeStats, N };
}

// Multi-player line chart (one line per player, in player color). Tolerates nulls.
function _renderMultiLine(canvasId, seriesArr) {
  const cv = document.getElementById(canvasId);
  if (!cv || !replay) return;
  const c = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  c.clearRect(0, 0, W, H);
  const N = (seriesArr[0] || []).length;
  if (!N) return;
  let maxV = 1;
  for (const s of seriesArr) for (const v of s) if (v != null && v > maxV) maxV = v;
  const padL = 36, padR = 8, padT = 8, padB = 16;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  c.strokeStyle = "#1e293b"; c.fillStyle = "#64748b"; c.font = "10px monospace";
  for (let i = 0; i <= 4; i++) {
    const y = padT + plotH * i / 4;
    c.beginPath(); c.moveTo(padL, y); c.lineTo(W - padR, y); c.stroke();
    c.fillText((maxV * (1 - i / 4)).toFixed(0), 2, y + 4);
  }
  const xstep = Math.max(1, Math.floor(N / 10));
  for (let t = 0; t < N; t += xstep) {
    const x = padL + plotW * t / Math.max(1, N - 1);
    c.fillText(t.toString(), x - 6, H - 4);
  }
  for (let pid = 0; pid < seriesArr.length; pid++) {
    const arr = seriesArr[pid];
    c.strokeStyle = COLORS[String(pid)] || "#888"; c.lineWidth = 1.5;
    c.beginPath(); let started = false;
    for (let t = 0; t < N; t++) {
      const v = arr[t]; if (v == null) { started = false; continue; }
      const x = padL + plotW * t / Math.max(1, N - 1);
      const y = padT + plotH * (1 - v / maxV);
      if (!started) { c.moveTo(x, y); started = true; } else c.lineTo(x, y);
    }
    c.stroke();
  }
  if (curTurn < N) {
    const x = padL + plotW * curTurn / Math.max(1, N - 1);
    c.strokeStyle = "#22c55e"; c.lineWidth = 1.5;
    c.beginPath(); c.moveTo(x, padT); c.lineTo(x, H - padB); c.stroke();
    const vals = seriesArr.map((s, pid) => s[curTurn] != null ? `P${pid}=${(+s[curTurn]).toFixed(0)}` : null)
      .filter(Boolean).join(" ");
    c.fillStyle = "#22c55e";
    c.fillText(`t=${curTurn} ${vals}`, Math.min(x + 4, W - 190), padT + 10);
  }
}

// Normalize _aux.winprob to {seatIdxStr: [per-turn ...]}. Two shapes exist:
//   - dict {"0":[..],"1":[..],...} (current 2p runner + 4p ffa_battery; one key per RL seat)
//   - flat per-turn array (2p legacy — the RL side's line; seat unknown, mapped to "0")
function _winprobDict() {
  const wp = (typeof window !== "undefined") ? window.WINPROB : null;
  if (!wp) return null;
  if (Array.isArray(wp)) return wp.length ? { "0": wp } : null;
  const out = {};
  for (const k of Object.keys(wp)) if (Array.isArray(wp[k]) && wp[k].length) out[k] = wp[k];
  return Object.keys(out).length ? out : null;
}

function _renderWinprob(canvasId) {
  // RL value-head win-confidence (V(s)+1)/2 per turn, offline-computed into _aux.winprob.
  // One line per recorded seat (4p can field multiple RL seats), in that seat's color.
  const cv = document.getElementById(canvasId);
  const panel = document.getElementById("winprobPanel");
  const wp = _winprobDict();
  if (!cv || !replay || !wp) { if (panel) panel.style.display = "none"; return; }
  if (panel) panel.style.display = "";
  const c = cv.getContext("2d"), Wd = cv.width, Hd = cv.height;
  c.clearRect(0, 0, Wd, Hd);
  const seats = Object.keys(wp).sort((a, b) => (+a) - (+b));
  const N = seats.reduce((m, s) => Math.max(m, wp[s].length), 0);
  if (!N) return;
  const padL = 36, padR = 8, padT = 8, padB = 16, plotW = Wd - padL - padR, plotH = Hd - padT - padB;
  c.font = "10px monospace";
  for (let i = 0; i <= 4; i++) {
    const y = padT + plotH * i / 4, pct = 100 * (1 - i / 4);
    c.strokeStyle = (pct === 50) ? "#475569" : "#1e293b";
    c.beginPath(); c.moveTo(padL, y); c.lineTo(Wd - padR, y); c.stroke();
    c.fillStyle = "#64748b"; c.fillText(pct.toFixed(0) + "%", 2, y + 4);
  }
  const xstep = Math.max(1, Math.floor(N / 10));
  for (let t = 0; t < N; t += xstep) { const x = padL + plotW * t / Math.max(1, N - 1); c.fillStyle = "#64748b"; c.fillText(t.toString(), x - 6, Hd - 4); }
  for (const s of seats) {
    const arr = wp[s];
    c.strokeStyle = COLORS[s] || "#888"; c.lineWidth = 1.5;
    c.beginPath(); let started = false;
    for (let t = 0; t < N; t++) {
      const v = arr[t]; if (v == null) { started = false; continue; }
      const x = padL + plotW * t / Math.max(1, N - 1), y = padT + plotH * (1 - Math.max(0, Math.min(1, v)));
      if (!started) { c.moveTo(x, y); started = true; } else c.lineTo(x, y);
    }
    c.stroke();
  }
  if (curTurn < N) {
    const x = padL + plotW * curTurn / Math.max(1, N - 1);
    c.strokeStyle = "#22c55e"; c.lineWidth = 1.5;
    c.beginPath(); c.moveTo(x, padT); c.lineTo(x, Hd - padB); c.stroke();
    const txt = seats.map(s => (wp[s][curTurn] != null) ? `P${s}=${Math.round(100 * Math.max(0, Math.min(1, wp[s][curTurn])))}%` : null).filter(Boolean).join(" ");
    c.fillStyle = "#22c55e"; c.fillText(`t=${curTurn} ${txt}`, Math.min(x + 4, Wd - 150), padT + 10);
  }
}

function renderCharts() {
  _renderWinprob("winprobCanvas");          // RL win-confidence (independent of _series)
  if (!_series) return;
  // per-turn agent time + overtime (local replays only; canvases absent in official viewer)
  const tmCv = document.getElementById("timeCanvas");
  if (tmCv) {
    const panel = document.getElementById("timePanel");
    if (panel) panel.style.display = _series.time ? "" : "none";
    if (_series.time) {
      _renderMultiLine("timeCanvas", _series.time);
      const hdr = document.getElementById("timeHead");
      if (hdr && _series.timeStats) {
        hdr.textContent = _series.timeStats.map((s, p) =>
          s ? `P${p} avg=${s.avg.toFixed(0)} min=${s.min.toFixed(0)} max=${s.max.toFixed(0)}ms` : "")
          .filter(Boolean).join("   ·   ");
      }
    }
  }
  const obCv = document.getElementById("overageBankCanvas");
  if (obCv) {
    const panel = document.getElementById("overageBankPanel");
    if (panel) panel.style.display = _series.overageBank ? "" : "none";
    if (_series.overageBank) _renderMultiLine("overageBankCanvas", _series.overageBank);
  }
  _renderMultiLine("shipsCanvas", _series.ships);
  _renderMultiLine("prodCanvas", _series.prod);
  const ovCv = document.getElementById("overageCanvas");
  if (ovCv) {
    const hasOv = _series.overage.some(s => s.some(v => v != null));
    const panel = document.getElementById("overagePanel");
    if (panel) panel.style.display = hasOv ? "" : "none";
    if (hasOv) _renderMultiLine("overageCanvas", _series.overage);
  }
}

// Player username for slot `pid`: official replays carry the full TeamNames in
// meta.agent_names (all 4 for 4p); local replays only set agent_a/agent_b (2p).
function playerName(meta, pid) {
  if (meta.agent_names && meta.agent_names[pid]) return meta.agent_names[pid];
  if (pid === 0) return meta.agent_a;
  if (pid === 1) return meta.agent_b;
  return null;
}

// Right-side per-turn panel: player totals (incl. overage bank) + fleets in flight.
function renderSidePanel() {
  const pane = document.getElementById("sideList");
  if (!pane) return;
  if (!replay) { pane.textContent = "—"; return; }
  const st = replay.steps[curTurn], nA = replay.meta.n_agents || 2;
  const ships = new Array(nA).fill(0), prod = new Array(nA).fill(0), planets = new Array(nA).fill(0);
  for (const p of st.planets) { const o = p[1]; if (o >= 0 && o < nA) { ships[o] += p[5]; prod[o] += p[6]; planets[o]++; } }
  for (const f of st.fleets) { const o = f[1]; if (o >= 0 && o < nA) ships[o] += f[6]; }  // f[6]=ships (f[5]=from_planet_id)
  const meta = replay.meta;
  let html = `<div class="turnhdr">Turn ${st.t} / ${meta.n_turns}</div>`;
  for (let pid = 0; pid < nA; pid++) {
    const ov = (st.overage && st.overage[pid] != null) ? ` · overage ${(+st.overage[pid]).toFixed(1)}s` : "";
    const nm = playerName(meta, pid);
    html += `<div class="pstat" style="color:${COLORS[String(pid)] || COLORS["-1"]}">` +
      `<b>P${pid}</b>${nm ? ` ${nm}` : ""}: ${ships[pid]} ships · +${prod[pid]}/t · ${planets[pid]} planets${ov}</div>`;
  }
  const _wp = _winprobDict();
  if (_wp) {
    let wtxt = "";
    for (const s of Object.keys(_wp).sort((a, b) => (+a) - (+b))) if (_wp[s][curTurn] != null)
      wtxt += `<span style="color:${COLORS[s] || "#888"}"><b>P${s}</b> ${Math.round(100 * Math.max(0, Math.min(1, _wp[s][curTurn])))}%</span> `;
    if (wtxt) html += `<div class="pstat" style="color:#94a3b8">RL win-confidence (value head): ${wtxt}</div>`;
  }
  const fl = [...st.fleets].sort((a, b) => (a[1] - b[1]) || (a[0] - b[0]));
  html += `<div class="fhdr">Fleets in flight (${fl.length})</div>`;
  if (!fl.length) html += `<div style="color:#64748b">none</div>`;
  for (const ft of fl) {
    // env schema: [id, owner, x, y, angle, from_planet_id, ships]
    const [fid, owner, x, y, angle, src, sh] = ft;
    const color = COLORS[String(owner)] || "#888";
    const deg = (angle * 180 / Math.PI).toFixed(0);
    const launch = _fleetLaunch[fid];
    const age = launch != null ? (st.t - launch) : null;
    html += `<div class="fleet-item" style="border-color:${color}">` +
      `<span class="row1"><b style="color:${color}">P${owner}</b> f${fid} · ${sh} ships · from p${src}</span>` +
      `<span class="row2">heading ${deg}° · pos (${x.toFixed(1)},${y.toFixed(1)})` +
      `${age != null ? ` · in flight ${age}t` : ""}</span>` +
      `</div>`;
  }
  pane.innerHTML = html;
}

// ============================================================================
// Passive ship-count projection (no new launches) — TABLE.
//
// At the viewed turn t0, project EVERY real planet's ship count forward, using
// ONLY the fleets in flight at t0. Horizon = min(PROJ_HORIZON, 499 - t0): we
// cap at the game's true last turn (499), NOT at this replay's recorded end —
// a live agent doesn't know when the game stops, so the projection shouldn't
// either. That means we can't lean on the replay's future rows for the tail:
// planet positions are computed ANALYTICALLY (orbital motion, the same rc(t)=
// max(0,t-1) rule the agent cache uses), fleets fly straight, and current
// comets (short-lived, always within the recorded window) are read from the
// replay for the collision check. Fleets never interact with each other (only
// with planets), so each fleet's landing (which planet, which turn) is fixed
// and launch-independent; the only thing future launches change is ship counts
// (combat) — exactly what we compute. Each planet's curve is production +
// combat of its arrivals, resolved PER PLANET INDEPENDENTLY (no cross-planet
// coupling — that independence IS the "only recompute the hit planet" rule;
// recompute is <~5ms so we just redo it each frame). Mirrors forward_sim
// (step 4 production for owner>=0, step 7 combat) + shared/physics (collide/
// geom/motion). On-the-fly, no reprocessing — works on local & official alike.
// ============================================================================
const PROJ_HORIZON = 100;          // max turns to project forward
const LAST_TURN_ABS = 499;         // game's true final turn (episodeSteps 500)
const _ROTATION_LIMIT = 50;        // orbital_r + radius >= this => static planet
const _MAX_SPEED = 6.0, _LOG1000 = Math.log(1000.0);

function _rc(n) { return n > 1 ? n - 1 : 0; }   // env rotation count at obs.step n = max(0,n-1)

// Orbital params for a planet tuple [pid,owner,x,y,r,...] observed at t0.
function _orbital(p) {
  const dx = p[2] - SUN_X, dy = p[3] - SUN_Y, r = Math.hypot(dx, dy);
  if (r + p[4] >= _ROTATION_LIMIT) return { orbiting: false, x: p[2], y: p[3] };
  return { orbiting: true, base: Math.atan2(dy, dx), r };
}
// Analytic position of that planet k turns after t0 (mirrors motion.planet_pos_after).
function _planetPosAt(orb, av, t0, k) {
  if (!orb.orbiting) return [orb.x, orb.y];
  const ang = orb.base + av * (_rc(t0 + k) - _rc(t0));
  return [SUN_X + orb.r * Math.cos(ang), SUN_Y + orb.r * Math.sin(ang)];
}

function _fleetSpeed(ships) {                       // mirrors geom.fleet_speed
  let n = ships | 0;
  if (n <= 1) return 1.0;
  if (n > 1000) n = 1000;
  return 1.0 + (_MAX_SPEED - 1.0) * Math.pow(Math.log(n) / _LOG1000, 1.5);
}
function _pointSegDist2(cx, cy, x0, y0, x1, y1) {   // mirrors geom.point_seg_dist2
  const dx = x1 - x0, dy = y1 - y0, L2 = dx * dx + dy * dy;
  if (L2 < 1e-12) return (cx - x0) * (cx - x0) + (cy - y0) * (cy - y0);
  let t = ((cx - x0) * dx + (cy - y0) * dy) / L2;
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  const fx = x0 + t * dx, fy = y0 + t * dy;
  return (cx - fx) * (cx - fx) + (cy - fy) * (cy - fy);
}
function _sweptPairHit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r) {  // mirrors geom.swept_pair_hit
  const d0x = ax - p0x, d0y = ay - p0y;
  const dvx = (bx - ax) - (p1x - p0x), dvy = (by - ay) - (p1y - p0y);
  const a = dvx * dvx + dvy * dvy;
  const b = 2.0 * (d0x * dvx + d0y * dvy);
  const c = d0x * d0x + d0y * d0y - r * r;
  if (a < 1e-12) return c <= 0.0;
  const disc = b * b - 4.0 * a * c;
  if (disc < 0.0) return false;
  const sq = Math.sqrt(disc);
  const t1 = (-b - sq) / (2.0 * a), t2 = (-b + sq) / (2.0 * a);
  return t2 >= 0.0 && t1 <= 1.0;
}

// pid -> planet tuple [pid,owner,x,y,r,ships,prod] at a replay step index.
function _planetMap(idx) {
  const m = {}, st = replay.steps[idx];
  if (st) for (const p of st.planets) m[p[0]] = p;
  return m;
}

// Walk one in-flight fleet's straight line and report the FIRST event. Mirrors
// shared/physics first_hit_from (planet > oob > sun; first planet in env list
// order wins — real planets, present from turn 0, precede appended comets).
// Real planets are positioned ANALYTICALLY (so the walk runs to abs turn 499,
// past the replay's end); current comets are read from the replay (always
// within the recorded window during their short life). Returns {kind,pid?,turn?}.
//   realList : real planet tuples in env (obs) order at t0
//   realOrbs : Map pid -> orbital params (from _orbital)
function _fleetLanding(fx, fy, angle, ships, t0, lastIdx, av, realList, realOrbs) {
  const sp = _fleetSpeed(ships), vx = Math.cos(angle) * sp, vy = Math.sin(angle) * sp;
  const sunR2 = SUN_R * SUN_R;
  const maxK = Math.min(PROJ_HORIZON, LAST_TURN_ABS - t0);
  for (let k = 1; k <= maxK; k++) {
    const ax = fx + (k - 1) * vx, ay = fy + (k - 1) * vy;
    const bx = fx + k * vx, by = fy + k * vy;
    for (const p of realList) {                     // real planets (analytic), env order
      const orb = realOrbs.get(p[0]);
      const P0 = _planetPosAt(orb, av, t0, k - 1), P1 = _planetPosAt(orb, av, t0, k);
      if (_sweptPairHit(ax, ay, bx, by, P0[0], P0[1], P1[0], P1[1], p[4]))
        return { kind: "planet", pid: p[0], turn: t0 + k };
    }
    if (t0 + k <= lastIdx) {                         // comets: read from replay (their lifetime)
      const prev = _planetMap(t0 + k - 1), curStep = replay.steps[t0 + k];
      for (const p of curStep.planets) {
        if (realOrbs.has(p[0])) continue;            // skip reals (handled above)
        const pp = prev[p[0]];
        if (!pp) continue;
        if (_sweptPairHit(ax, ay, bx, by, pp[2], pp[3], p[2], p[3], p[4]))
          return { kind: "comet", pid: p[0], turn: t0 + k };   // fleet dies on the comet
      }
    }
    if (!(bx >= 0 && bx <= BOARD && by >= 0 && by <= BOARD)) return { kind: "oob", turn: t0 + k };
    if (_pointSegDist2(SUN_X, SUN_Y, ax, ay, bx, by) < sunR2) return { kind: "sun", turn: t0 + k };
  }
  return { kind: "none" };
}

// One planet's combat for a single turn (mirrors forward_sim._resolve_combat).
// incoming = [{owner, ships}, ...] arriving THIS turn. Returns [owner, ships].
function _resolveCombat(owner, ships, incoming) {
  if (!incoming.length) return [owner, ships];
  const byOwner = {};
  for (const f of incoming) byOwner[f.owner] = (byOwner[f.owner] || 0) + f.ships;
  const g = Object.keys(byOwner).map(o => [+o, byOwner[o]]).sort((a, b) => b[1] - a[1]);
  let so, ss;
  if (g.length === 1) { so = g[0][0]; ss = g[0][1]; }
  else {
    if (g[0][1] === g[1][1]) return [owner, ships];   // two-way tie -> all destroyed
    so = g[0][0]; ss = g[0][1] - g[1][1];             // top minus second (3rd+ destroyed)
  }
  if (so === owner) return [owner, ships + ss];        // reinforcement
  if (ss > ships) return [so, ss - ships];             // capture
  return [owner, ships - ss];                          // repelled
}

let _projCache = null;   // last computed projection (for hover/debug if needed)

const _COMET_SLOTS = 4;        // reserved comet-id rows (env keeps <=4 comets alive)

function _computeProjection() {
  if (!replay) return null;
  const t0 = curTurn, lastIdx = replay.steps.length - 1;
  const st = replay.steps[t0];
  const av = (replay.static && replay.static.angular_velocity) || 0;
  const initIds = new Set(((replay.static && replay.static.initial_planets) || []).map(ip => ip[0]));
  const maxInit = initIds.size ? Math.max(...initIds) : -1;
  const cometSlots = [];                             // reserved comet ids = maxInit+1..+4
  for (let i = 1; i <= _COMET_SLOTS; i++) cometSlots.push(maxInit + i);

  // Real planets present at t0 (env/obs order) + their analytic orbital params.
  const realList = st.planets.filter(p => initIds.has(p[0]));
  const realOrbs = new Map();
  for (const p of realList) realOrbs.set(p[0], _orbital(p));

  // Arrivals from CURRENT in-flight fleets only. fleet = [id,owner,x,y,angle,src,ships]
  const arrivals = {};                               // pid -> { absTurn -> [{owner,ships}] }
  for (const f of st.fleets) {
    const land = _fleetLanding(f[2], f[3], f[4], f[6], t0, lastIdx, av, realList, realOrbs);
    if (land.kind !== "planet" || !initIds.has(land.pid)) continue;  // died / hit a comet / left board
    const byTurn = (arrivals[land.pid] = arrivals[land.pid] || {});
    (byTurn[land.turn] = byTurn[land.turn] || []).push({ owner: f[1], ships: f[6] });
  }

  const H = Math.max(0, Math.min(PROJ_HORIZON, LAST_TURN_ABS - t0));

  // Real-planet rows: production (owner>=0) + combat of arrivals, per planet.
  const rows = [];
  const realSorted = [...realList].sort((a, b) => a[0] - b[0]);
  for (const p of realSorted) {
    const pid = p[0], prod = p[6];
    let owner = p[1], ships = p[5];
    const shipsArr = [ships], ownerArr = [owner], aliveArr = [true];
    const arrP = arrivals[pid] || {};
    for (let k = 1; k <= H; k++) {
      if (owner >= 0) ships += prod;                 // step 4: production (owned planets only)
      const inc = arrP[t0 + k];                      // step 7: combat with this turn's arrivals
      if (inc) { const r = _resolveCombat(owner, ships, inc); owner = r[0]; ships = r[1]; }
      shipsArr.push(ships); ownerArr.push(owner); aliveArr.push(true);
    }
    rows.push({ pid, isComet: false, shipsArr, ownerArr, aliveArr });
  }

  // Reserved comet rows: present/ships read from the replay (their lifetime is
  // always within the recorded window). Dim when this comet id is not alive.
  for (const cid of cometSlots) {
    const shipsArr = [], ownerArr = [], aliveArr = [];
    for (let k = 0; k <= H; k++) {
      const idx = t0 + k;
      let alive = false, sh = null, ow = -1;
      if (idx <= lastIdx) {
        const cp = _planetMap(idx)[cid];
        if (cp) { alive = true; sh = cp[5]; ow = cp[1]; }
      }
      aliveArr.push(alive); shipsArr.push(sh); ownerArr.push(ow);
    }
    rows.push({ pid: cid, isComet: true, shipsArr, ownerArr, aliveArr });
  }
  return { t0, H, rows, nFleets: st.fleets.length };
}

function _ensureProjStyle() {
  if (document.getElementById("projtblStyle")) return;
  const s = document.createElement("style");
  s.id = "projtblStyle";
  s.textContent =
    "#projTable{overflow-x:auto;max-height:340px;overflow-y:auto}" +
    "table.projtbl{border-collapse:collapse;font:11px monospace;color:#e2e8f0}" +
    "table.projtbl th,table.projtbl td{border:1px solid #1e293b;padding:1px 6px;text-align:right}" +
    "table.projtbl th{color:#64748b;font-weight:normal;background:#0b1220;position:sticky;top:0}" +
    "table.projtbl td.dim{color:#334155}" +
    "table.projtbl tr.cometrow{opacity:0.35}" +
    "table.projtbl td:first-child,table.projtbl th:first-child{text-align:left;position:sticky;left:0;background:#0f172a;z-index:1}";
  document.head.appendChild(s);
}

function renderProjTable() {
  const pane = document.getElementById("projTable");
  if (!pane) return;
  _ensureProjStyle();
  if (!replay) { pane.textContent = "—"; return; }
  // Prefer the RECORDED projection (exact forward_sim roll in _aux.projection); fall back to
  // the JS client recompute only for replays without it (e.g. older official files).
  const rec = replay.steps[curTurn] && replay.steps[curTurn].projection;
  if (rec) { _renderRecordedProj(pane, rec); return; }
  const proj = _computeProjection();
  _projCache = proj;
  if (!proj || !proj.rows.length) { pane.innerHTML = '<div style="color:#64748b">no planets</div>'; return; }
  let html = `<div style="color:#94a3b8;font-size:11px;margin-bottom:5px">Passive projection (no new launches) from turn ` +
    `<b style="color:#e2e8f0">${proj.t0}</b> over the next <b style="color:#e2e8f0">${proj.H}</b> turns ` +
    `(to abs turn ${proj.t0 + proj.H}) · ${proj.nFleets} fleet(s) in flight · number colored by projected owner.</div>`;
  const cols = [];                                   // show every 5th turn (+ final H)
  for (let k = 5; k <= proj.H; k += 5) cols.push(k);
  if (proj.H >= 1 && (cols.length === 0 || cols[cols.length - 1] !== proj.H)) cols.push(proj.H);
  html += '<table class="projtbl"><thead><tr><th>pid \\ Δt</th>';
  for (const k of cols) html += `<th>${k}</th>`;
  html += "</tr></thead><tbody>";
  for (const row of proj.rows) {
    const dimRow = row.isComet && !row.aliveArr[0];  // comet not alive at t0 -> whole row dim
    html += `<tr${dimRow ? ' class="cometrow"' : ""}><th>p${row.pid}</th>`;
    for (const k of cols) {
      const s = row.shipsArr[k];
      if (s == null || (row.isComet && !row.aliveArr[k])) { html += '<td class="dim">·</td>'; continue; }
      const col = COLORS[String(row.ownerArr[k])] || COLORS["-1"];
      html += `<td style="color:${col}">${Math.round(s)}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  pane.innerHTML = html;
}

// ============================================================================
// RL model-input panel (from a parallel *.rlinput.json sidecar; original replay
// json untouched). Shows, for the current turn, the per-planet model-input matrix
// the RL input-builder produced + that turn's pure-Python build time.
// ============================================================================
let rlSidecar = null;                       // set by the page after fetching the sidecar
function setRLSidecar(s) { rlSidecar = s; }

function _ensureRLStyle() {
  if (document.getElementById("rltblStyle")) return;
  const s = document.createElement("style");
  s.id = "rltblStyle";
  s.textContent =
    "#rlInput{overflow:auto;max-height:340px}" +
    "table.rltbl{border-collapse:collapse;font:10px monospace;color:#cbd5e1}" +
    "table.rltbl th,table.rltbl td{border:1px solid #1e293b;padding:1px 5px;text-align:right;white-space:nowrap}" +
    "table.rltbl th{color:#64748b;font-weight:normal;background:#0b1220;position:sticky;top:0}" +
    "table.rltbl td:first-child,table.rltbl th:first-child{position:sticky;left:0;background:#0f172a;text-align:left;color:#e2e8f0}" +
    "table.rltbl tr.cometrow td{color:#a78bfa}";
  document.head.appendChild(s);
}

function renderRLPanel() {
  const pane = document.getElementById("rlInput");
  if (!pane) return;                        // page has no RL panel (e.g. local viewer)
  _ensureRLStyle();
  const head = document.getElementById("rlHead");
  if (!rlSidecar) { pane.innerHTML = '<span style="color:#64748b">no RL sidecar for this replay</span>'; if (head) head.textContent = ""; return; }
  const m = rlSidecar.meta, s = rlSidecar.steps[curTurn];
  if (head) head.textContent =
    `player ${m.player} · P×Fp = ${(s ? s.P : "?")}×${m.Fp} · build this turn ${s ? s.build_ms : "?"} ms ` +
    `· steady mean ${m.build_ms.mean_ms} / median ${m.build_ms.median_ms} / p95 ${m.build_ms.p95_ms} / max ${m.build_ms.max_ms} ms`;
  if (!s) { pane.innerHTML = "—"; return; }
  const names = m.feature_names, R = s.real_ids.length;
  let html = '<table class="rltbl"><thead><tr><th>row</th>';
  for (const n of names) html += `<th>${n}</th>`;
  html += "</tr></thead><tbody>";
  for (let r = 0; r < s.planets.length; r++) {
    let label, comet = false;
    if (r < R) label = "p" + s.real_ids[r];
    else { const j = r - R; comet = true; label = (j < s.comet_ids.length) ? "p" + s.comet_ids[j] + "☄" : "c" + j + "·∅"; }
    html += `<tr${comet ? ' class="cometrow"' : ""}><td>${label}</td>`;
    for (const v of s.planets[r]) html += `<td>${v}</td>`;
    html += "</tr>";
  }
  html += "</tbody></table>";
  pane.innerHTML = html;
}

// ============================================================================
// Playback
// ============================================================================
function renderFrame() {
  if (!replay) return;
  clear(); drawSun(); drawPlanets(); drawFleets();
  renderSidePanel();
  renderCharts();
  renderProjTable();
  renderRLPanel();
  document.getElementById("turnLabel").textContent =
    `${curTurn}/${replay.steps.length - 1}`;
  document.getElementById("turnSlider").value = curTurn;
}

let playTimer = null;
function setPlay(on) {
  playing = on;
  document.getElementById("playPause").textContent = on ? "⏸ Pause" : "▶ Play";
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  if (on && replay) {
    if (curTurn >= replay.steps.length - 1) { curTurn = 0; renderFrame(); }
    const frameMs = Math.max(20, (1000 / 3) * speedMultiplier);
    playTimer = setInterval(() => {
      if (!playing || !replay) { clearInterval(playTimer); playTimer = null; return; }
      curTurn++;
      if (curTurn >= replay.steps.length) {
        curTurn = replay.steps.length - 1;
        clearInterval(playTimer); playTimer = null;
        playing = false;
        document.getElementById("playPause").textContent = "▶ Play";
      }
      renderFrame();
    }, frameMs);
  }
}

// Official-schema replay (steps[t] = [per-player {observation,...}]) + optional top-level
// `_aux` (our recorder's extras) -> canonical {meta, static, steps} the core renders.
// BOTH local replays (shared/sim/runner) and official daily replays are this schema, so one
// adapter serves both. _aux.time -> step.time (per-turn agent wall-clock); _aux.projection ->
// step.projection (recorded passive 100-turn forecast); _aux.meta -> winner/scores/team_names.
function adaptOfficial(od) {
  const aux = od._aux || {};
  const auxTime = aux.time || [];
  const auxProj = aux.projection || [];
  const steps = od.steps.map((arr, t) => {
    const obs = (arr && arr[0] && arr[0].observation) || {};
    const overage = arr.map(a => (a && a.observation && a.observation.remainingOverageTime != null)
      ? a.observation.remainingOverageTime : null);
    const step = { t: (obs.step != null ? obs.step : t), planets: obs.planets || [], fleets: obs.fleets || [], overage };
    if (auxTime[t]) step.time = auxTime[t];                 // {"0":sec,"1":sec} per-turn agent time (local recorder)
    if (auxProj && auxProj[t]) step.projection = auxProj[t]; // {pid:[signed ships...]} recorded forward-roll projection
    return step;
  });
  const init = (od.steps[0] && od.steps[0][0].observation.initial_planets) || [];
  const am = aux.meta || {};
  const rewards = od.rewards || [];
  let winner = (am.winner != null) ? am.winner : -1;
  if (am.winner == null && rewards.length >= 2) {
    const mx = Math.max(...rewards);
    const top = rewards.map((r, i) => (r === mx ? i : -1)).filter(i => i >= 0);
    winner = (top.length === 1) ? top[0] : -1;
  }
  const names = am.team_names || (od.info && od.info.TeamNames) || [];
  const nAgents = (od.steps[0] && od.steps[0].length) || rewards.length || 2;
  return {
    meta: {
      winner, scores: am.scores || rewards, n_turns: steps.length - 1, n_agents: nAgents,
      agent_a: names[0] || "P0", agent_b: names[1] || "P1", agent_names: names,
      seed: (am.seed != null ? am.seed : (od.info && od.info.seed)),
    },
    static: { initial_planets: init, angular_velocity: (od.steps[0] && od.steps[0][0].observation.angular_velocity) || 0 },
    steps,
  };
}

// Render a RECORDED passive projection (step.projection = {pid: [signed ships...]}, p0 view:
// + = p0's expected ships, - = opponent, 0 = neutral). Exact (forward_sim roll), no recompute.
function _renderRecordedProj(pane, proj) {
  const pids = Object.keys(proj).sort((a, b) => (+a) - (+b));
  const H = pids.length ? proj[pids[0]].length : 0;
  const cols = [];
  for (let k = 5; k <= H; k += 5) cols.push(k);
  if (H >= 1 && (cols.length === 0 || cols[cols.length - 1] !== H)) cols.push(H);
  let html = `<div style="color:#94a3b8;font-size:11px;margin-bottom:5px">Passive projection (recorded, no new launches) from turn ` +
    `<b style="color:#e2e8f0">${replay.steps[curTurn].t}</b> over the next <b style="color:#e2e8f0">${H}</b> turns · ` +
    `signed p0 view (<span style="color:${COLORS["0"]}">+P0</span> / <span style="color:${COLORS["1"]}">−P1</span>).</div>`;
  html += '<table class="projtbl"><thead><tr><th>pid \\ Δt</th>';
  for (const k of cols) html += `<th>${k}</th>`;
  html += "</tr></thead><tbody>";
  for (const pid of pids) {
    const arr = proj[pid];
    html += `<tr><th>p${pid}</th>`;
    for (const k of cols) {
      const v = arr[k - 1];                              // arr[0] = Δt=1
      if (v == null) { html += '<td class="dim">·</td>'; continue; }
      const col = v > 0 ? COLORS["0"] : (v < 0 ? COLORS["1"] : COLORS["-1"]);
      html += `<td style="color:${col}">${Math.abs(Math.round(v))}</td>`;
    }
    html += "</tr>";
  }
  pane.innerHTML = html + "</tbody></table>";
}

// Called by the page after fetching/adapting a replay into the canonical shape.
function loadReplayObject(r) {
  replay = r;
  curTurn = 0;
  if (replay.meta.n_agents == null) replay.meta.n_agents = _inferAgents();
  _computeFleetLaunch();
  _computeSeries();
  const slider = document.getElementById("turnSlider");
  slider.max = replay.steps.length - 1;
  slider.value = 0;
  setPlay(false);
  renderFrame();
}

// Wire the playback controls (call once on load). Selection wiring is per-page.
function initCoreControls() {
  document.getElementById("turnSlider").addEventListener("input", (e) => {
    curTurn = parseInt(e.target.value); renderFrame();
  });
  document.getElementById("prevBtn").addEventListener("click", () => {
    curTurn = Math.max(0, curTurn - 1); renderFrame();
  });
  document.getElementById("nextBtn").addEventListener("click", () => {
    if (replay) curTurn = Math.min(replay.steps.length - 1, curTurn + 1);
    renderFrame();
  });
  document.getElementById("playPause").addEventListener("click", () => setPlay(!playing));
  document.getElementById("speed").addEventListener("change", (e) => {
    speedMultiplier = parseFloat(e.target.value);
    if (playing) { setPlay(false); setPlay(true); }
  });
  document.addEventListener("keydown", (e) => {
    if (!replay) return;
    if (e.key === "ArrowLeft") { curTurn = Math.max(0, curTurn - 1); renderFrame(); }
    else if (e.key === "ArrowRight") { curTurn = Math.min(replay.steps.length - 1, curTurn + 1); renderFrame(); }
    else if (e.key === " ") { e.preventDefault(); setPlay(!playing); }
  });
}

async function fetchJSON(url) {
  const resp = await fetch(url, { cache: "no-cache" });
  if (!resp.ok) throw new Error(`Fetch failed ${resp.status}: ${url}`);
  return await resp.json();
}
async function fetchText(url) {
  const resp = await fetch(url, { cache: "no-cache" });
  if (!resp.ok) throw new Error(`Fetch failed ${resp.status}: ${url}`);
  return await resp.text();
}
