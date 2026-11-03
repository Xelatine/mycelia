/* Mycelia cockpit — vanilla JS, no CDN: canvas graph, live curves, chat traces. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const COLORS = {
    green: "#38e08a", greenDim: "#1f7a4c", orange: "#ff9d3d", grey: "#7a8a99",
    red: "#ff5d5d", blue: "#4bb8ff", violet: "#b58bff", edge: "#20374a",
    active: "#4d7f9e", special: "#b58bff", text: "#cfe3f5",
  };
  const STATE_COLOR = {
    mature: COLORS.green, new: COLORS.orange, active: COLORS.active,
    isolated: COLORS.grey, dead: "#33414d", special: COLORS.special,
  };
  const NODE_R = { mature: 5.5, new: 4.5, active: 3.6, isolated: 3.2, dead: 2.5, special: 7 };

  const S = {
    state: null, trainer: null, graph: null, events: [], curve: [],
    selected: null, card: null, impulses: [], hover: null,
    view: { k: 1, tx: 0, ty: 0 }, dragging: null, fit: false, pos: new Map(),
  };

  // ------------------------------------------------------------------ utils
  async function api(path, body) {
    const opts = body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    };
    const res = await fetch(path, opts);
    const text = await res.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { detail: text }; }
    if (!res.ok) throw new Error(data.detail || res.statusText || "requête échouée");
    return data;
  }

  let toastTimer = null;
  function toast(message, isError) {
    const el = $("toast");
    el.textContent = message;
    el.classList.remove("hidden");
    el.style.borderColor = isError ? COLORS.red : COLORS.edge;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), isError ? 6500 : 3200);
  }

  const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v)) ? "—"
    : (typeof v === "number" ? v.toLocaleString("fr-FR", { maximumFractionDigits: d ?? 2 }) : v);

  function sizes(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.parentElement.clientWidth;
    const h = canvas.clientHeight || parseInt(canvas.getAttribute("height") || "80", 10);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  // ----------------------------------------------------------------- têtes
  const TRAINER_LABEL = {
    running: "en marche", paused: "en pause", finished: "terminé",
    idle: "au repos", error: "erreur",
  };

  function renderHeader() {
    const st = S.state || {};
    $("pill-step").textContent = st.step
      ? `step ${fmt(st.step)} / ${fmt(st.steps_total)}` : "step —";
    $("pill-phase").textContent = `phase ${st.phase || "—"}`;
    $("pill-device").textContent = (st.config && st.config.device) || "—";
    const state = (S.trainer && S.trainer.state) || st.trainer_state || "idle";
    const pill = $("pill-state");
    pill.textContent = TRAINER_LABEL[state] || state;
    pill.className = "pill state " + state;
  }

  function renderTrainer() {
    const info = S.trainer;
    if (!info) return;
    const enabled = info.control_enabled !== false;
    $("ctl-start").disabled = !enabled || !$("ctl-config").value || !info.can_start;
    $("ctl-pause").disabled = !enabled || !info.can_pause;
    $("ctl-resume").disabled = !enabled || !$("ctl-config").value || !info.can_resume;
    $("ctl-stop").disabled = !enabled || !info.can_stop;
    $("ctl-save").disabled = !enabled || !info.can_pause;
    $("ctl-hint").textContent = !enabled ? "pilotage désactivé (--no-control)"
      : info.state === "error" && info.error ? info.error
      : info.state === "paused" ? "checkpoint sauvegardé — « Reprendre » continue ici"
      : info.state === "running" ? `pid ${info.pid} · ${(info.log || "").split(/[\\/]/).pop()}`
      : "les boutons pilotent un trainer détaché : fermer la page ne l'arrête pas";
    const el = $("ctl-state");
    el.textContent = TRAINER_LABEL[info.state] || info.state || "—";
    el.className = "ctl-state " + (info.state || "");
    el.title = (info.last_lines || []).slice(-6).join("\n");
  }

  function renderKpis() {
    const st = S.state || {};
    const n = st.nodes || {};
    const counters = st.counters || {};
    const items = [
      ["nœuds", fmt(n.alive ?? n.total), `${fmt(n.total)} au total`],
      ["arêtes", fmt(st.edges)],
      ["/nœud", fmt(st.edges_per_node, 2)],
      ["naissances", fmt(counters.births ?? 0)],
      ["liens coupés", fmt(counters.link_cuts ?? 0)],
      ["morts", fmt(n.deaths ?? counters.deaths ?? 0)],
      ["cimetière", fmt(st.cemetery ?? 0)],
      ["isolés", fmt(n.isolated ?? 0)],
      ["perplexité", fmt(st.best_val ? Math.exp(st.best_val) : null)],
    ];
    $("kpis").innerHTML = items.map(([l, v]) =>
      `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  }

  function renderDiagnose() {
    const st = S.state || {};
    const n = st.nodes || {};
    const counters = st.counters || {};
    const live = st.live || {};
    const lines = [];
    const epn = st.edges_per_node || 0;
    if (!st.step) lines.push("en attente du premier pas d'entraînement.");
    if (st.step) {
      lines.push(epn < 1.2
        ? `<span class="warn">câblage très faible</span> (${fmt(epn, 2)} arête/nœud) — baisse la pression L1 ou laisse tourner.`
        : `câblage <b>${fmt(epn, 2)}</b> arête/nœud.`);
      if (!(counters.births > 0)) {
        lines.push("aucune naissance encore : le score de croissance n'a pas dépassé le seuil.");
      }
      if (counter_rose(counters)) lines.push("les liens naissent plus qu'ils ne meurent actuellement.");
      lines.push(`énergie résiduelle moyenne <b>${fmt(live.energy, 4)}</b> · `
        + `${fmt(live.hops, 1)} signaux vivants/token · ${fmt(live.visited)} nœuds visités.`);
      lines.push(`durée de pas <b>${fmt((live.step_time || 0) * 1000, 0)} ms</b> · `
        + `FLOPs/token ≈ ${fmt(live.flops_per_token, 0)}.`);
      if (live.aux) {
        lines.push(`circulation ${fmt(live.aux.circ, 3)} · entropie routage ${fmt(live.aux.entropy, 3)} `
          + `· redondance ${fmt(live.aux.redundancy, 4)}.`);
      }
    }
    $("diagnose").innerHTML = lines.join("<br />");
  }

  function counter_rose(counters) {
    return (counters.edges_born || 0) > (counters.link_cuts || 0) * 0.2;
  }

  // ---------------------------------------------------------------- courbes
  function drawSeries(canvas, series, opts) {
    const { ctx, w, h } = sizes(canvas);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#070d13";
    ctx.fillRect(0, 0, w, h);
    let values = [];
    series.forEach((s) => { values = values.concat(s.points.filter((v) => v !== null)); });
    if (!values.length) {
      ctx.fillStyle = "#2c4254";
      ctx.font = "11px Segoe UI";
      ctx.fillText(opts.empty || "pas encore de données", 10, h / 2);
      return;
    }
    let min = Math.min.apply(null, values), max = Math.max.apply(null, values);
    if (max - min < 1e-9) { max = min + 1; }
    const pad = 4;
    const X = (i, n) => pad + (w - 2 * pad) * (n <= 1 ? 0.5 : i / (n - 1));
    const Y = (v) => h - pad - (h - 2 * pad) * ((v - min) / (max - min));
    ctx.strokeStyle = "#0f1c26";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 3; g++) {
      const y = pad + (h - 2 * pad) * g / 3;
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
    }
    series.forEach((s) => {
      const pts = s.points;
      ctx.beginPath();
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width || 1.6;
      let started = false;
      pts.forEach((v, i) => {
        if (v === null || v === undefined) return;
        const x = X(i, pts.length), y = Y(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
      });
      ctx.stroke();
      if (s.fill) {
        ctx.lineTo(X(pts.length - 1, pts.length), h - pad);
        ctx.lineTo(X(0, pts.length), h - pad);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, s.fill);
        grad.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = grad;
        ctx.fill();
      }
    });
    ctx.fillStyle = "#44607a";
    ctx.font = "10px Segoe UI";
    ctx.fillText(fmt(max, 1), 6, 11);
    ctx.fillText(fmt(min, 1), 6, h - 4);
  }

  function renderCharts() {
    const curve = (S.state && S.state.curve) || [];
    const pplPts = curve.map((p) => (p.val_ppl !== undefined ? p.val_ppl : null));
    const lossPts = curve.map((p) => (p.ppl !== undefined ? p.ppl : null));
    drawSeries($("chart-ppl"), [
      { points: lossPts, color: "rgba(75,184,255,0.55)", width: 1.2 },
      { points: pplPts, color: COLORS.green, width: 1.9, fill: "rgba(56,224,138,0.18)" },
    ], { empty: "perplexité dès la première validation" });
    drawSeries($("chart-edges"), [
      { points: curve.map((p) => (p.edges !== undefined ? p.edges : null)),
        color: COLORS.violet, width: 1.8, fill: "rgba(181,139,255,0.14)" },
    ], { empty: "arêtes vivantes au fil des steps" });
    drawSeries($("chart-nodes"), [
      { points: curve.map((p) => (p.nodes !== undefined ? p.nodes : null)),
        color: COLORS.orange, width: 1.8, fill: "rgba(255,157,61,0.12)" },
    ], { empty: "population" });
  }

  function drawHeatmap(canvas, matrix, palette) {
    const { ctx, w, h } = sizes(canvas);
    ctx.clearRect(0, 0, w, h);
    if (!matrix || !matrix.length) {
      ctx.fillStyle = "#243746"; ctx.font = "11px Segoe UI";
      ctx.fillText("pas encore de données", 10, h / 2);
      return;
    }
    const n = matrix.length;
    const cell = Math.min(w / n, h / n);
    const offX = (w - cell * n) / 2, offY = (h - cell * n) / 2;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = Math.max(0, Math.min(1, matrix[i][j]));
        ctx.fillStyle = palette(v);
        ctx.fillRect(offX + j * cell, offY + i * cell, cell + 0.5, cell + 0.5);
      }
    }
  }

  function renderHeatmaps() {
    const g = S.graph;
    drawHeatmap($("heat"), g && g.heat,
      (v) => `rgba(56,224,138,${0.10 + 0.9 * v})`);
    drawHeatmap($("sim"), g && g.sim,
      (v) => `rgba(75,184,255,${0.06 + 0.55 * Math.max(0, v)})`);
  }

  // ----------------------------------------------------------------- graphe
  function fitView() {
    const canvas = $("graph");
    const w = canvas.clientWidth, h = canvas.clientHeight;
    S.view = { k: Math.min(w, h) * 0.46, tx: w / 2, ty: h / 2 };
    S.fit = true;
  }

  function toScreen(x, y) {
    return [S.view.tx + x * S.view.k, S.view.ty + y * S.view.k];
  }

  function renderGraph(now) {
    const canvas = $("graph");
    const { ctx, w, h } = sizes(canvas);
    if (!S.fit) fitView();
    ctx.clearRect(0, 0, w, h);
    const bg = ctx.createRadialGradient(w / 2, h / 2, 20, w / 2, h / 2, Math.max(w, h) * 0.7);
    bg.addColorStop(0, "#0a141d"); bg.addColorStop(1, "#05090d");
    ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);

    const g = S.graph;
    if (!g || !g.nodes || !g.nodes.length) {
      ctx.fillStyle = "#2c4254"; ctx.font = "12px Segoe UI";
      ctx.fillText("en attente du premier checkpoint…", 18, h / 2);
      return;
    }
    S.pos = new Map();
    g.nodes.forEach((n) => S.pos.set(n.id, n));

    // edges
    g.edges.forEach((e) => {
      const a = g.nodes[e.s], b = g.nodes[e.t];
      if (!a || !b) return;
      const [x1, y1] = toScreen(a.x, a.y), [x2, y2] = toScreen(b.x, b.y);
      ctx.beginPath();
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
      ctx.strokeStyle = e.strong ? `rgba(56,224,138,${0.10 + 0.35 * e.w})`
        : `rgba(56,120,160,${0.06 + 0.20 * e.w})`;
      ctx.lineWidth = e.strong ? 1.0 + e.w : 0.7;
      ctx.stroke();
    });

    // nodes
    const blink = 0.5 + 0.5 * Math.sin(now / 260);
    g.nodes.forEach((n) => {
      const [x, y] = toScreen(n.x, n.y);
      const r = (NODE_R[n.state] || 3) * Math.max(0.7, Math.min(1.6, S.view.k / 220));
      const selected = S.selected === n.id;
      const color = STATE_COLOR[n.state] || COLORS.active;
      ctx.beginPath();
      ctx.arc(x, y, r * (selected ? 1.8 : 1), 0, Math.PI * 2);
      if (n.state === "isolated") {
        ctx.fillStyle = `rgba(122,138,153,${0.25 + 0.55 * blink})`;
      } else {
        ctx.fillStyle = color;
      }
      ctx.fill();
      if (n.state === "mature" || n.state === "special") {
        ctx.beginPath(); ctx.arc(x, y, r * 2.1, 0, Math.PI * 2);
        ctx.fillStyle = color.replace(")", ",0.10)").replace("rgb", "rgba");
        ctx.globalAlpha = 0.16; ctx.fill(); ctx.globalAlpha = 1;
      }
      if (selected) {
        ctx.strokeStyle = COLORS.blue; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.arc(x, y, r * 2.6, 0, Math.PI * 2); ctx.stroke();
      }
      if ($("opt-labels").checked && n.degree >= 1) {
        ctx.fillStyle = "#6f8ba1"; ctx.font = "9px Segoe UI";
        ctx.fillText(String(n.id), x + r + 2, y - r);
      }
    });

    // impulses
    if ($("opt-impulses").checked) {
      S.impulses = S.impulses.filter((imp) => now - imp.t0 < imp.dur);
      S.impulses.forEach((imp) => {
        const a = S.pos.get(imp.a), b = S.pos.get(imp.b);
        if (!a || !b) return;
        const p = Math.min(1, (now - imp.t0) / imp.dur);
        const ease = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
        const [x1, y1] = toScreen(a.x, a.y), [x2, y2] = toScreen(b.x, b.y);
        const x = x1 + (x2 - x1) * ease, y = y1 + (y2 - y1) * ease;
        const alpha = 1 - Math.pow(p, 3);
        const glow = ctx.createRadialGradient(x, y, 0, x, y, 13);
        glow.addColorStop(0, `rgba(120,255,200,${0.85 * alpha})`);
        glow.addColorStop(0.35, `rgba(56,224,138,${0.45 * alpha})`);
        glow.addColorStop(1, "rgba(56,224,138,0)");
        ctx.fillStyle = glow;
        ctx.beginPath(); ctx.arc(x, y, 13, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(x, y, 2.4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(210,255,235,${alpha})`; ctx.fill();
      });
    }

    $("graph-info").textContent =
      `${fmt(g.shown_nodes)} / ${fmt(g.total_nodes)} nœuds · ${fmt(g.edges.length)} arêtes affichées `
      + `sur ${fmt(g.total_edges)} · step ${fmt(g.step)}`;
    if (S.hover) {
      const n = S.hover;
      $("graph-hover").textContent =
        `#${n.id} ${n.state} · deg ${n.degree} · usage ${fmt(n.usage, 3)} · conf ${fmt(n.conf, 2)}`;
    }
  }

  function fireImpulse(a, b, delay, dur) {
    S.impulses.push({ a, b, t0: performance.now() + (delay || 0), dur: dur || 520 });
    if (S.impulses.length > 900) S.impulses.splice(0, S.impulses.length - 900);
  }

  function playTrace(tokens) {
    if (!S.graph) return;
    const visible = new Set((S.graph.nodes || []).map((n) => n.id));
    let delay = 0;
    tokens.forEach((tok) => {
      (tok.edges || []).slice(0, 36).forEach((pair, i) => {
        if (!visible.has(pair[0]) || !visible.has(pair[1])) return;
        fireImpulse(pair[0], pair[1], delay + i * 4, 460);
      });
      (tok.nodes || []).slice(0, 40).forEach((id, i) => {
        const n = S.pos.get(id);
        if (n) fireImpulse(id, id, delay + i * 4, 380);
      });
      delay += 210;
    });
  }

  // ----------------------------------------------------------------- journal
  function renderJournal() {
    const rows = S.events.slice(-140).reverse();
    $("journal").innerHTML = rows.map((e) => {
      const msg = e.message || `${e.type} ${e.node !== undefined ? "#" + e.node : ""}`;
      return `<div class="ev ${e.type}"><div class="t">${fmt(e.step ?? e.t, 0)}</div>`
        + `<div class="m">${escapeHtml(msg)}</div></div>`;
    }).join("") || '<div class="hint">aucun événement pour l\'instant</div>';
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // --------------------------------------------------------------- node card
  async function loadNode(id) {
    S.selected = id;
    try {
      S.card = await api(`/api/node/${id}`);
      renderNodeCard();
    } catch (err) {
      $("node-card").innerHTML = `<div class="hint">${escapeHtml(err.message)}</div>`;
    }
  }

  function renderNodeCard() {
    const c = S.card;
    if (!c) return;
    const kv = [
      ["état", c.state], ["né au step", fmt(c.born)], ["âge", fmt(c.age) + " pas"],
      ["degrés entrée/sortie", `${fmt(c.in_degree)} / ${fmt(c.out_degree)}`],
      ["usage", fmt(c.usage, 4)], ["confiance", fmt(c.confidence, 3)],
      ["croissance", fmt(c.growth, 3)], ["dissipation", fmt(c.dissipation, 3)],
      ["biais de porte", fmt(c.gate_bias, 3)], ["‖W‖", fmt(c.weight_norm, 2)],
      ["cimetière", c.cemetery ? `oui (step ${c.cemetery_step})` : "non"],
    ];
    const links = (list, key) => list.map((e) =>
      `<div><code>${key === "to" ? "→" : "←"} ${e[key]}</code> w ${fmt(e.w, 2)} · u ${fmt(e.usage, 3)}</div>`).join("");
    $("node-card").innerHTML = `
      <div class="title">nœud #${c.id}</div>
      <div class="kv">${kv.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("")}</div>
      <div class="links"><b>sorties</b>${links(c.outgoing, "to") || "<div>aucune</div>"}
      <b>entrées</b>${links(c.incoming, "from") || "<div>aucune</div>"}
      ${c.events && c.events.length ? `<b>historique</b>${c.events.map((e) => `<div>${fmt(e.step, 0)} · ${escapeHtml(e.type)}</div>`).join("")}` : ""}
      </div>`;
  }

  // ------------------------------------------------------------------ chat
  async function sendChat() {
    const prompt = $("chat-prompt").value.trim() || "Il était une fois";
    $("chat-send").disabled = true;
    $("chat-meta").textContent = "génération…";
    try {
      const out = await api("/api/generate", {
        prompt, tokens: Number($("chat-tokens").value) || 32,
        temperature: Number($("out-temp").textContent) || 0.85,
        top_p: Number($("chat-topp").value) || 0.9,
      });
      const body = out.text || "";
      $("chat-out").textContent = "";
      const chars = body.split("");
      let i = 0;
      const timer = setInterval(() => {
        $("chat-out").textContent += chars.slice(i, i + 3).join("");
        i += 3;
        if (i >= chars.length) clearInterval(timer);
      }, 16);
      const ntok = (out.tokens || []).length;
      $("chat-meta").textContent =
        `${fmt(ntok)} tokens · ${fmt(out.nodes_per_token, 1)} nœuds/token · step ${fmt(out.step)}`
        + (out.guard_fired ? " · garde-fou anti-répétition déclenché" : "");
      if ($("chat-live").checked) playTrace(out.tokens || []);
    } catch (err) {
      $("chat-meta").textContent = err.message;
      toast(err.message, true);
    } finally {
      $("chat-send").disabled = false;
    }
  }

  // ------------------------------------------------------------- interaction
  function bindGraph() {
    const canvas = $("graph");
    canvas.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      const factor = ev.deltaY < 0 ? 1.12 : 0.89;
      const k = Math.max(40, Math.min(4000, S.view.k * factor));
      const scale = k / S.view.k;
      S.view.tx = mx - (mx - S.view.tx) * scale;
      S.view.ty = my - (my - S.view.ty) * scale;
      S.view.k = k;
    }, { passive: false });
    canvas.addEventListener("mousedown", (ev) => {
      const rect = canvas.getBoundingClientRect();
      S.dragging = { x: ev.clientX - rect.left, y: ev.clientY - rect.top,
                     tx: S.view.tx, ty: S.view.ty, moved: false };
    });
    window.addEventListener("mousemove", (ev) => {
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      if (S.dragging) {
        S.view.tx = S.dragging.tx + (mx - S.dragging.x);
        S.view.ty = S.dragging.ty + (my - S.dragging.y);
        if (Math.abs(mx - S.dragging.x) + Math.abs(my - S.dragging.y) > 3) S.dragging.moved = true;
        return;
      }
      if (mx < 0 || my < 0 || mx > rect.width || my > rect.height) { S.hover = null; return; }
      let best = null, bestD = 12;
      (S.graph && S.graph.nodes || []).forEach((n) => {
        const [x, y] = toScreen(n.x, n.y);
        const d = Math.hypot(x - mx, y - my);
        if (d < bestD) { bestD = d; best = n; }
      });
      S.hover = best;
    });
    canvas.addEventListener("mouseup", () => {
      const drag = S.dragging;
      S.dragging = null;
      if (drag && !drag.moved && S.hover) loadNode(S.hover.id);
    });
    $("g-reset").addEventListener("click", () => { S.fit = false; });
  }

  // ---------------------------------------------------------------- trainer
  function controlPayload(fresh) {
    const seed = $("ctl-seed").value.trim();
    const steps = $("ctl-steps").value.trim();
    return {
      config: $("ctl-config").value,
      seed: seed ? Number(seed) : null,
      steps: steps ? Number(steps) : null,
      fresh: !!fresh,
    };
  }

  async function startRun(fresh) {
    try {
      S.trainer = await api("/api/train/start", controlPayload(fresh));
      renderTrainer();
      toast(`entraînement lancé (pid ${S.trainer.pid})`);
      setTimeout(refreshTrainer, 900);
    } catch (err) {
      if (!fresh && /checkpoint already exists/i.test(err.message || "")
          && confirm("Un checkpoint existe déjà.\n\nNouveau run et écraser ?")) {
        return startRun(true);
      }
      toast(err.message || "impossible de lancer", true);
      refreshTrainer();
    }
  }

  async function pauseRun() {
    try { S.trainer = await api("/api/train/pause", {}); renderTrainer();
      toast("pause demandée — sauvegarde en fin de pas"); }
    catch (err) { toast(err.message, true); }
  }

  async function resumeRun() {
    try {
      S.trainer = await api("/api/train/resume", controlPayload(false));
      renderTrainer();
      toast(`reprise depuis le checkpoint (pid ${S.trainer.pid})`);
      setTimeout(refreshTrainer, 900);
    } catch (err) { toast(err.message || "reprise impossible", true); refreshTrainer(); }
  }

  async function stopRun() {
    try { S.trainer = await api("/api/train/stop", {}); renderTrainer();
      toast("arrêt propre demandé — checkpoint puis sortie"); }
    catch (err) { toast(err.message, true); }
  }

  async function saveNow() {
    try { S.trainer = await api("/api/train/save", {}); renderTrainer();
      toast("sauvegarde demandée"); }
    catch (err) { toast(err.message, true); }
  }

  async function refreshTrainer() {
    try { S.trainer = await api("/api/train"); renderTrainer(); }
    catch (err) { /* serveur qui redémarre */ }
  }

  let pushTimer = null;
  function pushParams() {
    clearTimeout(pushTimer);
    pushTimer = setTimeout(() => {
      const payload = {
        lr: Number($("sl-lr").value) * 3.0e-5,
        l1_scale: Number($("sl-l1").value) / 10,
        tau_scale: Number($("sl-tau").value) / 10,
        temperature: Number($("sl-temp").value) / 100,
      };
      api("/api/params", payload).catch((err) => toast(err.message, true));
    }, 160);
  }

  function bindControls() {
    $("ctl-start").addEventListener("click", () => startRun(false));
    $("ctl-pause").addEventListener("click", pauseRun);
    $("ctl-resume").addEventListener("click", resumeRun);
    $("ctl-stop").addEventListener("click", stopRun);
    $("ctl-save").addEventListener("click", saveNow);
    $("chat-send").addEventListener("click", sendChat);
    $("chat-prompt").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });
    ["sl-lr", "sl-l1", "sl-tau", "sl-temp"].forEach((id) => {
      $(id).addEventListener("input", () => {
        $("out-lr").textContent = (Number($("sl-lr").value) * 3e-5).toExponential(1);
        $("out-l1").textContent = "×" + (Number($("sl-l1").value) / 10).toFixed(1);
        $("out-tau").textContent = "×" + (Number($("sl-tau").value) / 10).toFixed(1);
        $("out-temp").textContent = (Number($("sl-temp").value) / 100).toFixed(2);
        pushParams();
      });
    });
  }

  async function loadConfigs() {
    try {
      const configs = await api("/api/configs");
      const select = $("ctl-config");
      select.innerHTML = "";
      configs.forEach((item) => {
        const opt = document.createElement("option");
        opt.value = item.name; opt.textContent = item.label;
        opt.title = item.description || "";
        select.appendChild(opt);
      });
      const demo = configs.find((c) => c.label === "demo") || configs[0];
      if (demo) select.value = demo.name;
    } catch (err) { /* pas prêt */ }
    refreshTrainer();
  }

  // -------------------------------------------------------------- websocket
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.state) S.state = msg.state;
      if (msg.trainer) { S.trainer = msg.trainer; renderTrainer(); }
      if (msg.graph) S.graph = msg.graph;
      if (msg.events && msg.events.length) {
        S.events = S.events.concat(msg.events).slice(-500);
        renderJournal();
      }
      renderHeader(); renderKpis(); renderCharts(); renderHeatmaps(); renderDiagnose();
      if (S.selected && !S.card) loadNode(S.selected);
    };
    ws.onclose = () => setTimeout(connect, 1500);
    ws.onerror = () => ws.close();
  }

  function loop(now) {
    renderGraph(now);
    requestAnimationFrame(loop);
  }

  function boot() {
    bindControls();
    bindGraph();
    loadConfigs();
    connect();
    requestAnimationFrame(loop);
    setInterval(refreshTrainer, 4000);
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
