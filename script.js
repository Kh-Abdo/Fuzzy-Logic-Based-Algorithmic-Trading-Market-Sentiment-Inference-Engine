'use strict';

/* ==========================================================================
   1. FUZZY ENGINE
   Pure functions, no DOM access. Mamdani inference:
   AND = MIN, aggregation = MAX, centroid defuzzification at 0.5 resolution.
   Version 2 parameters: input sets form strict partitions (degrees sum to 1),
   RSI sets cross at the textbook 30 / 70, and the strong output sets reach about
   +/-80. Rule base, plain-sum centroid and classification are unchanged from the
   original audited script.js.
   ========================================================================== */
const Engine = (() => {
  const SIGNAL_MIN = -100;
  const SIGNAL_MAX = 100;
  const STEP = 0.5;

  // Membership functions as trapezoids [a, b, c, d]. A triangle repeats its peak: [a, b, b, c].
  const RSI_SETS = [
    { id: 'oversold',   label: 'Oversold',   pts: [0, 0, 10, 50] },
    { id: 'neutral',    label: 'Neutral',    pts: [10, 50, 50, 90] },
    { id: 'overbought', label: 'Overbought', pts: [50, 90, 100, 100] }
  ];

  const SENTIMENT_SETS = [
    { id: 'bearish', label: 'Bearish', pts: [-100, -100, -50, -10] },
    { id: 'neutral', label: 'Neutral', pts: [-50, -10, 10, 50] },
    { id: 'bullish', label: 'Bullish', pts: [10, 50, 100, 100] }
  ];

  const OUTPUT_SETS = [
    { id: 'strongSell', label: 'Strong Sell', pts: [-100, -100, -80, -45], color: '#ff3b6b' },
    { id: 'sell',       label: 'Sell',        pts: [-55, -30, -30, -5],   color: '#ff9a3d' },
    { id: 'hold',       label: 'Hold',        pts: [-20, 0, 0, 20],        color: '#ffd93d' },
    { id: 'buy',        label: 'Buy',         pts: [5, 30, 30, 55],        color: '#9be564' },
    { id: 'strongBuy',  label: 'Strong Buy',  pts: [45, 80, 100, 100],     color: '#1ff2a5' }
  ];

  // Complete 3x3 rule base. Rows follow RSI_SETS, columns follow SENTIMENT_SETS.
  const RULES = [
    ['hold',       'buy',  'strongBuy'],   // RSI oversold
    ['sell',       'hold', 'buy'],         // RSI neutral
    ['strongSell', 'sell', 'hold']         // RSI overbought
  ];

  // Execution zones, read on the raw score: <= -41, <= -11, <= +10, <= +40, above.
  const ZONES = [
    { id: 'strongSell', action: 'Strong Sell / Short',  range: '\u2212100 to \u221241', max: -41, stance: 'Open or hold a short position' },
    { id: 'sell',       action: 'Sell / Take Profit',   range: '\u221240 to \u221211',  max: -11, stance: 'Cut long exposure and take profit' },
    { id: 'hold',       action: 'Hold / No Trade',      range: '\u221210 to +10',       max: 10,  stance: 'Stay out of the market' },
    { id: 'buy',        action: 'Accumulate / Buy',     range: '+11 to +40',            max: 40,  stance: 'Build a long position gradually' },
    { id: 'strongBuy',  action: 'Strong Buy / Long',    range: '+41 to +100',           max: 100, stance: 'Hold a full long position' }
  ].map((z) => Object.assign({}, z, {
    label: OUTPUT_SETS.find((s) => s.id === z.id).label,
    color: OUTPUT_SETS.find((s) => s.id === z.id).color
  }));

  function mu(x, p) {
    const a = p[0], b = p[1], c = p[2], d = p[3];
    if (x < a || x > d) return 0;
    if (x >= b && x <= c) return 1;
    if (x < b) return (x - a) / (b - a);
    return (d - x) / (d - c);
  }

  function fuzzify(x, sets) {
    return sets.map((s) => ({ id: s.id, label: s.label, degree: mu(x, s.pts) }));
  }

  function classify(score) {
    if (score <= -41) return ZONES[0];
    if (score <= -11) return ZONES[1];
    if (score <= 10) return ZONES[2];
    if (score <= 40) return ZONES[3];
    return ZONES[4];
  }

  function infer(rsi, sentiment) {
    const rsiDeg = fuzzify(rsi, RSI_SETS);
    const sentDeg = fuzzify(sentiment, SENTIMENT_SETS);

    // Rule evaluation (AND = MIN) and per-output strength (rules sharing a consequent merge with MAX)
    const alpha = {};
    OUTPUT_SETS.forEach((s) => { alpha[s.id] = 0; });
    const firing = [];
    RULES.forEach((row, i) => {
      row.forEach((outId, j) => {
        const strength = Math.min(rsiDeg[i].degree, sentDeg[j].degree);
        firing.push({ rsiIndex: i, sentIndex: j, output: outId, strength: strength });
        alpha[outId] = Math.max(alpha[outId], strength);
      });
    });

    // Aggregation (MAX of truncated output sets) and centroid
    const n = Math.round((SIGNAL_MAX - SIGNAL_MIN) / STEP) + 1;
    const xs = new Array(n);
    const agg = new Array(n);
    let numerator = 0;
    let denominator = 0;
    for (let k = 0; k < n; k++) {
      const x = SIGNAL_MIN + k * STEP;
      let m = 0;
      for (let s = 0; s < OUTPUT_SETS.length; s++) {
        const set = OUTPUT_SETS[s];
        m = Math.max(m, Math.min(alpha[set.id], mu(x, set.pts)));
      }
      xs[k] = x;
      agg[k] = m;
      numerator += x * m;
      denominator += m;
    }

    const raw = denominator > 0 ? numerator / denominator : 0;
    const score = Math.round(raw * 1e6) / 1e6 + 0; // removes float noise and negative zero

    return {
      rsi: rsi,
      sentiment: sentiment,
      rsiDeg: rsiDeg,
      sentDeg: sentDeg,
      firing: firing,
      alpha: alpha,
      curve: { xs: xs, mu: agg },
      numerator: numerator,
      denominator: denominator,
      score: score,
      zone: classify(score)
    };
  }

  return {
    SIGNAL_MIN: SIGNAL_MIN,
    SIGNAL_MAX: SIGNAL_MAX,
    STEP: STEP,
    RSI_SETS: RSI_SETS,
    SENTIMENT_SETS: SENTIMENT_SETS,
    OUTPUT_SETS: OUTPUT_SETS,
    RULES: RULES,
    ZONES: ZONES,
    mu: mu,
    fuzzify: fuzzify,
    classify: classify,
    infer: infer
  };
})();


/* ==========================================================================
   2. USER INTERFACE
   Dials and gauge update live. The engine runs only on an explicit action:
   the "Analyser & Exécuter" button or an example scenario.
   ========================================================================== */
function initUI() {
  const E = Engine;
  const $ = (id) => document.getElementById(id);
  const reduceMotion = typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const STEP_MS = 700;
  const SET_COLORS = ['#4dd6ff', '#b69cff', '#ff7ac8'];
  const STAGE_NAMES = ['Fuzzification', 'Rule evaluation', 'Aggregation', 'Defuzzification', 'Execution signal'];
  const MINUS = '\u2212';

  const els = {
    cockpit: $('cockpit'),
    run: $('runBtn'),
    status: $('statusLine'),
    sr: $('srStatus'),
    rsi: $('rsiInput'),
    sent: $('sentInput'),
    led: $('ledScore'),
    ledZone: $('ledZone'),
    gauge: $('gaugeSvg'),
    needle: $('gNeedle'),
    stages: Array.prototype.slice.call(document.querySelectorAll('.stage')),
    steps: Array.prototype.slice.call(document.querySelectorAll('#stepper li')),
    bodies: [0, 1, 2, 3, 4].map((i) => $('stage' + i))
  };

  let timers = [];
  let ledFrame = 0;
  let lastRun = null;

  /* ---------- formatting ---------- */
  function fixed(v, d) {
    const n = Number(v.toFixed(d));
    return (n === 0 ? 0 : n).toFixed(d);
  }
  function signed(v, d) {
    const n = Number(v.toFixed(d));
    if (n === 0) return (0).toFixed(d);
    return (n < 0 ? MINUS : '+') + Math.abs(n).toFixed(d);
  }
  function grouped(v) {
    const s = Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return (Math.round(v * 100) < 0 ? MINUS : '') + s;
  }
  function axisNum(v, withPlus) {
    if (v === 0) return '0';
    if (v < 0) return MINUS + Math.abs(v);
    return (withPlus ? '+' : '') + v;
  }

  /* ---------- svg geometry ---------- */
  function polar(cx, cy, r, deg) {
    const a = deg * Math.PI / 180;
    return [cx + r * Math.sin(a), cy - r * Math.cos(a)];
  }
  function arcPath(cx, cy, r, a0, a1) {
    const p0 = polar(cx, cy, r, a0);
    const p1 = polar(cx, cy, r, a1);
    const large = Math.abs(a1 - a0) > 180 ? 1 : 0;
    return 'M' + p0[0].toFixed(2) + ' ' + p0[1].toFixed(2) +
      'A' + r + ' ' + r + ' 0 ' + large + ' 1 ' + p1[0].toFixed(2) + ' ' + p1[1].toFixed(2);
  }
  // Outline of a trapezoid as [x, membership] points. Shoulders at the domain edge start at full membership.
  function outline(p) {
    const pts = [];
    if (p[0] === p[1]) pts.push([p[1], 1]); else { pts.push([p[0], 0]); pts.push([p[1], 1]); }
    if (p[2] !== p[1]) pts.push([p[2], 1]);
    if (p[3] === p[2]) { /* closed by the shoulder */ } else pts.push([p[3], 0]);
    return pts;
  }

  /* ---------- input dials ---------- */
  const DIALS = [
    { id: 'rsi',  svg: $('rsiSvg'),  input: els.rsi,  min: 0,    max: 100, minor: 5,  major: 25, unit: 'RSI',
      fmt: (v) => String(v), edge: ['0', '100'] },
    { id: 'sent', svg: $('sentSvg'), input: els.sent, min: -100, max: 100, minor: 10, major: 50, unit: 'sentiment',
      fmt: (v) => signed(v, 0), edge: [MINUS + '100', '+100'] }
  ];
  const A0 = -135;
  const A1 = 135;

  function dialAngle(d, v) {
    return A0 + (v - d.min) / (d.max - d.min) * (A1 - A0);
  }

  function buildDial(d) {
    let h = '<defs><linearGradient id="dg-' + d.id + '" x1="0" y1="0" x2="1" y2="1">' +
      '<stop offset="0" stop-color="#4dd6ff"/><stop offset="1" stop-color="#ff7ac8"/></linearGradient></defs>';
    h += '<path class="dial-track" d="' + arcPath(100, 100, 80, A0, A1) + '"/>';
    h += '<path class="dial-progress" id="' + d.id + 'Progress" pathLength="100" stroke="url(#dg-' + d.id + ')" d="' +
      arcPath(100, 100, 80, A0, A1) + '"/>';
    for (let v = d.min; v <= d.max; v += d.minor) {
      const major = (v - d.min) % d.major === 0;
      const a = dialAngle(d, v);
      const p0 = polar(100, 100, 70, a);
      const p1 = polar(100, 100, major ? 57 : 63, a);
      h += '<line class="dial-tick' + (major ? ' is-major' : '') + '" x1="' + p0[0].toFixed(2) + '" y1="' + p0[1].toFixed(2) +
        '" x2="' + p1[0].toFixed(2) + '" y2="' + p1[1].toFixed(2) + '"/>';
    }
    const l0 = polar(100, 100, 94, A0);
    const l1 = polar(100, 100, 94, A1);
    h += '<text class="dial-edge" x="' + (l0[0] + 2).toFixed(1) + '" y="' + (l0[1] + 12).toFixed(1) + '" text-anchor="middle">' + d.edge[0] + '</text>';
    h += '<text class="dial-edge" x="' + (l1[0] - 2).toFixed(1) + '" y="' + (l1[1] + 12).toFixed(1) + '" text-anchor="middle">' + d.edge[1] + '</text>';
    h += '<g class="dial-needle" id="' + d.id + 'Needle"><path d="M97 106 L100 36 L103 106 Z"/>' +
      '<circle cx="100" cy="100" r="8"/><circle class="dial-core" cx="100" cy="100" r="3"/></g>';
    h += '<text class="dial-value" id="' + d.id + 'Value" x="100" y="152" text-anchor="middle"></text>';
    h += '<text class="dial-unit" x="100" y="170" text-anchor="middle">' + d.unit + '</text>';
    d.svg.innerHTML = h;
    d.needle = $(d.id + 'Needle');
    d.progress = $(d.id + 'Progress');
    d.value = $(d.id + 'Value');
  }

  function updateDial(d) {
    const v = Number(d.input.value);
    const pct = (v - d.min) / (d.max - d.min) * 100;
    d.needle.style.transform = 'rotate(' + dialAngle(d, v).toFixed(2) + 'deg)';
    d.progress.style.strokeDasharray = pct.toFixed(2) + ' 100';
    d.value.textContent = d.fmt(v);
  }

  /* ---------- output gauge ---------- */
  const G = { cx: 200, cy: 200, r: 138 };
  const BOUNDS = [-100, -41, -11, 10, 40, 100];
  const gAngle = (v) => v * 0.9;

  function buildGauge() {
    let zones = '';
    E.ZONES.forEach((z, i) => {
      const a0 = gAngle(BOUNDS[i]) + (i === 0 ? 0 : 0.7);
      const a1 = gAngle(BOUNDS[i + 1]) - (i === E.ZONES.length - 1 ? 0 : 0.7);
      zones += '<path class="g-zone" data-zone="' + z.id + '" style="--c:' + z.color + '" d="' + arcPath(G.cx, G.cy, G.r, a0, a1) + '"/>';
    });
    $('gZones').innerHTML = zones;

    let ticks = '';
    for (let v = -100; v <= 100; v += 10) {
      const major = v % 50 === 0;
      const a = gAngle(v);
      const p0 = polar(G.cx, G.cy, 124, a);
      const p1 = polar(G.cx, G.cy, major ? 110 : 117, a);
      ticks += '<line class="g-tick' + (major ? ' is-major' : '') + '" x1="' + p0[0].toFixed(2) + '" y1="' + p0[1].toFixed(2) +
        '" x2="' + p1[0].toFixed(2) + '" y2="' + p1[1].toFixed(2) + '"/>';
    }
    $('gTicks').innerHTML = ticks;

    let labels = '';
    [-100, -50, 0, 50, 100].forEach((v) => {
      const p = polar(G.cx, G.cy, 166, gAngle(v));
      labels += '<text class="g-label" x="' + p[0].toFixed(1) + '" y="' + (p[1] + 4).toFixed(1) + '" text-anchor="middle">' + axisNum(v, true) + '</text>';
    });
    $('gLabels').innerHTML = labels;
  }

  function animateLed(target) {
    cancelAnimationFrame(ledFrame);
    if (reduceMotion) { els.led.textContent = signed(target, 2); return; }
    const t0 = performance.now();
    const dur = 1100;
    const frame = (t) => {
      const p = Math.min(1, (t - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      els.led.textContent = signed(target * eased, 2);
      if (p < 1) ledFrame = requestAnimationFrame(frame);
      else els.led.textContent = signed(target, 2);
    };
    ledFrame = requestAnimationFrame(frame);
  }

  function resetGauge() {
    cancelAnimationFrame(ledFrame);
    els.needle.style.transform = 'rotate(-90deg)';
    Array.prototype.forEach.call(els.gauge.querySelectorAll('.g-zone'), (p) => {
      p.classList.remove('is-active', 'is-dim');
    });
    els.led.textContent = '--.--';
    els.ledZone.textContent = 'Awaiting analysis';
    els.cockpit.style.removeProperty('--zone');
    els.gauge.setAttribute('aria-label', 'Trading signal gauge, no result yet');
  }

  function setGauge(r) {
    els.needle.style.transform = 'rotate(' + (r.score * 0.9).toFixed(2) + 'deg)';
    Array.prototype.forEach.call(els.gauge.querySelectorAll('.g-zone'), (p) => {
      const on = p.getAttribute('data-zone') === r.zone.id;
      p.classList.toggle('is-active', on);
      p.classList.toggle('is-dim', !on);
    });
    els.cockpit.style.setProperty('--zone', r.zone.color);
    els.ledZone.textContent = r.zone.action;
    els.gauge.setAttribute('aria-label', 'Trading signal gauge: ' + signed(r.score, 2) + ', ' + r.zone.action);
    animateLed(r.score);
  }

  /* ---------- stage renderers ---------- */
  function membershipChart(sets, degrees, value, domain, withPlus, caption) {
    const W = 320, H = 150, L = 10, R = 10, T = 22, B = 26;
    const X = (v) => L + (v - domain[0]) / (domain[1] - domain[0]) * (W - L - R);
    const Y = (m) => T + (1 - m) * (H - T - B);
    let s = '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + caption + '" focusable="false">';
    s += '<line class="grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(0) + '" y2="' + Y(0) + '"/>';
    s += '<line class="grid is-dashed" x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(1) + '" y2="' + Y(1) + '"/>';
    sets.forEach((set, i) => {
      const pts = outline(set.pts);
      const line = pts.map((p, k) => (k ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join('');
      const area = line + 'L' + X(pts[pts.length - 1][0]).toFixed(1) + ' ' + Y(0) + 'L' + X(pts[0][0]).toFixed(1) + ' ' + Y(0) + 'Z';
      const deg = degrees[i].degree;
      s += '<path d="' + area + '" fill="' + SET_COLORS[i] + '" fill-opacity="' + (0.07 + 0.33 * deg).toFixed(2) + '"/>';
      s += '<path d="' + line + '" fill="none" stroke="' + SET_COLORS[i] + '" stroke-width="2" stroke-linejoin="round" stroke-opacity="' + (deg > 0 ? 1 : 0.5) + '"/>';
    });
    const mx = X(value);
    s += '<line class="marker-line" x1="' + mx.toFixed(1) + '" x2="' + mx.toFixed(1) + '" y1="' + (T - 8) + '" y2="' + Y(0) + '"/>';
    degrees.forEach((dg, i) => {
      if (dg.degree > 0) {
        s += '<circle cx="' + mx.toFixed(1) + '" cy="' + Y(dg.degree).toFixed(1) + '" r="4.5" fill="' + SET_COLORS[i] + '" stroke="#fff" stroke-width="1.5"/>';
      }
    });
    const anchor = mx > W - 46 ? 'end' : (mx < 46 ? 'start' : 'middle');
    s += '<text class="marker-label" x="' + mx.toFixed(1) + '" y="10" text-anchor="' + anchor + '">' + (withPlus ? signed(value, 0) : value) + '</text>';
    const mid = (domain[0] + domain[1]) / 2;
    [domain[0], mid, domain[1]].forEach((v, k) => {
      s += '<text class="axis" x="' + X(v).toFixed(1) + '" y="' + (H - 8) + '" text-anchor="' + (k === 0 ? 'start' : (k === 2 ? 'end' : 'middle')) + '">' + axisNum(v, withPlus) + '</text>';
    });
    s += '</svg>';
    return s;
  }

  function degreeList(degrees) {
    let h = '<ul class="deg-list">';
    degrees.forEach((dg, i) => {
      h += '<li style="--c:' + SET_COLORS[i] + ';--w:' + (dg.degree * 100).toFixed(1) + '%">' +
        '<span class="deg-name"><i></i>' + dg.label + '</span>' +
        '<span class="deg-track"><b></b></span>' +
        '<span class="deg-val">' + fixed(dg.degree, 2) + '</span></li>';
    });
    return h + '</ul>';
  }

  function renderFuzzification(r) {
    return '<div class="fz-grid">' +
      '<section class="fz-panel"><h4 class="fz-title">RSI <span>' + r.rsi + '</span></h4>' +
      membershipChart(E.RSI_SETS, r.rsiDeg, r.rsi, [0, 100], false, 'RSI membership functions with the input marked') +
      degreeList(r.rsiDeg) + '</section>' +
      '<section class="fz-panel"><h4 class="fz-title">Market sentiment <span>' + signed(r.sentiment, 0) + '</span></h4>' +
      membershipChart(E.SENTIMENT_SETS, r.sentDeg, r.sentiment, [-100, 100], true, 'Sentiment membership functions with the input marked') +
      degreeList(r.sentDeg) + '</section></div>';
  }

  function renderRules(r) {
    const outSet = (id) => E.OUTPUT_SETS.find((s) => s.id === id);
    let h = '<div class="matrix" role="table" aria-label="Rule base with firing strengths">';
    h += '<div class="mx-corner" role="presentation">RSI \u00d7 sentiment</div>';
    r.sentDeg.forEach((d, j) => {
      h += '<div class="mx-head" role="columnheader" style="--c:' + SET_COLORS[j] + '"><span>' + d.label + '</span><b>' + fixed(d.degree, 2) + '</b></div>';
    });
    let active = 0;
    r.rsiDeg.forEach((rd, i) => {
      h += '<div class="mx-head is-row" role="rowheader" style="--c:' + SET_COLORS[i] + '"><span>' + rd.label + '</span><b>' + fixed(rd.degree, 2) + '</b></div>';
      r.sentDeg.forEach((sd, j) => {
        const rule = r.firing.find((f) => f.rsiIndex === i && f.sentIndex === j);
        const out = outSet(rule.output);
        const on = rule.strength > 0;
        if (on) active++;
        h += '<div class="mx-cell' + (on ? ' is-on' : '') + '" role="cell" style="--c:' + out.color + ';--s:' + rule.strength.toFixed(3) +
          '" title="min(' + fixed(rd.degree, 2) + ', ' + fixed(sd.degree, 2) + ') = ' + fixed(rule.strength, 2) + '">' +
          '<span class="mx-out">' + out.label + '</span><span class="mx-str">' + fixed(rule.strength, 2) + '</span></div>';
      });
    });
    h += '</div>';
    h += '<p class="note">' + active + (active === 1 ? ' rule fires' : ' rules fire') + ' out of 9. The number in each cell is min(RSI degree, sentiment degree).</p>';
    return h;
  }

  function signalChart(r, mode) {
    const W = 440, H = 190, L = 34, R = 12, T = 18, B = 30;
    const X = (v) => L + (v + 100) / 200 * (W - L - R);
    const Y = (m) => T + (1 - m) * (H - T - B);
    const base = Y(0);
    const xs = r.curve.xs;
    const f = (n) => n.toFixed(1);
    const label = mode === 'agg' ? 'Output sets cut at their rule strength and merged' : 'Aggregated shape with the centroid marked';
    let s = '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + label + '" focusable="false">';

    if (mode === 'def') {
      s += '<defs><linearGradient id="defFill" gradientUnits="userSpaceOnUse" x1="' + X(-100) + '" x2="' + X(100) + '" y1="0" y2="0">';
      E.OUTPUT_SETS.forEach((set, i) => {
        s += '<stop offset="' + (i / (E.OUTPUT_SETS.length - 1)) + '" stop-color="' + set.color + '" stop-opacity="0.6"/>';
      });
      s += '</linearGradient></defs>';
    }

    [0, 0.5, 1].forEach((m) => {
      s += '<line class="grid' + (m === 0 ? '' : ' is-dashed') + '" x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(m) + '" y2="' + Y(m) + '"/>';
      s += '<text class="axis" x="' + (L - 6) + '" y="' + (Y(m) + 3) + '" text-anchor="end">' + (m === 0 ? '0' : m === 1 ? '1' : '0.5') + '</text>';
    });
    [-100, -50, 0, 50, 100].forEach((v) => {
      s += '<text class="axis" x="' + X(v) + '" y="' + (H - 9) + '" text-anchor="middle">' + axisNum(v, true) + '</text>';
    });

    if (mode === 'agg') {
      E.OUTPUT_SETS.forEach((set) => {
        const pts = outline(set.pts);
        const d = pts.map((p, k) => (k ? 'L' : 'M') + f(X(p[0])) + ' ' + f(Y(p[1]))).join('');
        s += '<path d="' + d + '" fill="none" stroke="' + set.color + '" stroke-opacity="0.45" stroke-dasharray="3 4" stroke-width="1.2"/>';
      });
      E.OUTPUT_SETS.forEach((set) => {
        const a = r.alpha[set.id];
        if (a <= 0) return;
        let line = '';
        xs.forEach((x, k) => {
          const m = Math.min(a, E.mu(x, set.pts));
          line += (k ? 'L' : 'M') + f(X(x)) + ' ' + f(Y(m));
        });
        s += '<path d="' + line + 'L' + f(X(xs[xs.length - 1])) + ' ' + f(base) + 'L' + f(X(xs[0])) + ' ' + f(base) + 'Z" fill="' + set.color + '" fill-opacity="0.34"/>';
        s += '<path d="' + line + '" fill="none" stroke="' + set.color + '" stroke-width="1.6"/>';
        const cx = Math.max(L + 18, Math.min(W - R - 18, X((set.pts[1] + set.pts[2]) / 2)));
        s += '<text class="alpha-label" x="' + f(cx) + '" y="' + f(Y(a) - 6) + '" text-anchor="middle">\u03b1 ' + fixed(a, 2) + '</text>';
      });
    }

    let top = '';
    r.curve.mu.forEach((m, k) => { top += (k ? 'L' : 'M') + f(X(xs[k])) + ' ' + f(Y(m)); });
    if (mode === 'def') {
      s += '<path d="' + top + 'L' + f(X(xs[xs.length - 1])) + ' ' + f(base) + 'L' + f(X(xs[0])) + ' ' + f(base) + 'Z" fill="url(#defFill)"/>';
    }
    s += '<path d="' + top + '" fill="none" stroke="#eef1ff" stroke-width="' + (mode === 'agg' ? 2 : 1.6) + '" stroke-linejoin="round"/>';

    if (mode === 'def' && r.denominator > 0) {
      const cx = X(r.score);
      const anchor = cx > W - 90 ? 'end' : (cx < L + 90 ? 'start' : 'middle');
      s += '<line class="centroid-line" x1="' + f(cx) + '" x2="' + f(cx) + '" y1="' + (T - 4) + '" y2="' + f(base) + '" style="--c:' + r.zone.color + '"/>';
      s += '<path d="M' + f(cx - 6) + ' ' + f(base + 8) + 'L' + f(cx + 6) + ' ' + f(base + 8) + 'L' + f(cx) + ' ' + f(base + 1) + 'Z" fill="' + r.zone.color + '"/>';
      s += '<text class="centroid-label" x="' + f(cx) + '" y="10" text-anchor="' + anchor + '">x* = ' + signed(r.score, 2) + '</text>';
    }
    return s + '</svg>';
  }

  function renderAggregation(r) {
    let h = signalChart(r, 'agg');
    h += '<ul class="legend">';
    E.OUTPUT_SETS.forEach((set) => {
      const a = r.alpha[set.id];
      h += '<li class="legend-item' + (a > 0 ? ' is-on' : '') + '" style="--c:' + set.color + '"><i></i>' + set.label + '<b>' + fixed(a, 2) + '</b></li>';
    });
    return h + '</ul>';
  }

  function renderDefuzzification(r) {
    if (r.denominator <= 0) {
      return signalChart(r, 'def') + '<p class="note">No rule fired, so the score defaults to 0.</p>';
    }
    return signalChart(r, 'def') +
      '<div class="formula">' +
      '<p class="formula-eq">x* = \u03a3 x\u00b7\u03bc(x) \u00f7 \u03a3 \u03bc(x)</p>' +
      '<dl><dt>\u03a3 x\u00b7\u03bc(x)</dt><dd>' + grouped(r.numerator) + '</dd>' +
      '<dt>\u03a3 \u03bc(x)</dt><dd>' + grouped(r.denominator) + '</dd>' +
      '<dt>x*</dt><dd class="is-result" style="--c:' + r.zone.color + '">' + signed(r.score, 2) + '</dd></dl>' +
      '</div><p class="note">Sampled every 0.5 from \u2212100 to +100, 401 points.</p>';
  }

  function renderSignal(r) {
    const exposure = Math.round(Math.abs(r.score));
    let h = '<div class="signal" style="--c:' + r.zone.color + '">' +
      '<div class="signal-badge"><span class="signal-score">' + signed(r.score, 2) + '</span>' +
      '<span class="signal-name">' + r.zone.action + '</span></div>' +
      '<dl class="signal-facts">' +
      '<dt>Inputs</dt><dd>RSI ' + r.rsi + ', sentiment ' + signed(r.sentiment, 0) + '</dd>' +
      '<dt>Stance</dt><dd>' + r.zone.stance + '</dd>' +
      '<dt>Exposure</dt><dd>' + exposure + '% of the maximum position</dd></dl></div>';
    h += '<table class="zone-table"><caption class="sr-only">Execution zones</caption><tbody>';
    E.ZONES.forEach((z) => {
      h += '<tr' + (z.id === r.zone.id ? ' class="is-active"' : '') + ' style="--c:' + z.color + '">' +
        '<td><i></i></td><th scope="row">' + z.action + '</th><td>' + z.range + '</td></tr>';
    });
    return h + '</tbody></table>';
  }

  /* ---------- pipeline control ---------- */
  function clearTimers() {
    timers.forEach(clearTimeout);
    timers = [];
  }

  function setStepper(active) {
    els.steps.forEach((li, i) => {
      li.classList.toggle('is-done', i < active);
      li.classList.toggle('is-current', i === active);
    });
  }

  function resetStages() {
    els.stages.forEach((st) => st.classList.remove('is-visible', 'is-current'));
    setStepper(-1);
  }

  function renderStages(r) {
    els.bodies[0].innerHTML = renderFuzzification(r);
    els.bodies[1].innerHTML = renderRules(r);
    els.bodies[2].innerHTML = renderAggregation(r);
    els.bodies[3].innerHTML = renderDefuzzification(r);
    els.bodies[4].innerHTML = renderSignal(r);
  }

  function revealStage(i, r) {
    els.stages.forEach((st, k) => st.classList.toggle('is-current', k === i));
    els.stages[i].classList.add('is-visible');
    if (i < 4) {
      setStepper(i);
      els.status.textContent = 'Stage ' + (i + 1) + ' of 5: ' + STAGE_NAMES[i] + '.';
      return;
    }
    setStepper(5);
    setGauge(r);
    const msg = 'Analysis complete. ' + r.zone.action + ' at ' + signed(r.score, 2) + '.';
    els.status.textContent = msg;
    els.sr.textContent = msg;
    els.run.removeAttribute('aria-busy');
  }

  function pulseDials() {
    ['rsiDial', 'sentDial'].forEach((id) => {
      const el = $(id);
      el.classList.remove('is-sampled');
      void el.offsetWidth;
      el.classList.add('is-sampled');
    });
  }

  function runAnalysis() {
    const rsi = Number(els.rsi.value);
    const sentiment = Number(els.sent.value);
    clearTimers();
    resetStages();
    resetGauge();
    const result = E.infer(rsi, sentiment);
    lastRun = { rsi: rsi, sentiment: sentiment };
    els.cockpit.classList.remove('is-stale');
    els.run.setAttribute('aria-busy', 'true');
    els.sr.textContent = 'Analysis started.';
    pulseDials();
    renderStages(result);
    if (reduceMotion) {
      [0, 1, 2, 3, 4].forEach((i) => revealStage(i, result));
      return;
    }
    [0, 1, 2, 3, 4].forEach((i) => {
      timers.push(setTimeout(() => revealStage(i, result), i === 0 ? 120 : 120 + i * STEP_MS));
    });
  }

  function markStaleIfChanged() {
    if (!lastRun) return;
    const changed = Number(els.rsi.value) !== lastRun.rsi || Number(els.sent.value) !== lastRun.sentiment;
    els.cockpit.classList.toggle('is-stale', changed);
    if (changed) {
      els.status.textContent = 'Inputs changed since the last analysis. Run it again to update the signal.';
    }
  }

  /* ---------- wiring ---------- */
  DIALS.forEach((d) => {
    buildDial(d);
    updateDial(d);
    d.input.addEventListener('input', () => {   // live feedback only, never runs the engine
      updateDial(d);
      markStaleIfChanged();
    });
  });
  buildGauge();
  resetGauge();
  setStepper(-1);

  els.run.addEventListener('click', runAnalysis);
  Array.prototype.forEach.call(document.querySelectorAll('.chip'), (chip) => {
    chip.addEventListener('click', () => {
      els.rsi.value = chip.getAttribute('data-rsi');
      els.sent.value = chip.getAttribute('data-sent');
      DIALS.forEach(updateDial);
      runAnalysis();
    });
  });
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initUI);
  else initUI();
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Engine: Engine };
}