'use strict';

/* =============================================================================
   FUZZY LOGIC TRADING ENGINE
   -----------------------------------------------------------------------------
   Système d'inférence floue de type Mamdani combinant :
     - Entrée 1 : RSI (Relative Strength Index)      domaine [0, 100]
     - Entrée 2 : Market Sentiment Index               domaine [-100, +100]
   pour produire :
     - Sortie   : Trading Signal & Exposition          domaine [-100, +100]

   Pipeline : Fuzzification -> Évaluation des règles (MIN) -> Agrégation (MAX)
              -> Défuzzification (Centre de Gravité / Centroïde)
   ============================================================================= */

/* -----------------------------------------------------------------------------
   0. UTILITAIRES
   ----------------------------------------------------------------------------- */

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function hexToRgb(hex) {
  const clean = hex.replace('#', '');
  const full = clean.length === 3 ? clean.split('').map((c) => c + c).join('') : clean;
  const intVal = parseInt(full, 16);
  return { r: (intVal >> 16) & 255, g: (intVal >> 8) & 255, b: intVal & 255 };
}

function hexToRgba(hex, alpha) {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function relativeLuminance(hex) {
  const { r, g, b } = hexToRgb(hex);
  const channels = [r, g, b].map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function bestTextColor(bgHex) {
  return relativeLuminance(bgHex) > 0.45 ? '#0a0e14' : '#f3f5f9';
}

/* -----------------------------------------------------------------------------
   1. FONCTIONS D'APPARTENANCE (Membership Functions)
   ----------------------------------------------------------------------------- */

/**
 * Fonction d'appartenance trapézoïdale.
 * a, d : pieds du trapèze (appartenance = 0)
 * b, c : épaules du trapèze (appartenance = 1 sur [b, c])
 * Le cas a === b (respectivement c === d) supprime la rampe correspondante,
 * ce qui permet de modéliser un plateau qui commence/finit au bord du domaine.
 */
function trapMF(x, a, b, c, d) {
  if (x < a || x > d) return 0;
  if (x >= b && x <= c) return 1;
  if (x < b) return b === a ? 1 : (x - a) / (b - a);
  return d === c ? 1 : (d - x) / (d - c);
}

/**
 * Fonction d'appartenance triangulaire — cas particulier du trapèze où
 * l'épaule haute est réduite à un point unique (b === c).
 */
function triMF(x, a, b, c) {
  return trapMF(x, a, b, b, c);
}

function evalSet(setDef, x) {
  return setDef.type === 'trap'
    ? trapMF(x, setDef.params[0], setDef.params[1], setDef.params[2], setDef.params[3])
    : triMF(x, setDef.params[0], setDef.params[1], setDef.params[2]);
}

/* -----------------------------------------------------------------------------
   2. VARIABLES LINGUISTIQUES
   ----------------------------------------------------------------------------- */

const RSI_SETS = {
  oversold: { type: 'trap', params: [0, 0, 30, 50], label: 'Survendu' },
  neutral: { type: 'tri', params: [30, 50, 70], label: 'Neutre' },
  overbought: { type: 'trap', params: [50, 70, 100, 100], label: 'Suracheté' },
};

const SENT_SETS = {
  bearish: { type: 'trap', params: [-100, -100, -50, -10], label: 'Baissier' },
  neutral: { type: 'tri', params: [-30, 0, 30], label: 'Neutre' },
  bullish: { type: 'trap', params: [10, 50, 100, 100], label: 'Haussier' },
};

const OUTPUT_SETS = {
  strongSell: { type: 'trap', params: [-100, -100, -70, -40], label: 'Vente Forte', short: 'S.SELL' },
  sell: { type: 'tri', params: [-55, -30, -5], label: 'Vente', short: 'SELL' },
  hold: { type: 'tri', params: [-15, 0, 15], label: 'Conserver', short: 'HOLD' },
  buy: { type: 'tri', params: [5, 30, 55], label: 'Achat', short: 'BUY' },
  strongBuy: { type: 'trap', params: [40, 70, 100, 100], label: 'Achat Fort', short: 'S.BUY' },
};

/* Attribution des couleurs : palette "technique" (bleu/violet) pour le RSI,
   palette "signal" (rouge/ambre/vert) pour le Sentiment et pour la Sortie —
   la sortie et le sentiment partagent un langage visuel car les deux sont,
   par nature, des grandeurs directionnelles (baissier <-> haussier). */
const RSI_COLOR_KEY = { oversold: 'cyan', neutral: 'indigo', overbought: 'violet' };
const SENT_COLOR_KEY = { bearish: 'red', neutral: 'amber', bullish: 'green' };
const OUTPUT_COLOR_KEY = { strongSell: 'redStrong', sell: 'red', hold: 'amber', buy: 'green', strongBuy: 'greenStrong' };

let COLORS = {};
function colorFor(colorKeyMap, key) {
  return COLORS[colorKeyMap[key]];
}

/* -----------------------------------------------------------------------------
   3. BASE DE RÈGLES FLOUES (matrice complète 3x3 = 9 règles)
   -----------------------------------------------------------------------------
   Logique retenue : le RSI porte un biais directionnel implicite (Survendu ->
   tendance haussière potentielle de retournement ; Suracheté -> tendance
   baissière potentielle de retournement). Le Sentiment confirme, contredit ou
   reste neutre vis-à-vis de ce biais :
     - Biais confirmé (même direction)      -> signal amplifié
     - Sentiment neutre                      -> signal modéré
     - Signaux contradictoires               -> Conserver (Hold), prudence
   ----------------------------------------------------------------------------- */

const RULE_BASE = [
  { rsi: 'oversold', sent: 'bearish', out: 'hold' },
  { rsi: 'oversold', sent: 'neutral', out: 'buy' },
  { rsi: 'oversold', sent: 'bullish', out: 'strongBuy' },

  { rsi: 'neutral', sent: 'bearish', out: 'sell' },
  { rsi: 'neutral', sent: 'neutral', out: 'hold' },
  { rsi: 'neutral', sent: 'bullish', out: 'buy' },

  { rsi: 'overbought', sent: 'bearish', out: 'strongSell' },
  { rsi: 'overbought', sent: 'neutral', out: 'sell' },
  { rsi: 'overbought', sent: 'bullish', out: 'hold' },
];

/* -----------------------------------------------------------------------------
   4. MOTEUR D'INFÉRENCE
   ----------------------------------------------------------------------------- */

function fuzzify(sets, x) {
  const degrees = {};
  for (const key in sets) degrees[key] = evalSet(sets[key], x);
  return degrees;
}

/** Évaluation des règles : opérateur AND = MIN des degrés d'appartenance. */
function evaluateRules(rsiDeg, sentDeg) {
  return RULE_BASE.map((rule) => ({
    ...rule,
    strength: Math.min(rsiDeg[rule.rsi], sentDeg[rule.sent]),
  }));
}

/** Agrégation intra-ensemble : plusieurs règles peuvent pointer vers le même
 *  ensemble de sortie ; on retient la force de déclenchement maximale (MAX). */
function aggregateOutputStrengths(ruleResults) {
  const strengths = {};
  for (const key in OUTPUT_SETS) strengths[key] = 0;
  ruleResults.forEach((r) => {
    if (r.strength > strengths[r.out]) strengths[r.out] = r.strength;
  });
  return strengths;
}

/** Échantillonne la courbe de sortie agrégée : pour chaque point y du domaine,
 *  on écrête (MIN) chaque ensemble de sortie par sa force de déclenchement,
 *  puis on prend l'enveloppe supérieure (MAX) des 5 ensembles écrêtés. */
function sampleAggregatedCurve(outputStrengths, step) {
  const points = [];
  for (let y = -100; y <= 100 + 1e-9; y += step) {
    const yr = Math.min(100, Math.round(y * 100) / 100);
    let agg = 0;
    for (const key in OUTPUT_SETS) {
      const clipped = Math.min(outputStrengths[key], evalSet(OUTPUT_SETS[key], yr));
      if (clipped > agg) agg = clipped;
    }
    points.push({ x: yr, y: agg });
  }
  return points;
}

/** Défuzzification par Centre de Gravité (Centroïde), intégration numérique
 *  par sommation discrète sur le domaine échantillonné. */
function defuzzifyCentroid(curvePoints) {
  let numerator = 0;
  let denominator = 0;
  for (const p of curvePoints) {
    numerator += p.x * p.y;
    denominator += p.y;
  }
  return denominator === 0 ? 0 : numerator / denominator;
}

/** Classification du score net selon la table d'interprétation du cahier des
 *  charges (section 4.2). */
function classifySignal(score) {
  if (score <= -41) {
    return {
      key: 'strongSell',
      title: 'Vente Forte',
      sub: 'Strong Sell / Short',
      text: 'Convergence baissière marquée entre le momentum RSI et le sentiment de marché. Une position courte ou une réduction forte de l’exposition est justifiée.',
    };
  }
  if (score <= -11) {
    return {
      key: 'sell',
      title: 'Vente Modérée',
      sub: 'Sell / Take Profit',
      text: 'La pondération technique et le sentiment penchent vers la baisse. Une prise de profit partielle ou un allègement de position est recommandé.',
    };
  }
  if (score <= 10) {
    return {
      key: 'hold',
      title: 'Neutre / Conserver',
      sub: 'Hold / No Trade',
      text: 'Aucun consensus fort ne se dégage entre le RSI et le sentiment de marché. Le système recommande de rester en observation.',
    };
  }
  if (score <= 40) {
    return {
      key: 'buy',
      title: 'Achat Modéré',
      sub: 'Accumulate / Buy',
      text: 'Le momentum et le sentiment convergent modérément à la hausse. Une accumulation progressive est envisageable.',
    };
  }
  return {
    key: 'strongBuy',
    title: 'Achat Fort',
    sub: 'Strong Buy / Long',
    text: 'Convergence haussière marquée entre le RSI et le sentiment de marché. Signal d’entrée fort en position longue.',
  };
}

/* -----------------------------------------------------------------------------
   5. RENDU — GRAPHIQUES DES FONCTIONS D'APPARTENANCE (SVG)
   ----------------------------------------------------------------------------- */

function buildMFChartSVG(setsObj, colorKeyMap, domainMin, domainMax, currentX) {
  const W = 360;
  const H = 130;
  const padL = 27;
  const padR = 8;
  const padT = 12;
  const padB = 20;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const xToPx = (x) => padL + ((x - domainMin) / (domainMax - domainMin)) * plotW;
  const yToPx = (y) => padT + (1 - y) * plotH;

  let svg = '';

  svg += `<line x1="${padL}" y1="${yToPx(0)}" x2="${W - padR}" y2="${yToPx(0)}" stroke="${COLORS.line}" stroke-width="1"/>`;
  svg += `<line x1="${padL}" y1="${yToPx(1)}" x2="${W - padR}" y2="${yToPx(1)}" stroke="${COLORS.lineSoft}" stroke-width="1" stroke-dasharray="2 3"/>`;
  svg += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="${COLORS.line}" stroke-width="1"/>`;

  svg += `<text x="${padL - 6}" y="${yToPx(1) + 3}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="8" fill="${COLORS.mistDim}">1</text>`;
  svg += `<text x="${padL - 6}" y="${yToPx(0) + 3}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="8" fill="${COLORS.mistDim}">0</text>`;

  const mid = (domainMin + domainMax) / 2;
  svg += `<text x="${xToPx(domainMin)}" y="${H - 6}" text-anchor="start" font-family="IBM Plex Mono, monospace" font-size="8" fill="${COLORS.mistDim}">${domainMin}</text>`;
  svg += `<text x="${xToPx(mid)}" y="${H - 6}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="8" fill="${COLORS.mistDim}">${mid}</text>`;
  svg += `<text x="${xToPx(domainMax)}" y="${H - 6}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="8" fill="${COLORS.mistDim}">${domainMax}</text>`;

  for (const key in setsObj) {
    const def = setsObj[key];
    const color = colorFor(colorKeyMap, key);
    let pts = '';
    for (let x = domainMin; x <= domainMax; x += 1) {
      pts += `${xToPx(x)},${yToPx(evalSet(def, x))} `;
    }
    svg += `<polyline points="${pts.trim()}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  }

  if (currentX !== undefined && currentX !== null) {
    const mx = xToPx(clamp(currentX, domainMin, domainMax));
    svg += `<line x1="${mx}" y1="${padT}" x2="${mx}" y2="${H - padB}" stroke="${COLORS.ink}" stroke-width="1" stroke-dasharray="3 3" opacity="0.55"/>`;
    for (const key in setsObj) {
      const def = setsObj[key];
      const color = colorFor(colorKeyMap, key);
      const my = yToPx(evalSet(def, currentX));
      svg += `<circle cx="${mx}" cy="${my}" r="3.2" fill="${color}" stroke="${COLORS.bg}" stroke-width="1.2"/>`;
    }
  }

  return svg;
}

function buildOutputChartSVG(aggregatedCurve, score, categoryColor) {
  const W = 360;
  const H = 150;
  const padL = 27;
  const padR = 8;
  const padT = 20;
  const padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const domainMin = -100;
  const domainMax = 100;

  const xToPx = (x) => padL + ((x - domainMin) / (domainMax - domainMin)) * plotW;
  const yToPx = (y) => padT + (1 - y) * plotH;

  let svg = '';

  svg += `<line x1="${padL}" y1="${yToPx(0)}" x2="${W - padR}" y2="${yToPx(0)}" stroke="${COLORS.line}" stroke-width="1"/>`;
  svg += `<line x1="${padL}" y1="${yToPx(1)}" x2="${W - padR}" y2="${yToPx(1)}" stroke="${COLORS.lineSoft}" stroke-width="1" stroke-dasharray="2 3"/>`;
  svg += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="${COLORS.line}" stroke-width="1"/>`;

  svg += `<text x="${padL - 6}" y="${yToPx(1) + 3}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="8" fill="${COLORS.mistDim}">1</text>`;
  svg += `<text x="${padL - 6}" y="${yToPx(0) + 3}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="8" fill="${COLORS.mistDim}">0</text>`;
  svg += `<text x="${xToPx(domainMin)}" y="${H - 6}" text-anchor="start" font-family="IBM Plex Mono, monospace" font-size="8" fill="${COLORS.mistDim}">-100</text>`;
  svg += `<text x="${xToPx(0)}" y="${H - 6}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="8" fill="${COLORS.mistDim}">0</text>`;
  svg += `<text x="${xToPx(domainMax)}" y="${H - 6}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="8" fill="${COLORS.mistDim}">+100</text>`;

  // Ensembles de sortie de référence (traits fins, atténués)
  for (const key in OUTPUT_SETS) {
    const def = OUTPUT_SETS[key];
    const color = colorFor(OUTPUT_COLOR_KEY, key);
    let pts = '';
    for (let x = domainMin; x <= domainMax; x += 1) {
      pts += `${xToPx(x)},${yToPx(evalSet(def, x))} `;
    }
    svg += `<polyline points="${pts.trim()}" fill="none" stroke="${color}" stroke-width="1.2" opacity="0.32"/>`;
  }

  // Forme agrégée (résultat de l'écrêtage MIN + enveloppe MAX)
  const baseline = yToPx(0);
  let path = `M ${xToPx(aggregatedCurve[0].x)},${baseline} `;
  aggregatedCurve.forEach((p) => {
    path += `L ${xToPx(p.x)},${yToPx(p.y)} `;
  });
  path += `L ${xToPx(aggregatedCurve[aggregatedCurve.length - 1].x)},${baseline} Z`;
  svg += `<path d="${path}" fill="${hexToRgba(categoryColor, 0.28)}" stroke="${categoryColor}" stroke-width="1.8" stroke-linejoin="round"/>`;

  // Ligne du centroïde (score net défuzzifié)
  const cx = xToPx(clamp(score, domainMin, domainMax));
  svg += `<line x1="${cx}" y1="${padT - 8}" x2="${cx}" y2="${H - padB}" stroke="${COLORS.ink}" stroke-width="1.4" stroke-dasharray="4 3"/>`;
  svg += `<circle cx="${cx}" cy="${padT - 8}" r="3" fill="${COLORS.ink}"/>`;
  svg += `<text x="${clamp(cx, 30, W - 30)}" y="${padT - 11}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="9" font-weight="600" fill="${COLORS.ink}">${score.toFixed(1)}</text>`;

  return svg;
}

/* -----------------------------------------------------------------------------
   6. RENDU — LISTES DE DEGRÉS D'APPARTENANCE
   ----------------------------------------------------------------------------- */

function updateDegreeList(containerId, sets, degrees, colorKeyMap) {
  const container = document.getElementById(containerId);
  container.innerHTML = Object.keys(sets)
    .map((key) => {
      const color = colorFor(colorKeyMap, key);
      const val = degrees[key];
      const pct = clamp(val * 100, 0, 100).toFixed(0);
      return `<div class="degree-row">
        <span class="degree-swatch" style="background:${color}"></span>
        <span class="degree-name">${sets[key].label}</span>
        <span class="degree-value">${val.toFixed(2)}</span>
        <div class="degree-bar-track"><div class="degree-bar-fill" style="width:${pct}%; background:${color}"></div></div>
      </div>`;
    })
    .join('');
}

/* -----------------------------------------------------------------------------
   7. RENDU — MATRICE DE RÈGLES (3x3)
   ----------------------------------------------------------------------------- */

function buildRuleMatrixSkeleton() {
  const container = document.getElementById('ruleMatrixContainer');
  const rsiKeys = Object.keys(RSI_SETS);
  const sentKeys = Object.keys(SENT_SETS);

  let html = '<table class="rule-table"><thead><tr><th class="corner"></th>';
  sentKeys.forEach((sk) => {
    html += `<th>${SENT_SETS[sk].label}</th>`;
  });
  html += '</tr></thead><tbody>';

  rsiKeys.forEach((rk) => {
    html += `<tr><th>${RSI_SETS[rk].label}</th>`;
    sentKeys.forEach((sk) => {
      const rule = RULE_BASE.find((r) => r.rsi === rk && r.sent === sk);
      html += `<td class="rule-cell" id="cell-${rk}-${sk}">
        <span class="cell-out">${OUTPUT_SETS[rule.out].short}</span>
        <span class="cell-strength" id="strength-${rk}-${sk}">0.00</span>
      </td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  container.innerHTML = html;
}

function updateRuleMatrix(ruleResults) {
  ruleResults.forEach((r) => {
    const cell = document.getElementById(`cell-${r.rsi}-${r.sent}`);
    const strengthEl = document.getElementById(`strength-${r.rsi}-${r.sent}`);
    const color = colorFor(OUTPUT_COLOR_KEY, r.out);
    const alpha = 0.1 + r.strength * 0.55;

    cell.style.backgroundColor = hexToRgba(color, alpha);
    cell.style.color = r.strength > 0.3 ? bestTextColor(color) : COLORS.mist;
    strengthEl.textContent = r.strength.toFixed(2);

    if (r.strength > 0.01) {
      cell.classList.add('active');
      cell.style.boxShadow = `inset 0 0 0 1px ${hexToRgba(color, 0.75)}`;
    } else {
      cell.classList.remove('active');
      cell.style.boxShadow = 'none';
    }
  });
}

function updateActiveRulesList(ruleResults) {
  const container = document.getElementById('activeRulesList');
  const active = ruleResults.filter((r) => r.strength > 0.005).sort((a, b) => b.strength - a.strength);

  if (active.length === 0) {
    container.innerHTML = '<div class="active-rules-empty">Aucune règle active pour ces valeurs.</div>';
    return;
  }

  container.innerHTML = active
    .map((r) => {
      const color = colorFor(OUTPUT_COLOR_KEY, r.out);
      return `<div class="active-rule-row" style="border-left-color:${color}">
        <span class="rule-text">SI RSI=<strong>${RSI_SETS[r.rsi].label}</strong> ET Sentiment=<strong>${SENT_SETS[r.sent].label}</strong> → <strong>${OUTPUT_SETS[r.out].label}</strong></span>
        <span class="rule-strength">${r.strength.toFixed(2)}</span>
      </div>`;
    })
    .join('');
}

/* -----------------------------------------------------------------------------
   8. RENDU — JAUGE HORIZONTALE (FUSION STRIP) & RECOMMANDATION
   ----------------------------------------------------------------------------- */

const FUSION_BOUNDARIES = [-100, -40, -10, 10, 40, 100];

function buildFusionAxis() {
  const strip = document.getElementById('fusionStrip');
  const ticksContainer = document.getElementById('fusionTicks');

  let lineHtml = '';
  let labelHtml = '';

  FUSION_BOUNDARIES.forEach((b) => {
    const pct = ((b - -100) / 200) * 100;
    if (b !== -100 && b !== 100) {
      lineHtml += `<div class="tick-line" style="left:${pct}%"></div>`;
    }
    labelHtml += `<span class="tick-label" style="left:${pct}%">${b > 0 ? `+${b}` : b}</span>`;
  });

  strip.insertAdjacentHTML('afterbegin', lineHtml);
  ticksContainer.innerHTML = labelHtml;
}

function updateFusionMarker(score) {
  const marker = document.getElementById('fusionMarker');
  const pct = clamp(((score - -100) / 200) * 100, 0, 100);
  marker.style.left = `${pct}%`;
}

function updateRecommendation(score, classification) {
  const badge = document.getElementById('recBadge');
  const text = document.getElementById('recText');
  const scoreEl = document.getElementById('scoreValue');
  const color = colorFor(OUTPUT_COLOR_KEY, classification.key);

  badge.textContent = `${classification.title} · ${classification.sub}`;
  badge.style.backgroundColor = color;
  badge.style.color = bestTextColor(color);

  text.innerHTML = `Score net : <strong>${score.toFixed(1)} / 100</strong>. ${classification.text}`;

  scoreEl.textContent = score.toFixed(1);
  scoreEl.style.color = color;
}

/* -----------------------------------------------------------------------------
   9. SCÉNARIOS RAPIDES (PRESETS)
   ----------------------------------------------------------------------------- */

const PRESETS = [
  { label: 'Achat Fort', rsi: 15, sent: 85 },
  { label: 'Vente Forte', rsi: 88, sent: -85 },
  { label: 'Neutre', rsi: 50, sent: 0 },
  { label: 'Signaux Contradictoires', rsi: 20, sent: -80 },
  { label: 'Zone de Transition', rsi: 40, sent: 20 },
];

function buildPresetButtons() {
  const container = document.getElementById('presetButtons');
  container.innerHTML = PRESETS.map(
    (p, i) => `<button type="button" class="preset-btn" data-index="${i}">${p.label}</button>`
  ).join('');

  container.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const p = PRESETS[Number(btn.dataset.index)];
      document.getElementById('rsiSlider').value = String(p.rsi);
      document.getElementById('sentSlider').value = String(p.sent);
      runAnalysis();
    });
  });
}

/* -----------------------------------------------------------------------------
   10. BOUCLE PRINCIPALE D'ANALYSE
   ----------------------------------------------------------------------------- */

function runAnalysis() {
  const rsiSlider = document.getElementById('rsiSlider');
  const sentSlider = document.getElementById('sentSlider');
  const rsi = Number(rsiSlider.value);
  const sent = Number(sentSlider.value);

  document.getElementById('rsiValue').textContent = String(rsi);
  document.getElementById('sentValue').textContent = sent > 0 ? `+${sent}` : String(sent);

  // Étape 1 — Fuzzification
  const rsiDeg = fuzzify(RSI_SETS, rsi);
  const sentDeg = fuzzify(SENT_SETS, sent);

  // Étape 2 — Évaluation des règles
  const ruleResults = evaluateRules(rsiDeg, sentDeg);

  // Étape 3 — Agrégation
  const outputStrengths = aggregateOutputStrengths(ruleResults);
  const aggregatedCurve = sampleAggregatedCurve(outputStrengths, 0.5);

  // Étape 4 — Défuzzification
  const score = defuzzifyCentroid(aggregatedCurve);
  const classification = classifySignal(score);
  const categoryColor = colorFor(OUTPUT_COLOR_KEY, classification.key);

  // Rendu — Stage 02
  document.getElementById('rsiChart').innerHTML = buildMFChartSVG(RSI_SETS, RSI_COLOR_KEY, 0, 100, rsi);
  document.getElementById('sentChart').innerHTML = buildMFChartSVG(SENT_SETS, SENT_COLOR_KEY, -100, 100, sent);
  updateDegreeList('rsiDegreeList', RSI_SETS, rsiDeg, RSI_COLOR_KEY);
  updateDegreeList('sentDegreeList', SENT_SETS, sentDeg, SENT_COLOR_KEY);

  // Rendu — Stage 03
  updateRuleMatrix(ruleResults);
  updateActiveRulesList(ruleResults);

  // Rendu — Stage 04
  updateFusionMarker(score);
  updateRecommendation(score, classification);
  document.getElementById('outputChart').innerHTML = buildOutputChartSVG(aggregatedCurve, score, categoryColor);
}

/* -----------------------------------------------------------------------------
   11. INITIALISATION
   ----------------------------------------------------------------------------- */

function init() {
  COLORS = {
    cyan: cssVar('--cyan'),
    indigo: cssVar('--indigo'),
    violet: cssVar('--violet'),
    red: cssVar('--red'),
    redStrong: cssVar('--red-strong'),
    amber: cssVar('--amber'),
    green: cssVar('--green'),
    greenStrong: cssVar('--green-strong'),
    ink: cssVar('--ink'),
    mist: cssVar('--mist'),
    mistDim: cssVar('--mist-dim'),
    line: cssVar('--line'),
    lineSoft: cssVar('--line-soft'),
    bg: cssVar('--bg'),
    bgAlt: cssVar('--bg-alt'),
  };

  buildRuleMatrixSkeleton();
  buildFusionAxis();
  buildPresetButtons();

  document.getElementById('rsiSlider').addEventListener('input', runAnalysis);
  document.getElementById('sentSlider').addEventListener('input', runAnalysis);
  document.getElementById('analyzeBtn').addEventListener('click', runAnalysis);

  runAnalysis();
}

document.addEventListener('DOMContentLoaded', init);