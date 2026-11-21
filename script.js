/* ==========================================================================
   University Student ML Lab Pro
   Muallif: Sadullayev Jaxongir | Fan: Sun'iy intellekt texnologiyalari
   Tuzilishi: 1) Ma'lumotlar  2) ML algoritmlari  3) Tekshirish hisoblari
              4) Holat  5) UI render  6) Ishga tushirish
   ========================================================================== */
'use strict';

/* ==========================================================================
   1. MA'LUMOTLAR (o'quv maqsadidagi namunaviy dataset, o'zgarmas)
   ========================================================================== */
const CLASS_NAMES = ['Oddiy', 'Faol'];
const FEATURES = [
  { key: 'score', label: "O'rtacha ball", short: 'Ball' },
  { key: 'att', label: 'Qatnashish', short: 'Qatnashish' }
];
const STUDENTS = Object.freeze([
  [40, 45, 'Oddiy'], [48, 72, 'Oddiy'], [55, 85, 'Oddiy'], [59, 92, 'Oddiy'],
  [62, 66, 'Oddiy'], [65, 79, 'Oddiy'], [67, 96, 'Oddiy'], [69, 83, 'Oddiy'],
  [73, 45, 'Oddiy'], [76, 56, 'Oddiy'], [81, 61, 'Oddiy'], [88, 69, 'Oddiy'],
  [90, 72, 'Oddiy'], [72, 77, 'Faol'], [74, 81, 'Faol'], [79, 86, 'Faol'],
  [82, 79, 'Faol'], [85, 91, 'Faol'], [92, 95, 'Faol'], [96, 88, 'Faol']
].map((r, i) => Object.freeze({ id: 'T' + (i + 1), score: r[0], att: r[1], cls: r[2] })));

const PROFILES = {
  active: { score: 80, att: 85, label: 'Faol talaba' },
  ordinary: { score: 55, att: 50, label: 'Oddiy talaba' },
  border: { score: 72, att: 75, label: 'Chegaraviy talaba' }
};
const KNN_K = 3;
const TREE_MAX_DEPTH = 2;
const KMEANS_Q = 3;
const KMEANS_INIT_IDS = ['T1', 'T15', 'T20'];
const KMEANS_MAX_ITER = 100;
const EPS = 1e-10;

/* ==========================================================================
   2. ML ALGORITMLARI
   ========================================================================== */
function idNum(id) { return parseInt(String(id).replace(/\D/g, ''), 10) || 0; }
function isNum(v) { return typeof v === 'number' && Number.isFinite(v); }

/** Min-Max chegaralari: har parametr uchun alohida, faqat berilgan o'quv ma'lumotlaridan. */
function computeBounds(students) {
  const out = {};
  FEATURES.forEach(f => {
    const vals = students.map(s => s[f.key]);
    out[f.key] = { min: Math.min(...vals), max: Math.max(...vals) };
  });
  return out;
}

/** z = (x - min) / (max - min); max === min bo'lsa z = 0. Diapazondan tashqari qiymat kesilmaydi. */
function minMax(x, min, max) {
  if (!isNum(x) || !isNum(min) || !isNum(max)) return NaN;
  if (max === min) return 0;
  return (x - min) / (max - min);
}
function denorm(z, min, max) { return min + z * (max - min); }
function normalizePoint(p, bounds) {
  return { x: minMax(p.score, bounds.score.min, bounds.score.max), y: minMax(p.att, bounds.att.min, bounds.att.max) };
}
function euclid(a, b) { return Math.sqrt(sqEuclid(a, b)); }
function sqEuclid(a, b) { const dx = a.x - b.x, dy = a.y - b.y; return dx * dx + dy * dy; }

function countClasses(list) {
  const c = {};
  CLASS_NAMES.forEach(n => { c[n] = 0; });
  list.forEach(s => { c[s.cls] += 1; });
  return c;
}
/** Gini G = 1 - sum(p_c^2) */
function giniFromCounts(counts) {
  const n = CLASS_NAMES.reduce((a, k) => a + counts[k], 0);
  if (n === 0) return 0;
  return 1 - CLASS_NAMES.reduce((a, k) => a + Math.pow(counts[k] / n, 2), 0);
}
/** Ko'pchilik sinf; teng bo'lsa CLASS_NAMES tartibidagi birinchisi (deterministik). */
function majorityClass(counts) {
  let best = CLASS_NAMES[0];
  CLASS_NAMES.forEach(k => { if (counts[k] > counts[best]) best = k; });
  return best;
}

/* ---------- KNN ---------- */
function knnClassify(point, students, bounds, k) {
  const kk = k || KNN_K;
  const z = normalizePoint(point, bounds);
  const rows = students.map(s => {
    const zs = normalizePoint(s, bounds);
    return { student: s, z: zs, d: euclid(z, zs) };
  });
  const sorted = rows.slice().sort((a, b) => {
    if (Math.abs(a.d - b.d) > 1e-12) return a.d - b.d;
    return idNum(a.student.id) - idNum(b.student.id);
  });
  sorted.forEach((r, i) => { r.rank = i + 1; r.selected = i < kk; });
  const neighbors = sorted.slice(0, kk);
  const votes = {};
  CLASS_NAMES.forEach(n => { votes[n] = 0; });
  neighbors.forEach(r => { votes[r.student.cls] += 1; });
  let prediction = CLASS_NAMES[0];
  CLASS_NAMES.forEach(n => { if (votes[n] > votes[prediction]) prediction = n; });
  const top = Math.max(...CLASS_NAMES.map(n => votes[n]));
  const tied = CLASS_NAMES.filter(n => votes[n] === top);
  if (tied.length > 1) prediction = neighbors.find(r => tied.indexOf(r.student.cls) >= 0).student.cls;
  return { point, z, rows, sorted, neighbors, votes, prediction, k: kk };
}

/* ---------- Decision Tree (Gini) ---------- */
function evaluateSplit(list, feature, threshold) {
  const left = list.filter(s => s[feature] <= threshold);
  const right = list.filter(s => s[feature] > threshold);
  const cL = countClasses(left), cR = countClasses(right);
  const gL = giniFromCounts(cL), gR = giniFromCounts(cR);
  const n = list.length;
  const weighted = (left.length / n) * gL + (right.length / n) * gR;
  return { feature, threshold, left, right, nL: left.length, nR: right.length, cL, cR, gL, gR, weighted };
}

function buildTree(students, maxDepth) {
  const md = isNum(maxDepth) ? maxDepth : TREE_MAX_DEPTH;
  let counter = 0;
  function grow(list, depth) {
    const counts = countClasses(list);
    const node = {
      id: 'N' + (counter++), depth, n: list.length, counts, gini: giniFromCounts(counts),
      ids: list.map(s => s.id), candidates: [], split: null, left: null, right: null,
      leaf: true, prediction: majorityClass(counts), reason: ''
    };
    const pure = CLASS_NAMES.filter(k => counts[k] > 0).length <= 1;
    if (pure) { node.reason = "Bitta sinf qoldi (toza tugun)"; return node; }
    if (depth >= md) { node.reason = 'maxDepth = ' + md + ' ga yetildi'; return node; }
    FEATURES.forEach(f => {
      const vals = Array.from(new Set(list.map(s => s[f.key]))).sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const t = (vals[i] + vals[i + 1]) / 2;
        const ev = evaluateSplit(list, f.key, t);
        if (ev.nL > 0 && ev.nR > 0) node.candidates.push(ev);
      }
    });
    if (!node.candidates.length) { node.reason = "Yaroqli ajratish topilmadi"; return node; }
    // Eng kichik vaznli Gini; teng bo'lsa: avval ball, keyin qatnashish, so'ng kichikroq chegara
    let best = node.candidates[0];
    node.candidates.forEach(c => {
      if (c.weighted < best.weighted - 1e-12) best = c;
    });
    if (best.weighted >= node.gini - 1e-12) { node.reason = "Impurity yaxshilanmaydi"; return node; }
    node.best = best;
    node.split = { feature: best.feature, threshold: best.threshold, weighted: best.weighted };
    node.leaf = false;
    node.reason = '';
    node.left = grow(best.left, depth + 1);
    node.right = grow(best.right, depth + 1);
    return node;
  }
  return grow(students.slice(), 0);
}
function treeNodes(root) {
  const out = [];
  (function walk(n) { if (!n) return; out.push(n); walk(n.left); walk(n.right); })(root);
  return out;
}
function treePredict(root, point) {
  const path = [root];
  let n = root;
  while (!n.leaf) {
    n = point[n.split.feature] <= n.split.threshold ? n.left : n.right;
    path.push(n);
  }
  return { path, prediction: n.prediction, leaf: n };
}

/* ---------- K-means ---------- */
function kmeansCreate(points, centers) {
  return {
    iteration: 0, centers: centers.map(c => ({ x: c.x, y: c.y })), initialCenters: centers.map(c => ({ x: c.x, y: c.y })),
    assignments: null, records: [], jHistory: [], converged: false, finished: false, stopReason: ''
  };
}
function kmeansInitCenters(points, ids) {
  return ids.map(id => { const p = points.find(q => q.id === id); return { x: p.x, y: p.y }; });
}
/** Bitta to'liq iteratsiya: biriktirish, markazni yangilash, J (yangilangan markazlar bilan). */
function kmeansStep(km, points) {
  if (km.finished) return km;
  const q = km.centers.length;
  const prevCenters = km.centers.map(c => ({ x: c.x, y: c.y }));
  const distances = points.map(p => prevCenters.map(c => sqEuclid(p, c)));
  const assignments = distances.map(row => {
    let b = 0;
    row.forEach((d, j) => { if (d < row[b]) b = j; }); // teng bo'lsa kichik indeks
    return b;
  });
  const centers = prevCenters.map((old, j) => {
    const members = points.filter((p, i) => assignments[i] === j);
    if (!members.length) return { x: old.x, y: old.y };
    return { x: members.reduce((a, p) => a + p.x, 0) / members.length, y: members.reduce((a, p) => a + p.y, 0) / members.length };
  });
  let J = 0;
  points.forEach((p, i) => { J += sqEuclid(p, centers[assignments[i]]); });
  const changed = km.assignments === null ? true : assignments.some((a, i) => a !== km.assignments[i]);
  const iteration = km.iteration + 1;
  const prevJ = km.jHistory.length ? km.jHistory[km.jHistory.length - 1] : null;
  const clusters = [];
  for (let j = 0; j < q; j++) {
    const ids = points.filter((p, i) => assignments[i] === j).map(p => p.id);
    clusters.push({ index: j, ids, count: ids.length });
  }
  const shifts = centers.map((c, j) => euclid(c, prevCenters[j]));
  const converged = !changed;
  const record = {
    iteration, distances, assignments, clusters, prevCenters, centers, shifts, J, prevJ,
    dJ: prevJ === null ? null : J - prevJ, changed, converged
  };
  const finished = converged || iteration >= KMEANS_MAX_ITER;
  return {
    iteration, centers, initialCenters: km.initialCenters, assignments, records: km.records.concat([record]),
    jHistory: km.jHistory.concat([J]), converged, finished,
    stopReason: converged ? 'converged' : (iteration >= KMEANS_MAX_ITER ? 'maxIter' : '')
  };
}
function kmeansRun(km, points) {
  let cur = km;
  while (!cur.finished) cur = kmeansStep(cur, points);
  return cur;
}

/* ==========================================================================
   3. TEKSHIRISH HISOBLARI (asosiy kodga bog'lanmagan, xom ma'lumotdan mustaqil arifmetika)
   ========================================================================== */
const f4 = n => (isNum(n) ? n.toFixed(4) : 'noaniq');
const f10 = n => (isNum(n) ? n.toFixed(10) : 'noaniq');
function esc(v) {
  return String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function verdict(diff) {
  if (!isNum(diff)) return { ok: false, text: 'Hisoblashda farq aniqlandi (qiymat noaniq)' };
  return diff < EPS ? { ok: true, text: 'Mos keldi ✓' } : { ok: false, text: 'Hisoblashda farq aniqlandi: |farq| = ' + diff.toExponential(3) };
}

function manualKnn(point, students) {
  const sMin = Math.min(...students.map(s => s.score)), sMax = Math.max(...students.map(s => s.score));
  const aMin = Math.min(...students.map(s => s.att)), aMax = Math.max(...students.map(s => s.att));
  const nx = (point.score - sMin) / (sMax - sMin), ny = (point.att - aMin) / (aMax - aMin);
  const list = students.map(s => {
    // Mustaqil yo'l: farqlar ayirma diapazonga bo'linadi (Math.hypot)
    const d = Math.hypot((point.score - s.score) / (sMax - sMin), (point.att - s.att) / (aMax - aMin));
    return { s, d };
  }).sort((a, b) => (Math.abs(a.d - b.d) > 1e-12 ? a.d - b.d : idNum(a.s.id) - idNum(b.s.id)));
  const top = list.slice(0, KNN_K);
  const faol = top.filter(r => r.s.cls === 'Faol').length;
  const oddiy = KNN_K - faol;
  return { sMin, sMax, aMin, aMax, nx, ny, top, faol, oddiy, cls: faol > oddiy ? 'Faol' : 'Oddiy' };
}

function manualTreeRoot(students, feature, threshold) {
  const L = students.filter(s => s[feature] <= threshold), R = students.filter(s => s[feature] > threshold);
  const cnt = (arr, c) => arr.filter(s => s.cls === c).length;
  const n = students.length;
  const gl = 1 - Math.pow(cnt(L, 'Oddiy') / L.length, 2) - Math.pow(cnt(L, 'Faol') / L.length, 2);
  const gr = 1 - Math.pow(cnt(R, 'Oddiy') / R.length, 2) - Math.pow(cnt(R, 'Faol') / R.length, 2);
  return {
    nL: L.length, nR: R.length, lO: cnt(L, 'Oddiy'), lF: cnt(L, 'Faol'), rO: cnt(R, 'Oddiy'), rF: cnt(R, 'Faol'),
    gl, gr, w: (L.length / n) * gl + (R.length / n) * gr
  };
}

/** Birinchi iteratsiyaning mustaqil qo'lda hisobi (C1 uchun), xom ma'lumotdan. */
function manualKmeansCluster(students, centerIds, members, clusterIndex) {
  const sMin = Math.min(...students.map(s => s.score)), sMax = Math.max(...students.map(s => s.score));
  const aMin = Math.min(...students.map(s => s.att)), aMax = Math.max(...students.map(s => s.att));
  const nz = s => [(s.score - sMin) / (sMax - sMin), (s.att - aMin) / (aMax - aMin)];
  let list = members;
  if (!list) {
    const cs = centerIds.map(id => nz(students.find(s => s.id === id)));
    list = students.filter(s => {
      const z = nz(s);
      const d = cs.map(c => (z[0] - c[0]) * (z[0] - c[0]) + (z[1] - c[1]) * (z[1] - c[1]));
      return d.indexOf(Math.min(...d)) === clusterIndex;
    }).map(s => s.id);
  }
  const rows = list.map(id => ({ id, z: nz(students.find(s => s.id === id)) }));
  const sumX = rows.reduce((a, r) => a + r.z[0], 0), sumY = rows.reduce((a, r) => a + r.z[1], 0);
  const n = rows.length;
  return { rows, sumX, sumY, n, cx: n ? sumX / n : NaN, cy: n ? sumY / n : NaN };
}

/* ==========================================================================
   Node uchun eksport (DOM kodi himoyalangan)
   ========================================================================== */
/* ==========================================================================
   4. HOLAT (bitta izchil state)
   ========================================================================== */
const BOUNDS = computeBounds(STUDENTS);
const POINTS = STUDENTS.map(s => Object.assign({ id: s.id }, normalizePoint(s, BOUNDS)));
const TREE = buildTree(STUDENTS, TREE_MAX_DEPTH);

function freshKmeans() { return kmeansCreate(POINTS, kmeansInitCenters(POINTS, KMEANS_INIT_IDS)); }

const state = {
  theme: 'dark',
  newStudent: { score: 80, att: 85 },
  knn: null,
  tree: TREE,
  treeResult: null,
  kmeans: freshKmeans(),
  auto: { running: false, timer: null },
  chart: { mode: 'raw', highlightId: null },
  table: { search: '', cls: 'all', cluster: 'all', sortKey: null, sortDir: 1, activeId: null },
  inputError: { score: '', att: '' }
};

function recomputeStudent() {
  state.knn = knnClassify(state.newStudent, STUDENTS, BOUNDS, KNN_K);
  state.treeResult = treePredict(state.tree, state.newStudent);
}
function nearestCenter() {
  const z = normalizePoint(state.newStudent, BOUNDS);
  let b = 0, bd = Infinity;
  state.kmeans.centers.forEach((c, j) => { const d = sqEuclid(z, c); if (d < bd) { bd = d; b = j; } });
  return { index: b, d: Math.sqrt(bd) };
}
const CLUSTER_NAMES = ['C1', 'C2', 'C3'];

/* ==========================================================================
   5. UI RENDER
   ========================================================================== */
const hasDOM = typeof document !== 'undefined' && typeof window !== 'undefined';
const $ = id => (hasDOM ? document.getElementById(id) : null);
const reduceMotion = () => hasDOM && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let listenersBound = false;

function setText(id, text) { const el = $(id); if (el && el.textContent !== text) { el.textContent = text; pop(el); } }
function pop(el) {
  if (reduceMotion()) return;
  el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop');
}
function toast(msg, type) {
  const box = $('toasts');
  if (!box) return;
  const t = document.createElement('div');
  t.className = 'toast ' + (type || 'info');
  t.textContent = msg;
  box.appendChild(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 400); }, 3200);
}
function clsBadge(c) { return '<span class="badge ' + (c === 'Faol' ? 'b-active' : 'b-ordinary') + '">' + (c === 'Faol' ? '▲ ' : '● ') + esc(c) + '</span>'; }
function clusterBadge(j) { return '<span class="badge cl' + (j + 1) + '">' + (j === null || j === undefined ? '—' : '◆ C' + (j + 1)) + '</span>'; }
function clusterOf(id) {
  const a = state.kmeans.assignments;
  if (!a) return null;
  return a[STUDENTS.findIndex(s => s.id === id)];
}

/* ----- Bosh sahifa ----- */
function renderStats() {
  const c = countClasses(STUDENTS), km = state.kmeans;
  setText('statTotal', String(STUDENTS.length));
  setText('statActive', String(c.Faol));
  setText('statOrdinary', String(c.Oddiy));
  setText('statKnn', state.knn.prediction);
  setText('statTree', state.treeResult.prediction);
  setText('statIter', String(km.iteration));
  setText('statJ', km.jHistory.length ? f4(km.jHistory[km.jHistory.length - 1]) : '—');
  const conv = km.converged ? 'Ha (' + km.iteration + '-iteratsiya)' : (km.finished ? 'Yo\'q (100 chegara)' : (state.auto.running ? 'Jarayonda' : 'Yo\'q'));
  setText('statConv', conv);
  setText('statusKnn', 'tayyor');
  setText('statusTree', 'tayyor');
  setText('statusKmeans', km.iteration === 0 ? '0-iteratsiya' : (km.converged ? 'konvergensiya (' + km.iteration + ')' : km.iteration + '-iteratsiya'));
}

function renderInput() {
  const n = state.newStudent;
  const sIn = $('inputScore'), aIn = $('inputAtt');
  if (document.activeElement !== sIn) sIn.value = n.score;
  if (document.activeElement !== aIn) aIn.value = n.att;
  $('rangeScore').value = n.score; $('rangeAtt').value = n.att;
  $('barScore').style.width = Math.max(0, Math.min(100, n.score)) + '%';
  $('barAtt').style.width = Math.max(0, Math.min(100, n.att)) + '%';
  [['score', 'inputScore', 'msgScore'], ['att', 'inputAtt', 'msgAtt']].forEach(([k, inId, msgId]) => {
    const msg = state.inputError[k];
    $(msgId).textContent = msg || 'To\'g\'ri qiymat';
    $(msgId).className = 'field-msg ' + (msg ? 'bad' : 'good');
    $(inId).setAttribute('aria-invalid', msg ? 'true' : 'false');
  });
  const z = state.knn.z, nc = nearestCenter();
  $('currentStudent').innerHTML =
    '<strong>Joriy talaba:</strong> ball = ' + n.score + ', qatnashish = ' + n.att + '% &rarr; normalizatsiya: (' + f4(z.x) + '; ' + f4(z.y) + '). ' +
    'KNN: ' + clsBadge(state.knn.prediction) + ' Daraxt: ' + clsBadge(state.treeResult.prediction);
  $('nearestCenter').innerHTML = 'Eng yaqin klaster markazi (ixtiyoriy ko\'rsatkich, talaba klasterga qo\'shilmaydi): ' +
    clusterBadge(nc.index) + ' &nbsp; masofa = ' + f4(nc.d) + ' (' + (state.kmeans.iteration === 0 ? 'boshlang\'ich markazlar' : state.kmeans.iteration + '-iteratsiya markazlari') + ')';
}

function renderNorm() {
  const n = state.newStudent, z = state.knn.z, b = BOUNDS;
  $('normBox').innerHTML =
    '<table class="mini"><thead><tr><th>Parametr</th><th>Asl qiymat</th><th>min</th><th>max</th><th>Normalizatsiya (4 xona)</th><th>Formula (haqiqiy sonlar bilan)</th></tr></thead><tbody>' +
    '<tr><td>O\'rtacha ball</td><td>' + n.score + '</td><td>' + b.score.min + '</td><td>' + b.score.max + '</td><td class="num">' + f4(z.x) + '</td>' +
    '<td class="num">(' + n.score + ' &minus; ' + b.score.min + ') / (' + b.score.max + ' &minus; ' + b.score.min + ') = ' + f4(z.x) + '</td></tr>' +
    '<tr><td>Qatnashish (%)</td><td>' + n.att + '</td><td>' + b.att.min + '</td><td>' + b.att.max + '</td><td class="num">' + f4(z.y) + '</td>' +
    '<td class="num">(' + n.att + ' &minus; ' + b.att.min + ') / (' + b.att.max + ' &minus; ' + b.att.min + ') = ' + f4(z.y) + '</td></tr></tbody></table>' +
    '<p class="hint">min/max faqat 20 ta o\'quv talabadan olinadi. Ichki hisoblashda to\'liq aniqlik ishlatiladi, 4 xona faqat ekranda ko\'rsatiladi.</p>';
}

/* ----- Talabalar jadvali ----- */
function visibleRows() {
  const t = state.table;
  let rows = STUDENTS.map(s => ({ s, cl: clusterOf(s.id) }));
  const q = t.search.trim().toLowerCase();
  if (q) rows = rows.filter(r => r.s.id.toLowerCase().indexOf(q) >= 0);
  if (t.cls !== 'all') rows = rows.filter(r => r.s.cls === t.cls);
  if (t.cluster !== 'all') rows = rows.filter(r => r.cl !== null && String(r.cl) === t.cluster);
  if (t.sortKey) rows = rows.slice().sort((a, b) => (a.s[t.sortKey] - b.s[t.sortKey]) * t.sortDir || idNum(a.s.id) - idNum(b.s.id));
  return rows;
}
function renderTable() {
  const rows = visibleRows();
  const body = $('tableBody');
  body.innerHTML = rows.map(r => {
    const s = r.s, z = normalizePoint(s, BOUNDS);
    return '<tr data-id="' + s.id + '" tabindex="0" class="' + (state.table.activeId === s.id ? 'active' : '') + '">' +
      '<td><span class="cell"><b>' + s.id + '</b></span></td><td><span class="cell num">' + s.score + '</span></td><td><span class="cell num">' + s.att + '</span></td>' +
      '<td><span class="cell">' + clsBadge(s.cls) + '</span></td><td><span class="cell num">' + f4(z.x) + '</span></td><td><span class="cell num">' + f4(z.y) + '</span></td>' +
      '<td><span class="cell">' + (r.cl === null ? '<span class="muted">hali yo\'q</span>' : clusterBadge(r.cl)) + '</span></td></tr>';
  }).join('') || '<tr><td colspan="7" class="empty">Shartga mos talaba topilmadi.</td></tr>';
  setText('tableCount', rows.length + ' / ' + STUDENTS.length + ' talaba ko\'rsatilmoqda (o\'quv ma\'lumotlari 20 ta o\'zgarmaydi)');
  const dirTxt = k => (state.table.sortKey === k ? (state.table.sortDir === 1 ? ' ↑' : ' ↓') : '');
  $('sortScore').textContent = 'Ball bo\'yicha saralash' + dirTxt('score');
  $('sortAtt').textContent = 'Qatnashish bo\'yicha saralash' + dirTxt('att');
}
function tableText(sep) {
  const head = ['ID', "O'rtacha ball", 'Qatnashish (%)', 'Sinf', 'Norm. ball', 'Norm. qatnashish', 'Joriy klaster'];
  const lines = [head.join(sep)];
  visibleRows().forEach(r => {
    const z = normalizePoint(r.s, BOUNDS);
    lines.push([r.s.id, r.s.score, r.s.att, r.s.cls, z.x, z.y, r.cl === null ? '' : 'C' + (r.cl + 1)].join(sep));
  });
  return lines.join('\n');
}

/* ----- KNN ----- */
function renderKnn() {
  const k = state.knn, n = state.newStudent;
  $('knnResult').innerHTML =
    '<div class="big-result ' + (k.prediction === 'Faol' ? 'is-active' : 'is-ordinary') + '"><span class="label">KNN bashorati (k = ' + k.k + ')</span><strong>' + esc(k.prediction) + '</strong>' +
    '<span class="votes">Ovozlar: Faol = ' + k.votes.Faol + ', Oddiy = ' + k.votes.Oddiy + '</span></div>';
  $('knnNeighbors').innerHTML = k.neighbors.map((r, i) =>
    '<article class="neigh" style="--i:' + i + '"><header><span class="rank">' + (i + 1) + '-qo\'shni</span><b>' + r.student.id + '</b></header>' +
    '<dl><dt>Ball</dt><dd>' + r.student.score + '</dd><dt>Qatnashish</dt><dd>' + r.student.att + '%</dd><dt>Sinf</dt><dd>' + clsBadge(r.student.cls) + '</dd><dt>Masofa</dt><dd class="num">' + f4(r.d) + '</dd></dl></article>').join('');
  $('knnTableBody').innerHTML = k.sorted.map(r =>
    '<tr class="' + (r.selected ? 'picked' : '') + '"><td><span class="cell num">' + r.rank + '</span></td><td><span class="cell"><b>' + r.student.id + '</b></span></td><td><span class="cell num">' + r.student.score + '</span></td><td><span class="cell num">' + r.student.att + '</span></td>' +
    '<td><span class="cell">' + clsBadge(r.student.cls) + '</span></td><td><span class="cell num">' + f4(r.z.x) + '; ' + f4(r.z.y) + '</span></td><td><span class="cell num">' + f4(r.d) + '</span></td><td><span class="cell">' + (r.selected ? '★ tanlandi' : '') + '</span></td></tr>').join('');
  const lines = k.neighbors.map(r => r.student.id + ': d = &radic;((' + f4(k.z.x) + ' &minus; ' + f4(r.z.x) + ')<sup>2</sup> + (' + f4(k.z.y) + ' &minus; ' + f4(r.z.y) + ')<sup>2</sup>) = ' + f4(r.d));
  $('knnExplainSteps').innerHTML =
    '<ol><li>Yangi talaba (' + n.score + '; ' + n.att + ') normalizatsiya qilindi: (' + f4(k.z.x) + '; ' + f4(k.z.y) + ').</li>' +
    '<li>T1&ndash;T20 gacha 20 ta Evklid masofasi hisoblandi va o\'sish tartibida saralandi (teng bo\'lsa ID bo\'yicha).</li>' +
    '<li>Eng yaqin 3 talaba:<br>' + lines.join('<br>') + '</li>' +
    '<li>Sinflar: ' + k.neighbors.map(r => r.student.cls).join(', ') + '. Ovozlar: Faol = ' + k.votes.Faol + ', Oddiy = ' + k.votes.Oddiy + '.</li>' +
    '<li>Ko\'pchilik ovoz &rarr; <b>' + k.prediction + '</b>.</li></ol>';
}

/* ----- Qaror daraxti (SVG) ----- */
const NODE_W = 200, NODE_H = 104, GAP_X = 28, GAP_Y = 70;
function layoutTree(root) {
  let leafIdx = 0;
  (function place(n) {
    n.y = n.depth * (NODE_H + GAP_Y) + 10;
    if (n.leaf) { n.x = leafIdx * (NODE_W + GAP_X) + 10; leafIdx += 1; return; }
    place(n.left); place(n.right);
    n.x = (n.left.x + n.right.x) / 2;
  })(root);
  return { w: leafIdx * (NODE_W + GAP_X) - GAP_X + 20, h: (TREE_MAX_DEPTH + 1) * (NODE_H + GAP_Y) - GAP_Y + 20 };
}
function nodeTitle(n) {
  if (n.leaf) return 'Barg: ' + n.prediction;
  const f = FEATURES.find(x => x.key === n.split.feature);
  return f.short + ' <= ' + n.split.threshold;
}
function renderTree() {
  const root = state.tree, size = layoutTree(root), path = state.treeResult.path.map(n => n.id);
  const nodes = treeNodes(root);
  let svg = '<svg id="treeSvg" role="img" aria-label="Qaror daraxti sxemasi" viewBox="0 0 ' + size.w + ' ' + size.h + '" width="' + size.w + '" height="' + size.h + '">' +
    '<defs><linearGradient id="nodeGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="var(--node-a)"/><stop offset="1" stop-color="var(--node-b)"/></linearGradient>' +
    '<filter id="nodeShadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#000" flood-opacity="0.35"/></filter>' +
    '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--edge)"/></marker>' +
    '<marker id="arrowOn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--neon-green)"/></marker></defs>';
  nodes.forEach(n => {
    if (n.leaf) return;
    [['left', 'ha: <= ' + n.split.threshold], ['right', "yo'q: > " + n.split.threshold]].forEach(([side, label]) => {
      const c = n[side];
      const x1 = n.x + NODE_W / 2, y1 = n.y + NODE_H, x2 = c.x + NODE_W / 2, y2 = c.y - 2, my = (y1 + y2) / 2;
      const d = 'M' + x1 + ' ' + y1 + ' C' + x1 + ' ' + my + ' ' + x2 + ' ' + my + ' ' + x2 + ' ' + y2;
      const on = path.indexOf(n.id) >= 0 && path.indexOf(c.id) >= 0;
      svg += '<path class="edge' + (on ? ' on' : (path.length ? ' dim' : '')) + '" pathLength="1" d="' + d + '" marker-end="url(#' + (on ? 'arrowOn' : 'arrow') + ')"/>';
      if (on) svg += '<path class="edge-flow" pathLength="1" d="' + d + '"/>';
      svg += '<text class="edge-label' + (on ? ' on' : '') + '" x="' + ((x1 + x2) / 2 + (side === 'left' ? -8 : 8)) + '" y="' + (my - 4) + '" text-anchor="' + (side === 'left' ? 'end' : 'start') + '">' + esc(label) + '</text>';
    });
  });
  nodes.forEach((n, i) => {
    const on = path.indexOf(n.id) >= 0;
    const fcls = n.leaf ? (n.prediction === 'Faol' ? ' leaf-active' : ' leaf-ordinary') : '';
    svg += '<g class="tnode' + fcls + (on ? ' on' : ' dim') + '" style="animation-delay:' + (i * 0.18) + 's" transform="translate(' + n.x + ' ' + n.y + ')">' +
      '<rect width="' + NODE_W + '" height="' + NODE_H + '" rx="16" fill="url(#nodeGrad)" filter="url(#nodeShadow)" class="tbox"/>' +
      '<text class="t1" x="' + NODE_W / 2 + '" y="26" text-anchor="middle">' + esc(nodeTitle(n)) + '</text>' +
      '<text class="t2" x="' + NODE_W / 2 + '" y="46" text-anchor="middle">n = ' + n.n + '  ·  Gini = ' + f4(n.gini) + '</text>';
    const total = n.n || 1, wO = NODE_W - 40, oddW = wO * n.counts.Oddiy / total;
    svg += '<rect x="20" y="56" width="' + oddW + '" height="12" rx="3" fill="var(--c-ordinary)"/><rect x="' + (20 + oddW) + '" y="56" width="' + (wO - oddW) + '" height="12" rx="3" fill="var(--c-active)"/>' +
      '<text class="t3" x="' + NODE_W / 2 + '" y="86" text-anchor="middle">Oddiy: ' + n.counts.Oddiy + ' | Faol: ' + n.counts.Faol + '</text>' +
      (n.leaf ? '<text class="t4" x="' + NODE_W / 2 + '" y="100" text-anchor="middle">Bashorat: ' + n.prediction + '</text>' : '') + '</g>';
  });
  svg += '</svg>';
  $('treeWrap').innerHTML = svg;
  const pathTxt = state.treeResult.path.map(n => n.leaf ? 'Barg (' + n.prediction + ')' : nodeTitle(n)).join('  →  ');
  $('treePathText').innerHTML = '<b>Yangi talaba yo\'li:</b> ' + esc(pathTxt) + ' &nbsp; ' + clsBadge(state.treeResult.prediction);
  const r = state.tree, b = r.best;
  $('treeInfo').innerHTML = 'Ildiz: <b>' + esc(nodeTitle(r)) + '</b>, vaznli Gini = <b>' + f10(b.weighted) + '</b>. O\'ng tarmoq: <b>' + esc(nodeTitle(r.right)) + '</b>. To\'xtash qoidalari: bir sinf, maxDepth = ' + TREE_MAX_DEPTH + ', yaroqli ajratish yo\'q, impurity yaxshilanmasligi.';
  $('treeCandBody').innerHTML = r.candidates.map(c => {
    const best = c === b;
    return '<tr class="' + (best ? 'picked' : '') + '"><td><span class="cell">' + esc(FEATURES.find(f => f.key === c.feature).short) + '</span></td><td><span class="cell num">' + c.threshold + '</span></td>' +
      '<td><span class="cell num">' + c.nL + ' / ' + c.nR + '</span></td><td><span class="cell num">O ' + c.cL.Oddiy + ', F ' + c.cL.Faol + '</span></td><td><span class="cell num">O ' + c.cR.Oddiy + ', F ' + c.cR.Faol + '</span></td>' +
      '<td><span class="cell num">' + f4(c.gL) + '</span></td><td><span class="cell num">' + f4(c.gR) + '</span></td><td><span class="cell num">' + f4(c.weighted) + '</span></td><td><span class="cell">' + (best ? '★ eng yaxshi' : '') + '</span></td></tr>';
  }).join('');
}

/* ----- K-means ----- */
function centerRaw(c) { return { score: denorm(c.x, BOUNDS.score.min, BOUNDS.score.max), att: denorm(c.y, BOUNDS.att.min, BOUNDS.att.max) }; }
function pt(c) { return '(' + f4(c.x) + '; ' + f4(c.y) + ')'; }
function clusterList(rec) {
  return rec.clusters.map(c => '<div class="cl-card cl' + (c.index + 1) + '"><h4>' + clusterBadge(c.index) + ' <span>' + c.count + ' ta</span></h4><p>' + (c.ids.join(', ') || 'bo\'sh klaster') + '</p></div>').join('');
}
function renderKmeans() {
  const km = state.kmeans;
  setText('kmIter', String(km.iteration));
  const btn = (id, dis) => { $(id).disabled = dis; };
  btn('btnStep', km.finished || state.auto.running);
  btn('btnAuto', km.finished || state.auto.running);
  btn('btnStop', !state.auto.running);
  btn('btnReset', false);
  const statusTxt = state.auto.running ? 'Avtomatik yakunlash jarayonda...' :
    (km.converged ? 'Konvergensiyaga erishildi: ' + km.iteration + '-iteratsiyada biriktirishlar o\'zgarmadi.' :
      (km.finished ? '100-iteratsiya chegarasiga yetildi.' : (km.iteration === 0 ? '0-iteratsiya: faqat boshlang\'ich markazlar tanlangan.' : 'Davom ettirish mumkin.')));
  $('kmStatus').textContent = statusTxt;
  $('kmInit').innerHTML = '<h4>Boshlang\'ich markazlar (0-iteratsiya)</h4><div class="cl-grid">' + km.initialCenters.map((c, j) => {
    const r = centerRaw(c);
    return '<div class="cl-card cl' + (j + 1) + '"><h4>' + clusterBadge(j) + ' = ' + KMEANS_INIT_IDS[j] + '</h4><p>Asl: (' + r.score + '; ' + r.att + ')<br>Norm.: ' + pt(c) + '</p></div>';
  }).join('') + '</div><p class="hint">Tanlash mantig\'i: C1 = T1 (eng past ball), C2 = T15 (o\'rta), C3 = T20 (eng yuqori ball). Markazlar sinf yorlig\'iga emas, koordinatalarga asoslanadi. Keyingi bosqichlar: biriktirish &rarr; markazni yangilash &rarr; J ni hisoblash &rarr; o\'zgarishni tekshirish.</p>';
  const rec = km.records[km.records.length - 1];
  if (!rec) {
    $('kmLatest').innerHTML = '<p class="muted">Hali iteratsiya bajarilmagan. «Keyingi qadam» yoki «Avtomatik yakunlash» tugmasini bosing. J qiymati birinchi iteratsiyadan keyin hisoblanadi.</p>';
    $('kmHistory').innerHTML = '';
  } else {
    $('kmLatest').innerHTML = iterHtml(rec);
    $('kmHistory').innerHTML = km.records.slice(0, -1).reverse().map(r => '<details><summary>' + r.iteration + '-iteratsiya: J = ' + f4(r.J) + ', klasterlar ' + r.clusters.map(c => c.count).join('/') + '</summary>' + iterHtml(r) + '</details>').join('');
  }
  $('kmDemo').innerHTML = demoHtml();
  drawJChart();
}
function iterHtml(rec) {
  const dJ = rec.dJ === null ? 'birinchi iteratsiya (oldingi J yo\'q)' : (rec.dJ <= 1e-12 ? '' : '+') + rec.dJ.toFixed(10);
  const rows = rec.distances.map((d, i) => '<tr><td>' + STUDENTS[i].id + '</td>' + d.map(v => '<td class="num">' + f4(v) + '</td>').join('') + '<td>' + clusterBadge(rec.assignments[i]) + '</td></tr>').join('');
  return '<h4>' + rec.iteration + '-iteratsiya (bajarilgan)</h4><div class="cl-grid">' + clusterList(rec) + '</div>' +
    '<table class="mini"><thead><tr><th>Markaz</th><th>Oldingi markaz</th><th>Yangi markaz</th><th>Siljish</th></tr></thead><tbody>' +
    rec.centers.map((c, j) => '<tr><td>' + clusterBadge(j) + '</td><td class="num">' + pt(rec.prevCenters[j]) + '</td><td class="num">' + pt(c) + '</td><td class="num">' + f4(rec.shifts[j]) + '</td></tr>').join('') + '</tbody></table>' +
    '<p>J = <b>' + f10(rec.J) + '</b> (4 xonada: ' + f4(rec.J) + '). Oldingi J dan farq: ' + dJ + '. Biriktirishlar o\'zgardi: <b>' + (rec.changed ? 'ha' : 'yo\'q') + '</b>. Konvergensiya: <b>' + (rec.converged ? 'ha' : 'yo\'q') + '</b>.</p>' +
    '<details><summary>20 talaba uchun kvadrat masofalar jadvali</summary><div class="table-scroll"><table class="mini"><thead><tr><th>ID</th><th>d&sup2; C1</th><th>d&sup2; C2</th><th>d&sup2; C3</th><th>Eng yaqin</th></tr></thead><tbody>' + rows + '</tbody></table></div></details>';
}
function demoHtml() {
  if (state.kmeans.iteration > 0) return '';
  const demo = kmeansStep(freshKmeans(), POINTS), r = demo.records[0];
  return '<div class="demo-box"><span class="tag">Namoyish, bajarilgan iteratsiya emas</span><p>Agar «Keyingi qadam» bosilsa, 1-iteratsiyada klasterlar soni: ' + r.clusters.map((c, j) => 'C' + (j + 1) + ' = ' + c.count).join(', ') + '; J = ' + f4(r.J) + '. Bu qiymatlar holatga yozilmagan.</p></div>';
}
function drawJChart(progress) {
  const cv = $('jChart');
  if (!cv) return;
  const hist = state.kmeans.jHistory;
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 300, h = 150;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const col = cssVars();
  ctx.clearRect(0, 0, w, h);
  ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = col.muted; ctx.strokeStyle = col.grid;
  ctx.strokeRect(34.5, 10.5, w - 44, h - 36);
  if (!hist.length) { ctx.fillText('J grafigi: iteratsiyalardan keyin to\'ladi', 44, h / 2); return; }
  const p = progress === undefined ? 1 : progress;
  const max = Math.max(...hist), min = Math.min(...hist), span = max - min || max || 1;
  const X = i => 34 + (hist.length === 1 ? (w - 44) / 2 : (i / (hist.length - 1)) * (w - 44));
  const Y = v => 10 + (h - 36) - ((v - min + span * 0.1) / (span * 1.2)) * (h - 36);
  ctx.beginPath();
  hist.forEach((v, i) => { const x = X(i), y = Y(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, 34 + (w - 44) * p + 6, h); ctx.clip();
  ctx.strokeStyle = col.cyan; ctx.lineWidth = 2.5; ctx.stroke();
  hist.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(i), Y(v), 4, 0, Math.PI * 2); ctx.fillStyle = col.cyan; ctx.fill(); });
  ctx.restore();
  ctx.fillStyle = col.muted;
  ctx.fillText(f4(max), 2, 20); ctx.fillText(f4(min), 2, h - 30);
  ctx.fillText('iteratsiya: 1 … ' + hist.length, 40, h - 8);
  if (p < 1 && !reduceMotion()) requestAnimationFrame(() => drawJChart(Math.min(1, p + 0.08)));
}

/* ----- Interaktiv grafik (Canvas) ----- */
const chart = { cv: null, ctx: null, w: 600, h: 400, dpr: 1, hits: [], shown: null, visible: true, raf: 0, lastHit: null };
function cssVars() {
  const cs = getComputedStyle(document.documentElement), g = n => cs.getPropertyValue(n).trim();
  return {
    text: g('--text'), muted: g('--muted'), grid: g('--grid'), bg: g('--card-solid'), cyan: g('--cyan'), blue: g('--blue'), green: g('--green'),
    amber: g('--amber'), purple: g('--purple'), neon: g('--neon-green'), cl: [g('--cyan'), g('--amber'), g('--purple')]
  };
}
function niceStep(range, ticks) {
  const raw = range / ticks, pow = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / pow;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * pow;
}
function chartRange(mode) {
  const norm = mode === 'norm', lo0 = norm ? 0 : 0, hi0 = norm ? 1 : 100;
  const ns = state.knn.z, vals = { x: [], y: [] };
  if (norm) {
    POINTS.forEach(p => { vals.x.push(p.x); vals.y.push(p.y); });
    vals.x.push(ns.x); vals.y.push(ns.y);
    state.kmeans.centers.forEach(c => { vals.x.push(c.x); vals.y.push(c.y); });
  } else {
    STUDENTS.forEach(s => { vals.x.push(s.score); vals.y.push(s.att); });
    vals.x.push(state.newStudent.score); vals.y.push(state.newStudent.att);
    state.kmeans.centers.forEach(c => { const r = centerRaw(c); vals.x.push(r.score); vals.y.push(r.att); });
  }
  const pad = norm ? 0.04 : 4;
  const rng = a => [Math.min(lo0, ...a) - (Math.min(...a) < lo0 ? pad : 0), Math.max(hi0, ...a) + (Math.max(...a) > hi0 ? pad : 0)];
  const rx = rng(vals.x), ry = rng(vals.y);
  const ex = norm ? 0.03 : 2;
  return { x0: rx[0] - ex, x1: rx[1] + ex, y0: ry[0] - ex, y1: ry[1] + ex };
}
function resizeChart() {
  const cv = chart.cv, wrap = cv.parentElement;
  const w = Math.max(280, wrap.clientWidth), h = Math.max(300, Math.min(560, Math.round(w * (w < 600 ? 0.95 : 0.6))));
  chart.dpr = window.devicePixelRatio || 1; chart.w = w; chart.h = h;
  cv.style.height = h + 'px';
  cv.width = Math.round(w * chart.dpr); cv.height = Math.round(h * chart.dpr);
  chart.ctx.setTransform(chart.dpr, 0, 0, chart.dpr, 0, 0);
}
function drawShape(ctx, kind, x, y, r) {
  ctx.beginPath();
  if (kind === 'circle') ctx.arc(x, y, r, 0, Math.PI * 2);
  else if (kind === 'tri') { ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 1.1, y + r * 0.85); ctx.lineTo(x - r * 1.1, y + r * 0.85); ctx.closePath(); }
  else if (kind === 'diamond') { ctx.moveTo(x, y - r * 1.5); ctx.lineTo(x + r * 1.5, y); ctx.lineTo(x, y + r * 1.5); ctx.lineTo(x - r * 1.5, y); ctx.closePath(); }
  else if (kind === 'star') {
    for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.55 : r * 1.5; ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
    ctx.closePath();
  }
}
function drawChart(ts) {
  const ctx = chart.ctx; if (!ctx) return;
  const col = cssVars(), mode = state.chart.mode, norm = mode === 'norm';
  const R = chartRange(mode), W = chart.w, H = chart.h, L = 52, T = 14, Rr = 14, B = 44;
  const X = v => L + ((v - R.x0) / (R.x1 - R.x0)) * (W - L - Rr);
  const Y = v => H - B - ((v - R.y0) / (R.y1 - R.y0)) * (H - T - B);
  const sx = p => X(norm ? p.x : p.score), sy = p => Y(norm ? p.y : p.att);
  ctx.clearRect(0, 0, W, H);
  ctx.font = '11px system-ui, sans-serif';
  const stepX = niceStep(R.x1 - R.x0, 8), stepY = niceStep(R.y1 - R.y0, 6);
  ctx.lineWidth = 1; ctx.textBaseline = 'middle';
  const dec = norm ? 2 : 0;
  for (let v = Math.ceil(R.x0 / stepX) * stepX; v <= R.x1 + 1e-9; v += stepX) {
    ctx.strokeStyle = col.grid; ctx.beginPath(); ctx.moveTo(X(v) + 0.5, T); ctx.lineTo(X(v) + 0.5, H - B); ctx.stroke();
    ctx.fillStyle = col.muted; ctx.textAlign = 'center'; ctx.fillText(v.toFixed(dec), X(v), H - B + 14);
  }
  for (let v = Math.ceil(R.y0 / stepY) * stepY; v <= R.y1 + 1e-9; v += stepY) {
    ctx.strokeStyle = col.grid; ctx.beginPath(); ctx.moveTo(L, Y(v) + 0.5); ctx.lineTo(W - Rr, Y(v) + 0.5); ctx.stroke();
    ctx.fillStyle = col.muted; ctx.textAlign = 'right'; ctx.fillText(v.toFixed(dec), L - 6, Y(v));
  }
  ctx.strokeStyle = col.muted; ctx.strokeRect(L + 0.5, T + 0.5, W - L - Rr, H - T - B);
  ctx.fillStyle = col.text; ctx.textAlign = 'center';
  ctx.fillText(norm ? "X: O'rtacha ball (normalizatsiyalangan)" : "X: O'rtacha ball", L + (W - L - Rr) / 2, H - 12);
  ctx.save(); ctx.translate(13, T + (H - T - B) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(norm ? 'Y: Qatnashish (normalizatsiyalangan)' : 'Y: Qatnashish (%)', 0, 0); ctx.restore();

  const hits = [];
  const asg = state.kmeans.assignments;
  const neighborIds = state.knn.neighbors.map(r => r.student.id);
  const ns = state.newStudent, nsz = state.knn.z;
  const nsP = norm ? { score: nsz.x, att: nsz.y } : ns;
  const nsx = sx(norm ? { x: nsz.x, y: nsz.y } : ns), nsy = sy(norm ? { x: nsz.x, y: nsz.y } : ns);
  // Yangi talaba va qo'shnilar orasidagi animatsiyali punktir chiziqlar
  ctx.save(); ctx.setLineDash([6, 5]); ctx.lineDashOffset = reduceMotion() ? 0 : -((ts || 0) / 45) % 11;
  ctx.strokeStyle = col.neon; ctx.lineWidth = 1.6; ctx.globalAlpha = 0.9;
  state.knn.neighbors.forEach(r => {
    const st = r.student, p = norm ? { x: r.z.x, y: r.z.y } : st;
    ctx.beginPath(); ctx.moveTo(nsx, nsy); ctx.lineTo(sx(p), sy(p)); ctx.stroke();
  });
  ctx.restore();
  STUDENTS.forEach((s, i) => {
    const p = norm ? POINTS[i] : s, x = sx(p), y = sy(p);
    const cl = asg ? asg[i] : null, isNb = neighborIds.indexOf(s.id) >= 0, hi = state.chart.highlightId === s.id;
    const edge = s.cls === 'Faol' ? col.green : col.blue;
    if (isNb || hi) {
      ctx.beginPath(); ctx.arc(x, y, hi ? 17 : 14, 0, Math.PI * 2); ctx.strokeStyle = hi ? col.amber : col.neon; ctx.lineWidth = 2.5; ctx.stroke();
    }
    drawShape(ctx, s.cls === 'Faol' ? 'tri' : 'circle', x, y, 6.5);
    ctx.fillStyle = cl === null ? col.bg : col.cl[cl]; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = edge; ctx.stroke();
    ctx.fillStyle = col.text; ctx.textAlign = 'left'; ctx.fillText(s.id, x + 9, y - 9);
    const z = POINTS[i];
    hits.push({ x, y, r: 13, html: '<b>' + s.id + '</b><br>Ball: ' + s.score + '<br>Qatnashish: ' + s.att + '%<br>Sinf: ' + s.cls + '<br>Klaster: ' + (cl === null ? 'hali yo\'q' : 'C' + (cl + 1)) + '<br>Norm.: (' + f4(z.x) + '; ' + f4(z.y) + ')' });
  });
  // Markazlar (animatsiyali siljish)
  const sh = chart.shown;
  state.kmeans.centers.forEach((c, j) => {
    const cc = sh && sh[j] ? sh[j] : c, p = norm ? cc : centerRaw(cc), x = sx(norm ? cc : p), y = sy(norm ? cc : p);
    drawShape(ctx, 'diamond', x, y, 7); ctx.fillStyle = col.cl[j]; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = col.text; ctx.stroke();
    ctx.fillStyle = col.text; ctx.font = 'bold 12px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.fillText('C' + (j + 1), x + 12, y + 2); ctx.font = '11px system-ui, sans-serif';
    hits.push({ x, y, r: 14, html: '<b>Markaz C' + (j + 1) + '</b><br>Norm.: (' + f4(c.x) + '; ' + f4(c.y) + ')<br>Asl: (' + centerRaw(c).score.toFixed(2) + '; ' + centerRaw(c).att.toFixed(2) + ')' });
  });
  // Yangi talaba (yulduzcha)
  drawShape(ctx, 'star', nsx, nsy, 8); ctx.fillStyle = col.neon; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = col.text; ctx.stroke();
  ctx.fillStyle = col.text; ctx.textAlign = 'left'; ctx.fillText('Yangi', nsx + 12, nsy + 12);
  hits.push({ x: nsx, y: nsy, r: 16, html: '<b>Yangi talaba</b><br>Ball: ' + ns.score + '<br>Qatnashish: ' + ns.att + '%<br>KNN: ' + state.knn.prediction + '<br>Daraxt: ' + state.treeResult.prediction + '<br>Norm.: (' + f4(nsz.x) + '; ' + f4(nsz.y) + ')' });
  void nsP;
  chart.hits = hits;
}
function stepChartAnim(ts) {
  chart.raf = 0;
  if (!chart.ctx || document.hidden || !chart.visible) return;
  const targets = state.kmeans.centers;
  if (!chart.shown) chart.shown = targets.map(c => ({ x: c.x, y: c.y }));
  let moving = false;
  chart.shown.forEach((s, j) => {
    const t = targets[j], dx = t.x - s.x, dy = t.y - s.y;
    if (reduceMotion() || (Math.abs(dx) < 0.0005 && Math.abs(dy) < 0.0005)) { s.x = t.x; s.y = t.y; } else { s.x += dx * 0.14; s.y += dy * 0.14; moving = true; }
  });
  drawChart(ts);
  if (moving || !reduceMotion()) chart.raf = requestAnimationFrame(stepChartAnim);
}
function requestChart() {
  if (!chart.ctx) return;
  if (!chart.raf) chart.raf = requestAnimationFrame(stepChartAnim);
}
function renderChartText() {
  const k = state.knn, ns = state.newStudent, km = state.kmeans;
  $('chartText').innerHTML = 'Rejim: <b>' + (state.chart.mode === 'raw' ? 'Asl qiymatlar (0–100)' : 'Normalizatsiyalangan') + '</b>. Yangi talaba (' + ns.score + '; ' + ns.att + '): KNN bashorati <b>' + k.prediction + '</b>, eng yaqin 3 qo\'shni: ' + k.neighbors.map(r => r.student.id + ' (' + f4(r.d) + ')').join(', ') +
    '. Daraxt bashorati: <b>' + state.treeResult.prediction + '</b>. K-means: ' + km.iteration + '-iteratsiya' + (km.assignments ? ', klasterlar ' + km.records[km.records.length - 1].clusters.map((c, j) => 'C' + (j + 1) + '=' + c.count).join(', ') : ', klasterlar hali yo\'q') + '.';
  document.querySelectorAll('[data-mode]').forEach(b => { b.setAttribute('aria-pressed', String(b.dataset.mode === state.chart.mode)); });
}
function showTip(hit, ev) {
  const tip = $('tooltip'), wrap = chart.cv.parentElement.getBoundingClientRect();
  if (!hit) { tip.hidden = true; return; }
  tip.innerHTML = hit.html; tip.hidden = false;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let x = hit.x + 14, y = hit.y - th / 2;
  if (x + tw > wrap.width) x = hit.x - tw - 14;
  tip.style.left = Math.max(4, x) + 'px'; tip.style.top = Math.max(4, Math.min(y, chart.h - th - 4)) + 'px';
  void ev;
}
function hitAt(ev) {
  const r = chart.cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  let best = null, bd = Infinity;
  chart.hits.forEach(h => { const d = Math.hypot(h.x - x, h.y - y); if (d <= h.r && d < bd) { bd = d; best = h; } });
  return best;
}

/* ----- Tekshirish bo'limi ----- */
function buildVerification() {
  const out = {}, k = state.knn, n = state.newStudent;
  // 18.1 KNN
  const m = manualKnn(n, STUDENTS);
  const idsProg = k.neighbors.map(r => r.student.id).join(', '), idsMan = m.top.map(r => r.s.id).join(', ');
  let maxDiff = Math.max(Math.abs(m.nx - k.z.x), Math.abs(m.ny - k.z.y));
  m.top.forEach((r, i) => { maxDiff = Math.max(maxDiff, Math.abs(r.d - k.neighbors[i].d)); });
  const kv = idsProg === idsMan && m.cls === k.prediction ? verdict(maxDiff) : { ok: false, text: 'Hisoblashda farq aniqlandi: qo\'shnilar yoki sinf mos emas' };
  const zs = k.z;
  const kLines = [
    'Yangi talaba: ball = ' + n.score + ', qatnashish = ' + n.att,
    "Ball: min = " + m.sMin + ', max = ' + m.sMax + '; qatnashish: min = ' + m.aMin + ', max = ' + m.aMax,
    'x = (' + n.score + ' - ' + m.sMin + ') / (' + m.sMax + ' - ' + m.sMin + ') = ' + f10(m.nx),
    'y = (' + n.att + ' - ' + m.aMin + ') / (' + m.aMax + ' - ' + m.aMin + ') = ' + f10(m.ny)
  ];
  m.top.forEach((r, i) => {
    const q = k.neighbors[i];
    kLines.push((i + 1) + '-qo\'shni ' + r.s.id + ' (ball ' + r.s.score + ', qatnashish ' + r.s.att + ', ' + r.s.cls + '): norm. (' + f4(q.z.x) + '; ' + f4(q.z.y) + ')');
    kLines.push('   d = sqrt((' + f4(zs.x) + ' - ' + f4(q.z.x) + ')^2 + (' + f4(zs.y) + ' - ' + f4(q.z.y) + ')^2) = ' + f10(r.d));
  });
  kLines.push('Ovozlar: Faol = ' + m.faol + ', Oddiy = ' + m.oddiy);
  out.knn = {
    title: '18.1 KNN (k = 3)', lines: kLines, verdict: kv,
    manual: 'Qo\'shnilar: ' + idsMan + '; sinf: ' + m.cls + '; masofalar: ' + m.top.map(r => f10(r.d)).join(', '),
    program: 'Qo\'shnilar: ' + idsProg + '; sinf: ' + k.prediction + '; masofalar: ' + k.neighbors.map(r => f10(r.d)).join(', ')
  };
  // 18.2 Daraxt
  const root = state.tree, b = root.best;
  const mt = manualTreeRoot(STUDENTS, b.feature, b.threshold);
  const td = Math.max(Math.abs(mt.w - b.weighted), Math.abs(mt.gl - b.gL), Math.abs(mt.gr - b.gR));
  const tv = mt.nL === b.nL && mt.nR === b.nR && mt.lO === b.cL.Oddiy && mt.lF === b.cL.Faol && mt.rO === b.cR.Oddiy && mt.rF === b.cR.Faol ? verdict(td) : { ok: false, text: 'Hisoblashda farq aniqlandi: guruh tarkibi mos emas' };
  const fn = FEATURES.find(f => f.key === b.feature).label;
  out.tree = {
    title: '18.2 Qaror daraxti (ildiz)', verdict: tv,
    lines: [
      'Eng yaxshi nomzod: parametr = ' + fn + ', chegara = ' + b.threshold,
      'nL = ' + mt.nL + ', nR = ' + mt.nR + ' (n = ' + STUDENTS.length + ')',
      'Chap: Oddiy = ' + mt.lO + ', Faol = ' + mt.lF + '; o\'ng: Oddiy = ' + mt.rO + ', Faol = ' + mt.rF,
      'G_L = 1 - (' + mt.lO + '/' + mt.nL + ')^2 - (' + mt.lF + '/' + mt.nL + ')^2 = ' + f10(mt.gl),
      'G_R = 1 - (' + mt.rO + '/' + mt.nR + ')^2 - (' + mt.rF + '/' + mt.nR + ')^2 = ' + f10(mt.gr),
      'G_split = (' + mt.nL + '/' + STUDENTS.length + ') * ' + f10(mt.gl) + ' + (' + mt.nR + '/' + STUDENTS.length + ') * ' + f10(mt.gr) + ' = ' + f10(mt.w)
    ],
    manual: 'G_split = ' + f10(mt.w), program: 'G_split = ' + f10(b.weighted)
  };
  // 18.3 K-means
  const km = state.kmeans, rec = km.records[km.records.length - 1];
  let mk, prog, label, kmv;
  if (rec) {
    mk = manualKmeansCluster(STUDENTS, KMEANS_INIT_IDS, rec.clusters[0].ids, 0);
    label = km.iteration + '-iteratsiya, C1 (bajarilgan iteratsiya)';
    prog = { x: rec.centers[0].x, y: rec.centers[0].y };
  } else {
    mk = manualKmeansCluster(STUDENTS, KMEANS_INIT_IDS, null, 0);
    const demo = kmeansStep(freshKmeans(), POINTS).records[0];
    label = 'NAMOYISH: 1-iteratsiya, C1 (hali bajarilmagan, holatga yozilmagan)';
    prog = { x: demo.centers[0].x, y: demo.centers[0].y };
  }
  const kdiff = mk.n ? Math.max(Math.abs(mk.cx - prog.x), Math.abs(mk.cy - prog.y)) : (isNum(prog.x) ? 0 : NaN);
  out.km = {
    title: '18.3 K-means', verdict: verdict(kdiff),
    lines: [label].concat(mk.rows.map(r => r.id + ': (' + f4(r.z[0]) + '; ' + f4(r.z[1]) + ')'), [
      'X yig\'indisi = ' + f10(mk.sumX), 'Y yig\'indisi = ' + f10(mk.sumY), 'Soni |C1| = ' + mk.n,
      mk.n ? 'Markaz = (1/' + mk.n + ') * sum = (' + f10(mk.cx) + '; ' + f10(mk.cy) + ')' : 'Klaster bo\'sh: eski markaz saqlanadi'
    ]),
    manual: mk.n ? 'Markaz = (' + f10(mk.cx) + '; ' + f10(mk.cy) + ')' : 'bo\'sh klaster',
    program: 'Markaz = (' + f10(prog.x) + '; ' + f10(prog.y) + ')'
  };
  return out;
}
function renderVerify() {
  const v = buildVerification();
  [['verifyKnn', v.knn], ['verifyTree', v.tree], ['verifyKm', v.km]].forEach(([id, c]) => {
    $(id).innerHTML = '<h3>' + esc(c.title) + '</h3><ul class="calc">' + c.lines.map(l => '<li>' + esc(l) + '</li>').join('') + '</ul>' +
      '<div class="side"><div><span class="label">Qo\'lda hisoblangan natija</span><p class="num">' + esc(c.manual) + '</p></div><div><span class="label">Dastur hisoblagan natija</span><p class="num">' + esc(c.program) + '</p></div></div>' +
      '<p class="verdict ' + (c.verdict.ok ? 'ok' : 'bad') + '" role="status">' + (c.verdict.ok ? '' : '✗ ') + esc(c.verdict.text) + '</p>';
  });
}
function verifyText() {
  const v = buildVerification();
  return ['knn', 'tree', 'km'].map(k => v[k].title + '\n' + v[k].lines.join('\n') + '\nQo\'lda hisoblangan natija: ' + v[k].manual + '\nDastur hisoblagan natija: ' + v[k].program + '\nNatija: ' + v[k].verdict.text).join('\n\n');
}

/* ----- Umumiy yangilash ----- */
function renderStudentDependent() {
  renderStats(); renderInput(); renderNorm(); renderKnn(); renderTree(); renderChartText(); renderVerify(); requestChart();
}
function renderKmeansDependent() {
  renderStats(); renderKmeans(); renderTable(); renderInput(); renderChartText(); renderVerify(); requestChart();
}
function renderAll() { renderStudentDependent(); renderKmeans(); renderTable(); }

/* ----- Amallar ----- */
function parseField(raw) {
  if (raw === null || raw === undefined || String(raw).trim() === '') return { ok: false, msg: 'Qiymat kiritilmagan. 0 dan 100 gacha son kiriting.' };
  const v = Number(raw);
  if (!Number.isFinite(v)) return { ok: false, msg: 'Faqat son kiriting.' };
  if (v < 0 || v > 100) return { ok: false, msg: 'Qiymat 0 dan 100 gacha bo\'lishi kerak.' };
  return { ok: true, v };
}
function setStudentValue(key, raw) {
  const r = parseField(raw);
  state.inputError[key] = r.ok ? '' : r.msg;
  if (!r.ok) { renderInput(); return; }
  if (state.newStudent[key] === r.v) { renderInput(); return; }
  state.newStudent[key] = r.v;
  recomputeStudent();
  renderStudentDependent();
}
function setProfile(name) {
  const p = PROFILES[name]; if (!p) return;
  state.inputError.score = ''; state.inputError.att = '';
  state.newStudent = { score: p.score, att: p.att };
  recomputeStudent(); renderStudentDependent();
  toast(p.label + ' tanlandi (' + p.score + '; ' + p.att + ')', 'info');
}
function doStep() {
  if (state.kmeans.finished) return;
  state.kmeans = kmeansStep(state.kmeans, POINTS);
  renderKmeansDependent();
  if (state.kmeans.finished) toast(state.kmeans.converged ? 'K-means konvergensiyaga yetdi (' + state.kmeans.iteration + '-iteratsiya).' : '100-iteratsiya chegarasiga yetildi.', 'success');
}
function stopAuto(silent) {
  if (state.auto.timer) clearTimeout(state.auto.timer);
  state.auto.timer = null;
  const was = state.auto.running;
  state.auto.running = false;
  if (was && !silent) toast('Avtomatik jarayon to\'xtatildi. Holat saqlandi.', 'info');
}
function startAuto() {
  if (state.auto.running || state.kmeans.finished) return;
  state.auto.running = true;
  renderKmeans(); renderStats();
  const tick = () => {
    if (!state.auto.running) return;
    doStep();
    if (state.kmeans.finished) { stopAuto(true); renderKmeans(); renderStats(); return; }
    state.auto.timer = setTimeout(tick, reduceMotion() ? 200 : 900);
  };
  state.auto.timer = setTimeout(tick, 300);
}
function resetKmeans() {
  stopAuto(true);
  state.kmeans = freshKmeans();
  chart.shown = null;
  renderKmeansDependent();
  toast('K-means 0-iteratsiyaga qaytarildi (T1, T15, T20).', 'info');
}
function copyText(text) {
  const fallback = () => {
    const ta = document.createElement('textarea');
    ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    ta.remove();
    toast(ok ? 'Natijalar nusxalandi (muqobil usul).' : 'Nusxalashda xatolik: brauzer ruxsat bermadi.', ok ? 'success' : 'error');
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(() => toast('Natijalar nusxalandi.', 'success'), () => { toast('Clipboard ishlamadi, muqobil usul sinab ko\'riladi.', 'error'); fallback(); });
  } else fallback();
}
function exportCsv() {
  const csv = tableText(',').split('\n').map(l => l.split(',').map(c => '"' + String(c).replace(/"/g, '""') + '"').join(',')).join('\r\n');
  // CSV'da haqiqiy qiymatlar: asl ball va qatnashish
  try {
    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'talabalar.csv';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    toast('CSV fayl yuklab olindi.', 'success');
  } catch (e) { toast('CSV eksport qilishda xatolik yuz berdi.', 'error'); }
}
function applyTheme(t, persist) {
  state.theme = t;
  document.documentElement.setAttribute('data-theme', t);
  const b = $('themeToggle');
  if (b) { b.setAttribute('aria-pressed', String(t === 'light')); b.querySelector('.tl').textContent = t === 'light' ? 'Qorong\'i rejim' : 'Yorug\' rejim'; }
  if (persist) { try { localStorage.setItem('mlLabTheme', t); } catch (e) { /* saqlash imkonsiz */ } }
  if (chart.ctx) { requestChart(); drawJChart(1); }
}

/* ==========================================================================
   6. ISHGA TUSHIRISH
   ========================================================================== */
function bindEvents() {
  if (listenersBound) return;
  listenersBound = true;
  $('themeToggle').addEventListener('click', () => applyTheme(state.theme === 'dark' ? 'light' : 'dark', true));
  $('menuToggle').addEventListener('click', () => { const o = document.body.classList.toggle('menu-open'); $('menuToggle').setAttribute('aria-expanded', String(o)); });
  document.querySelectorAll('.nav a').forEach(a => a.addEventListener('click', () => { document.body.classList.remove('menu-open'); $('menuToggle').setAttribute('aria-expanded', 'false'); }));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') { document.body.classList.remove('menu-open'); $('menuToggle').setAttribute('aria-expanded', 'false'); } });
  $('inputScore').addEventListener('input', e => setStudentValue('score', e.target.value));
  $('inputAtt').addEventListener('input', e => setStudentValue('att', e.target.value));
  $('rangeScore').addEventListener('input', e => setStudentValue('score', e.target.value));
  $('rangeAtt').addEventListener('input', e => setStudentValue('att', e.target.value));
  document.querySelectorAll('[data-profile]').forEach(b => b.addEventListener('click', () => setProfile(b.dataset.profile)));
  document.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => { state.chart.mode = b.dataset.mode; renderChartText(); requestChart(); }));
  $('searchId').addEventListener('input', e => { state.table.search = e.target.value; renderTable(); });
  $('filterClass').addEventListener('change', e => { state.table.cls = e.target.value; renderTable(); });
  $('filterCluster').addEventListener('change', e => { state.table.cluster = e.target.value; renderTable(); });
  const sorter = key => { const t = state.table; if (t.sortKey !== key) { t.sortKey = key; t.sortDir = 1; } else if (t.sortDir === 1) t.sortDir = -1; else { t.sortKey = null; t.sortDir = 1; } renderTable(); };
  $('sortScore').addEventListener('click', () => sorter('score'));
  $('sortAtt').addEventListener('click', () => sorter('att'));
  const pick = tr => {
    if (!tr || !tr.dataset.id) return;
    state.table.activeId = tr.dataset.id; state.chart.highlightId = tr.dataset.id;
    renderTable(); requestChart(); toast(tr.dataset.id + ' grafikda ajratildi.', 'info');
  };
  $('tableBody').addEventListener('click', e => pick(e.target.closest('tr')));
  $('tableBody').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(e.target.closest('tr')); } });
  $('btnCopyTable').addEventListener('click', () => copyText(tableText('\t')));
  $('btnCsv').addEventListener('click', exportCsv);
  $('btnStep').addEventListener('click', doStep);
  $('btnAuto').addEventListener('click', startAuto);
  $('btnStop').addEventListener('click', () => stopAuto(false) || (renderKmeans(), renderStats()));
  $('btnReset').addEventListener('click', resetKmeans);
  $('btnCopyVerify').addEventListener('click', () => copyText(verifyText()));
  $('btnHelp').addEventListener('click', () => { const d = $('helpDialog'); if (d.showModal) d.showModal(); else d.setAttribute('open', ''); });
  $('btnHelpClose').addEventListener('click', () => { const d = $('helpDialog'); if (d.close) d.close(); else d.removeAttribute('open'); });
  const cv = chart.cv;
  cv.addEventListener('pointermove', e => { if (e.pointerType !== 'mouse') return; const h = hitAt(e); showTip(h, e); cv.style.cursor = h ? 'pointer' : 'default'; });
  cv.addEventListener('pointerleave', () => { $('tooltip').hidden = true; });
  cv.addEventListener('pointerdown', e => { const h = hitAt(e); showTip(h, e); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) requestChart(); });
}

function initTilt() {
  if (reduceMotion() || window.matchMedia('(max-width: 768px)').matches) return;
  document.querySelectorAll('.tilt').forEach(el => {
    el.addEventListener('pointermove', e => {
      const r = el.getBoundingClientRect(), px = (e.clientX - r.left) / r.width - 0.5, py = (e.clientY - r.top) / r.height - 0.5;
      el.style.setProperty('--rx', (-py * 9).toFixed(2) + 'deg'); el.style.setProperty('--ry', (px * 11).toFixed(2) + 'deg');
      el.style.setProperty('--sx', (-px * 18).toFixed(1) + 'px'); el.style.setProperty('--sy', (py * 18 + 14).toFixed(1) + 'px');
    });
    el.addEventListener('pointerleave', () => { ['--rx', '--ry', '--sx', '--sy'].forEach(p => el.style.removeProperty(p)); });
  });
}
function initAmbient() {
  const cv = $('ambient');
  if (!cv || reduceMotion()) return;
  const ctx = cv.getContext('2d'), N = window.innerWidth < 768 ? 16 : 36;
  let w = 0, h = 0;
  const ps = Array.from({ length: N }, () => ({ x: Math.random(), y: Math.random(), vx: (Math.random() - 0.5) * 0.00006, vy: (Math.random() - 0.5) * 0.00006, r: 1 + Math.random() * 1.8 }));
  const size = () => { w = cv.width = window.innerWidth; h = cv.height = window.innerHeight; };
  size(); window.addEventListener('resize', size);
  const loop = () => {
    if (!document.hidden) {
      ctx.clearRect(0, 0, w, h);
      const c = cssVars();
      ctx.fillStyle = c.cyan; ctx.strokeStyle = c.cyan;
      ps.forEach(p => { p.x = (p.x + p.vx * 16 + 1) % 1; p.y = (p.y + p.vy * 16 + 1) % 1; ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.arc(p.x * w, p.y * h, p.r, 0, Math.PI * 2); ctx.fill(); });
      ctx.globalAlpha = 0.08;
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        const dx = (ps[i].x - ps[j].x) * w, dy = (ps[i].y - ps[j].y) * h;
        if (dx * dx + dy * dy < 14000) { ctx.beginPath(); ctx.moveTo(ps[i].x * w, ps[i].y * h); ctx.lineTo(ps[j].x * w, ps[j].y * h); ctx.stroke(); }
      }
      ctx.globalAlpha = 1;
    }
    requestAnimationFrame(loop);
  };
  loop();
}
function initScrollEffects() {
  if (!('IntersectionObserver' in window)) { document.querySelectorAll('.reveal').forEach(e => e.classList.add('in')); return; }
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08 });
  document.querySelectorAll('.reveal').forEach(e => io.observe(e));
  const links = Array.from(document.querySelectorAll('.nav a'));
  const spy = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) links.forEach(a => { const on = a.getAttribute('href') === '#' + e.target.id; a.classList.toggle('active', on); if (on) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); });
  }), { rootMargin: '-40% 0px -55% 0px' });
  document.querySelectorAll('main section[id]').forEach(s => spy.observe(s));
  const cio = new IntersectionObserver(es => { chart.visible = es[0].isIntersecting; if (chart.visible) requestChart(); });
  cio.observe($('mainChart'));
}

function init() {
  let saved = null;
  try { saved = localStorage.getItem('mlLabTheme'); } catch (e) { saved = null; }
  applyTheme(saved === 'light' ? 'light' : 'dark', false);
  chart.cv = $('mainChart'); chart.ctx = chart.cv.getContext('2d');
  resizeChart();
  recomputeStudent();
  bindEvents();
  renderAll();
  if ('ResizeObserver' in window) new ResizeObserver(() => { resizeChart(); drawJChart(1); requestChart(); }).observe(chart.cv.parentElement);
  else window.addEventListener('resize', () => { resizeChart(); requestChart(); });
  initTilt(); initAmbient(); initScrollEffects();
  requestChart();
}

if (hasDOM) {
  const boot = () => {
    try { init(); } catch (err) {
      const box = document.getElementById('toasts');
      if (box) { const t = document.createElement('div'); t.className = 'toast error'; t.textContent = 'Dastur ishga tushmadi: ' + err.message; box.appendChild(t); }
      if (typeof console !== 'undefined') console.error(err);
    }
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    STUDENTS, CLASS_NAMES, FEATURES, PROFILES, BOUNDS, POINTS, EPS, KMEANS_INIT_IDS, KMEANS_MAX_ITER,
    computeBounds, minMax, normalizePoint, euclid, sqEuclid, countClasses, giniFromCounts, majorityClass,
    knnClassify, evaluateSplit, buildTree, treeNodes, treePredict, kmeansCreate, kmeansInitCenters, kmeansStep, kmeansRun,
    manualKnn, manualTreeRoot, manualKmeansCluster, verdict, parseField, freshKmeans
  };
}
