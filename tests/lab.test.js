'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const L = require('../script.js');

const close = (a, b, eps) => Math.abs(a - b) < (eps || 1e-9);
const POINTS = L.POINTS;

/* A. Dataset */
test('A1 dataset aynan 20 talabadan iborat', () => assert.equal(L.STUDENTS.length, 20));
test('A2 ID lar noyob', () => assert.equal(new Set(L.STUDENTS.map(s => s.id)).size, 20));
test('A3 qiymatlar 0-100 oralig\'ida', () => L.STUDENTS.forEach(s => {
  assert.ok(s.score >= 0 && s.score <= 100 && s.att >= 0 && s.att <= 100, s.id);
}));
test('A4 sinflar faqat Oddiy/Faol, soni 13/7', () => {
  const c = L.countClasses(L.STUDENTS);
  assert.deepEqual(L.STUDENTS.map(s => s.cls).filter(x => x !== 'Oddiy' && x !== 'Faol'), []);
  assert.equal(c.Oddiy, 13); assert.equal(c.Faol, 7);
});
test('A5 min/max: ball 40..96, qatnashish 45..96', () => {
  assert.deepEqual(L.BOUNDS.score, { min: 40, max: 96 });
  assert.deepEqual(L.BOUNDS.att, { min: 45, max: 96 });
});

/* B. Min-Max */
test('B1 min -> 0, max -> 1', () => {
  assert.equal(L.minMax(40, 40, 96), 0); assert.equal(L.minMax(96, 40, 96), 1);
});
test('B2 max === min bo\'lsa 0', () => assert.equal(L.minMax(5, 5, 5), 0));
test('B3 sinf normalizatsiyaga ta\'sir qilmaydi', () => {
  const a = L.normalizePoint({ score: 70, att: 60, cls: 'Oddiy' }, L.BOUNDS);
  const b = L.normalizePoint({ score: 70, att: 60, cls: 'Faol' }, L.BOUNDS);
  assert.deepEqual(a, b);
});
test('B4 diapazondan tashqari qiymat (ball 0) kesilmaydi', () => {
  assert.ok(close(L.minMax(0, 40, 96), -40 / 56));
});
test('B5 to\'liq aniqlik saqlanadi (toFixed ishlatilmaydi)', () => {
  const z = L.normalizePoint({ score: 80, att: 85 }, L.BOUNDS);
  assert.notEqual(z.x, Number(z.x.toFixed(4)));
});

/* C. KNN */
test('C1 (80,85): norm, qo\'shnilar T16,T17,T15, Faol', () => {
  const r = L.knnClassify({ score: 80, att: 85 }, L.STUDENTS, L.BOUNDS, 3);
  assert.ok(close(r.z.x, 0.7142857143, 1e-10)); assert.ok(close(r.z.y, 0.7843137255, 1e-10));
  assert.deepEqual(r.neighbors.map(n => n.student.id), ['T16', 'T17', 'T15']);
  assert.ok(r.neighbors.every(n => n.student.cls === 'Faol'));
  assert.equal(r.prediction, 'Faol'); assert.equal(r.votes.Faol, 3);
});
test('C2 (55,50) -> Oddiy', () => {
  assert.equal(L.knnClassify({ score: 55, att: 50 }, L.STUDENTS, L.BOUNDS, 3).prediction, 'Oddiy');
});
test('C3 teng masofada ID bo\'yicha deterministik tartib', () => {
  const ds = [
    { id: 'T2', score: 60, att: 50, cls: 'Faol' }, { id: 'T1', score: 40, att: 50, cls: 'Oddiy' },
    { id: 'T3', score: 50, att: 90, cls: 'Oddiy' }, { id: 'T4', score: 100, att: 10, cls: 'Faol' }
  ];
  const b = L.computeBounds(ds);
  const r = L.knnClassify({ score: 50, att: 50 }, ds, b, 3);
  assert.deepEqual(r.sorted.slice(0, 2).map(x => x.student.id), ['T1', 'T2']);
});

/* D. Decision Tree */
const tree = L.buildTree(L.STUDENTS, 2);
test('D1 ildiz chuqurligi 0, maxDepth 2 oshmaydi', () => {
  assert.equal(tree.depth, 0);
  assert.equal(Math.max(...L.treeNodes(tree).map(n => n.depth)), 2);
});
test('D2 ildiz: ball <= 70.5', () => { assert.equal(tree.split.feature, 'score'); assert.equal(tree.split.threshold, 70.5); });
test('D3 o\'ng: qatnashish <= 74.5', () => { assert.equal(tree.right.split.feature, 'att'); assert.equal(tree.right.split.threshold, 74.5); });
test('D4 ikkala parametr ishlatilgan', () => {
  const f = new Set(L.treeNodes(tree).filter(n => !n.leaf).map(n => n.split.feature));
  assert.ok(f.has('score') && f.has('att'));
});
test('D5 ildiz vaznli Gini = 0.2916666667', () => assert.ok(close(tree.best.weighted, 0.2916666667, 1e-10)));
test('D6 bargalar: chap Oddiy, o\'ng-chap Oddiy, o\'ng-o\'ng Faol', () => {
  assert.ok(tree.left.leaf && tree.left.prediction === 'Oddiy');
  assert.equal(tree.right.left.prediction, 'Oddiy'); assert.equal(tree.right.right.prediction, 'Faol');
});
test('D7 (80,85) -> Faol; (55,50) -> Oddiy', () => {
  assert.equal(L.treePredict(tree, { score: 80, att: 85 }).prediction, 'Faol');
  assert.equal(L.treePredict(tree, { score: 55, att: 50 }).prediction, 'Oddiy');
});
test('D8 Gini formulasi: 50/50 -> 0.5, toza -> 0', () => {
  assert.equal(L.giniFromCounts({ Oddiy: 5, Faol: 5 }), 0.5); assert.equal(L.giniFromCounts({ Oddiy: 4, Faol: 0 }), 0);
});

/* E. K-means */
test('E1 3 ta turli boshlang\'ich markaz va 0-iteratsiya', () => {
  const km = L.freshKmeans();
  assert.equal(km.iteration, 0); assert.equal(km.centers.length, 3);
  assert.equal(new Set(km.centers.map(c => c.x + ',' + c.y)).size, 3);
  assert.deepEqual(L.KMEANS_INIT_IDS, ['T1', 'T15', 'T20']);
});
test('E2 bir qadamdan so\'ng 1-iteratsiya, o\'lchamlar 2/14/4, C1=T1,T9, J, markaz', () => {
  const km = L.kmeansStep(L.freshKmeans(), POINTS), r = km.records[0];
  assert.equal(km.iteration, 1);
  assert.deepEqual(r.clusters.map(c => c.count), [2, 14, 4]);
  assert.deepEqual(r.clusters[0].ids, ['T1', 'T9']);
  assert.ok(close(r.J, 1.4908247832, 1e-9), 'J = ' + r.J);
  assert.ok(close(r.centers[0].x, 0.2946428571, 1e-9)); assert.equal(r.centers[0].y, 0);
});
test('E3 klasterlar 20 talabani qamraydi, har talaba bitta klasterda', () => {
  const km = L.kmeansStep(L.freshKmeans(), POINTS), ids = [].concat(...km.records[0].clusters.map(c => c.ids));
  assert.equal(ids.length, 20); assert.equal(new Set(ids).size, 20);
});
test('E4 konvergensiya <= 100, J kamaymaydigan emas (oshmaydi)', () => {
  const km = L.kmeansRun(L.freshKmeans(), POINTS);
  assert.ok(km.converged && km.finished); assert.ok(km.iteration <= 100);
  for (let i = 1; i < km.jHistory.length; i++) assert.ok(km.jHistory[i] <= km.jHistory[i - 1] + 1e-12);
});
test('E5 qayta boshlash 0-iteratsiyani tiklaydi', () => {
  const a = L.kmeansRun(L.freshKmeans(), POINTS), b = L.freshKmeans();
  assert.notEqual(a.iteration, 0); assert.equal(b.iteration, 0); assert.equal(b.records.length, 0); assert.equal(b.assignments, null);
});
test('E6 bo\'sh klaster xatoga olib kelmaydi, eski markaz saqlanadi', () => {
  const centers = L.kmeansInitCenters(POINTS, ['T1', 'T20']).concat([{ x: 5, y: 5 }]);
  const km = L.kmeansStep(L.kmeansCreate(POINTS, centers), POINTS);
  assert.equal(km.records[0].clusters[2].count, 0);
  assert.deepEqual(km.centers[2], { x: 5, y: 5 });
  assert.ok(Number.isFinite(km.records[0].J));
});
test('E7 100 iteratsiyadan keyin to\'xtaydi (finished)', () => {
  const km = L.kmeansRun(L.freshKmeans(), POINTS); assert.ok(km.iteration <= L.KMEANS_MAX_ITER);
});

/* G. Qo'lda tekshirish va validatsiya */
test('G1 mustaqil qo\'lda KNN hisobi dastur bilan mos', () => {
  const m = L.manualKnn({ score: 80, att: 85 }, L.STUDENTS), r = L.knnClassify({ score: 80, att: 85 }, L.STUDENTS, L.BOUNDS, 3);
  assert.deepEqual(m.top.map(x => x.s.id), r.neighbors.map(x => x.student.id));
  m.top.forEach((x, i) => assert.ok(Math.abs(x.d - r.neighbors[i].d) < L.EPS));
});
test('G2 mustaqil daraxt ildizi hisobi dastur bilan mos', () => {
  const m = L.manualTreeRoot(L.STUDENTS, 'score', 70.5);
  assert.ok(Math.abs(m.w - tree.best.weighted) < L.EPS); assert.equal(m.nL, 8); assert.equal(m.nR, 12);
});
test('G3 mustaqil K-means C1 hisobi (0-iteratsiyadagi namoyish)', () => {
  const m = L.manualKmeansCluster(L.STUDENTS, L.KMEANS_INIT_IDS, null, 0), r = L.kmeansStep(L.freshKmeans(), POINTS).records[0];
  assert.ok(Math.abs(m.cx - r.centers[0].x) < L.EPS && Math.abs(m.cy - r.centers[0].y) < L.EPS);
});
test('G4 verdict: farq katta bo\'lsa "Mos keldi" chiqmaydi', () => {
  assert.equal(L.verdict(0).ok, true); assert.equal(L.verdict(1e-3).ok, false); assert.equal(L.verdict(NaN).ok, false);
  assert.ok(!/Mos keldi/.test(L.verdict(1e-3).text));
});
test('G5 kiritish validatsiyasi: NaN, bo\'sh, diapazondan tashqari rad etiladi', () => {
  ['', 'abc', '-1', '101', 'Infinity'].forEach(v => assert.equal(L.parseField(v).ok, false, v));
  assert.equal(L.parseField('0').ok, true); assert.equal(L.parseField('100').ok, true);
});
test('G6 script.js DOM-siz yuklanadi', () => assert.equal(typeof document, 'undefined'));
