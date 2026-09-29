// 词汇量测试精度仿真
// 用已知"真值"的虚拟考生跑几千遍，量出各方案的实际误差
//
// 心理测量模型（2PL，因为有"不认识"按钮所以猜测参数 c=0）：
//   P(认识 | 词频排名 r) = sigmoid( a * (theta - ln r) )
//   theta = 50% 认识阈值（ln 排名单位），a = 曲线陡度
//   真实词汇量 V = Σ_{r=1..20000} P(认识 | r)

const N_RANK = 20000;
const LN = new Float64Array(N_RANK + 1);
for (let r = 1; r <= N_RANK; r++) LN[r] = Math.log(r);
const LN_MAX = LN[N_RANK];

const sig = z => 1 / (1 + Math.exp(-z));

function trueVocab(theta, a) {
  let v = 0;
  for (let r = 1; r <= N_RANK; r++) v += sig(a * (theta - LN[r]));
  return v;
}

// ---------- 极大似然估计 ----------
const A_GRID = [];
for (let a = 1.0; a <= 9.01; a += 0.4) A_GRID.push(a);

function logLik(theta, a, items) {
  let s = 0;
  for (const it of items) {
    const p = sig(a * (theta - it.x));
    const pc = Math.min(1 - 1e-9, Math.max(1e-9, p));
    s += it.y ? Math.log(pc) : Math.log(1 - pc);
  }
  return s;
}

function mleTheta(a, items) {
  let lo = -3, hi = LN_MAX + 3;
  for (let i = 0; i < 45; i++) {
    const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
    if (logLik(m1, a, items) < logLik(m2, a, items)) lo = m1; else hi = m2;
  }
  return (lo + hi) / 2;
}

function fit(items) {
  let best = null;
  for (const a of A_GRID) {
    const t = mleTheta(a, items);
    const l = logLik(t, a, items);
    if (!best || l > best.ll) best = { ll: l, theta: t, a };
  }
  return best;
}

// ---------- 虚拟考生 ----------
function makeLearner(theta, a, seed) {
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  return {
    theta, a, rnd,
    knows(rank) { return rnd() < sig(a * (theta - LN[rank])); },
  };
}

// ---------- 抽题策略 ----------

// 分层粗扫：在 ln 排名轴上等距分 k 层，每层随机抽 1 个未用过的词
function stratified(learner, used, nStrata, spanLo = LN[1], spanHi = LN_MAX) {
  const items = [];
  for (let i = 0; i < nStrata; i++) {
    const target = spanLo + (spanHi - spanLo) * ((i + 0.5) / nStrata);
    const rank = pickNear(target, used, learner);
    if (rank == null) continue;
    used.add(rank);
    items.push({ x: LN[rank], y: learner.knows(rank) ? 1 : 0, rank });
  }
  return items;
}

// 在 ln 排名 target 附近找一个未用过的词
function pickNear(target, used, learner) {
  let lo = 1, hi = N_RANK;
  while (lo < hi) { const m = (lo + hi) >> 1; if (LN[m] < target) lo = m + 1; else hi = m; }
  for (let d = 0; d < 400; d++) {
    for (const cand of [lo + d, lo - d]) {
      if (cand >= 1 && cand <= N_RANK && !used.has(cand)) return cand;
    }
  }
  return null;
}

// 自适应：每次在当前 theta 估计附近随机抽一题（最大信息点）
function adaptiveStep(learner, used, thetaHat, spread) {
  const target = thetaHat + (learner.rnd() * 2 - 1) * spread;
  const rank = pickNear(Math.max(LN[1], Math.min(LN_MAX, target)), used, learner);
  if (rank == null) return null;
  used.add(rank);
  return { x: LN[rank], y: learner.knows(rank) ? 1 : 0, rank };
}

// 方案 A：盲扫混合（先分层定位，再自适应加密）—— 原设计
function runBlindHybrid(learner, nTotal, sweepFrac) {
  const used = new Set();
  const nSweep = Math.round(nTotal * sweepFrac);
  const items = stratified(learner, used, nSweep);
  let f = fit(items);
  const nRefine = nTotal - items.length;
  for (let i = 0; i < nRefine; i++) {
    const spread = 0.9 * Math.pow(1 - i / nRefine, 0.6) + 0.15;
    const it = adaptiveStep(learner, used, f.theta, spread);
    if (!it) break;
    items.push(it);
    if (i % 4 === 3) f = fit(items);
  }
  f = fit(items);
  return { theta: f.theta, v: trueVocab(f.theta, f.a), n: items.length, items };
}

// 方案 B：先验播种（跳过盲扫，直接从先验附近开始自适应）
function runPriorSeeded(learner, nTotal, priorBias) {
  const used = new Set();
  const prior = learner.theta + priorBias;
  let thetaHat = prior;
  const items = [];
  // 先验附近撒 12 题摸底
  for (let i = 0; i < 12; i++) {
    const it = adaptiveStep(learner, used, prior, 1.1);
    if (it) items.push(it);
  }
  let f = fit(items);
  thetaHat = f.theta;
  for (let i = items.length; i < nTotal; i++) {
    const prog = (i - 12) / Math.max(1, nTotal - 12);
    const spread = 0.7 * Math.pow(1 - prog, 0.6) + 0.12;
    const it = adaptiveStep(learner, used, thetaHat, spread);
    if (!it) break;
    items.push(it);
    if (i % 4 === 3) { f = fit(items); thetaHat = f.theta; }
  }
  f = fit(items);
  return { theta: f.theta, v: trueVocab(f.theta, f.a), n: items.length, items };
}

// 方案 D：窄带追踪（只在阈值附近的固定难度带里抽题，测命中率）
function runNarrowBand(learner, nTotal, halfWidth) {
  const used = new Set();
  const lo = learner.theta - halfWidth, hi = learner.theta + halfWidth;
  const items = stratified(learner, used, nTotal, lo, hi);
  const hits = items.reduce((s, it) => s + it.y, 0);
  return { p: hits / items.length, n: items.length };
}

// ---------- 跑仿真 ----------
function sd(xs) {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1));
}

function simulate(label, nSims, fn) {
  const vs = [], ns = [];
  for (let i = 0; i < nSims; i++) {
    const r = fn(i);
    if (r == null) continue;
    vs.push(r.v); ns.push(r.n);
  }
  const truth = TRUTH_V;
  const bias = vs.reduce((a, b) => a + b, 0) / vs.length - truth;
  const s = sd(vs);
  return { label, n: Math.round(ns.reduce((a, b) => a + b, 0) / ns.length), mean: Math.round(truth + bias), bias: Math.round(bias), sd: Math.round(s), rel: (100 * s / truth).toFixed(1) + '%', ci95: '±' + Math.round(1.96 * s), detect95: Math.round(1.96 * Math.SQRT2 * s) };
}

// ---------- 主程序 ----------
const N_SIMS = parseInt(process.argv[2] || '800', 10);

for (const [theta, a, label] of [[Math.log(4200), 4.0, 'V≈5000'], [Math.log(9500), 4.0, 'V≈10000']]) {
  const TRUTH = trueVocab(theta, a);
  globalThis.TRUTH_V = TRUTH;
  console.log(`\n${'='.repeat(96)}`);
  console.log(`虚拟考生：theta=ln(${Math.round(Math.exp(theta))}), a=${a}  →  真实词汇量 ${Math.round(TRUTH)} 词  (${N_SIMS} 次重复)`);
  console.log('='.repeat(96));

  const rows = [];

  // 方案 A：盲扫混合，不同题量
  for (const n of [80, 120, 160, 200]) {
    rows.push(simulate(`A 盲扫混合  n=${n} (粗扫50%)`, N_SIMS, i => {
      const L = makeLearner(theta, a, i * 7919 + 13);
      return runBlindHybrid(L, n, 0.5);
    }));
  }
  // 方案 B：先验播种，不同题量 + 不同先验偏差
  for (const n of [80, 120, 160]) {
    for (const bias of [0, 0.6, -0.6]) {
      rows.push(simulate(`B 先验播种  n=${n}  先验偏差${bias >= 0 ? '+' : ''}${bias}`, N_SIMS, i => {
        const L = makeLearner(theta, a, i * 7919 + 13);
        return runPriorSeeded(L, n, bias);
      }));
    }
  }

  console.log('方案'.padEnd(38) + '题数'.padStart(5) + '估算均值'.padStart(9) + '偏差'.padStart(7) + '标准差'.padStart(8) + '相对误差'.padStart(9) + '95%CI'.padStart(9) + '可检出变化'.padStart(11));
  console.log('-'.repeat(96));
  for (const r of rows) {
    console.log(r.label.padEnd(36) + String(r.n).padStart(6) + String(r.mean).padStart(10) + String(r.bias).padStart(8) + String(r.sd).padStart(8) + r.rel.padStart(10) + r.ci95.padStart(10) + String(r.detect95).padStart(11));
  }
}

// ---------- 方案 D：窄带追踪的变化检出力 ----------
console.log(`\n${'='.repeat(96)}`);
console.log('方案 D 窄带追踪：能不能看出"这个带里多认识了 500 词"？');
console.log('='.repeat(96));
{
  const BAND_W = 3000, GAIN = 500;
  const theta0 = Math.log(6000), a = 4.0;
  const baseKnown = trueVocabBand(theta0, a, 6000, 6000 + BAND_W);
  const afterKnown = baseKnown + GAIN;
  // 反解 gain 后的 theta
  let lo = theta0 - 2, hi = theta0 + 2;
  for (let i = 0; i < 60; i++) {
    const m = (lo + hi) / 2;
    if (trueVocabBand(m, a, 6000, 6000 + BAND_W) < afterKnown) lo = m; else hi = m;
  }
  const theta1 = (lo + hi) / 2;
  const p0 = baseKnown / BAND_W, p1 = afterKnown / BAND_W;

  console.log(`难度带 = 词频排名 6000-9000（宽 3000 词）`);
  console.log(`进步前认识 ${Math.round(baseKnown)} 词 (命中率 ${(100 * p0).toFixed(1)}%) → 进步后 ${Math.round(afterKnown)} 词 (${(100 * p1).toFixed(1)}%)，真变化 +${(100 * (p1 - p0)).toFixed(1)} 个百分点\n`);

  console.log('每带题数'.padEnd(10) + '单次命中率标准误'.padStart(18) + '两次差值标准误'.padStart(16) + '真变化/标准误'.padStart(16) + '  能否 95% 检出');
  console.log('-'.repeat(96));
  for (const n of [40, 60, 80, 100, 140, 200, 300]) {
    const se0 = Math.sqrt(p0 * (1 - p0) / n), se1 = Math.sqrt(p1 * (1 - p1) / n);
    const sed = Math.sqrt(se0 * se0 + se1 * se1);
    const z = (p1 - p0) / sed;
    console.log(String(n).padEnd(12) + (100 * se0).toFixed(1).padStart(16) + '%' + (100 * sed).toFixed(1).padStart(15) + '%' + z.toFixed(2).padStart(16) + (z >= 1.96 ? '     ✓ 能' : '     ✗ 不能'));
  }
}

function trueVocabBand(theta, a, rLo, rHi) {
  let v = 0;
  for (let r = rLo; r < rHi; r++) v += sig(a * (theta - LN[r]));
  return v;
}
