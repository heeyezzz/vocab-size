// 词汇量测试精度仿真 v2 —— 提速 + 参数扫描
//
// 模型（2PL，有"不认识"按钮所以猜测参数 c=0）：
//   P(认识 | 词频排名 r) = sigmoid( a * (theta - ln r) )
//   theta = 50% 阈值（ln 排名），a = 曲线陡度（未知，需扫描）
//   真实词汇量 V = Σ_{r=1..20000} P(认识 | r)
//
// v1 → v2 提速：牛顿迭代替代三分搜索；V(theta) 预计算查找表；降低重拟合频率

import { performance } from 'node:perf_hooks';

const N_RANK = 20000;
const LN = new Float64Array(N_RANK + 1);
for (let r = 1; r <= N_RANK; r++) LN[r] = Math.log(r);
const LN_MAX = LN[N_RANK];
const sig = z => 1 / (1 + Math.exp(-z));

// ---------- V(theta) 查找表 ----------
const THETA_GRID = [];
for (let t = -1.0; t <= LN_MAX + 1.0; t += 0.02) THETA_GRID.push(t);

function buildVocabTable(a) {
  const tbl = new Float64Array(THETA_GRID.length);
  for (let i = 0; i < THETA_GRID.length; i++) {
    const t = THETA_GRID[i];
    let v = 0;
    for (let r = 1; r <= N_RANK; r++) v += sig(a * (t - LN[r]));
    tbl[i] = v;
  }
  return tbl;
}

function vocabAt(tbl, theta) {
  const idx = (theta - THETA_GRID[0]) / 0.02;
  if (idx <= 0) return tbl[0];
  if (idx >= tbl.length - 1) return tbl[tbl.length - 1];
  const i = Math.floor(idx), f = idx - i;
  return tbl[i] * (1 - f) + tbl[i + 1] * f;
}

// ---------- 极大似然 ----------
const A_GRID = [];
for (let a = 1.0; a <= 9.0; a += 0.5) A_GRID.push(a);

function mleTheta(a, items) {
  let t = items.reduce((s, it) => s + it.x, 0) / items.length;
  for (let i = 0; i < 25; i++) {
    let g = 0, h = 0;
    for (const it of items) {
      const p = sig(a * (t - it.x));
      g += it.y - p;
      h += p * (1 - p);
    }
    if (h < 1e-9) break;
    const step = g / (a * h);
    t += Math.max(-1.2, Math.min(1.2, step));
    if (Math.abs(step) < 1e-7) break;
  }
  // 全对/全错时逻辑斯谛回归无有限极大值，会跑飞 —— 夹到词表范围内
  return Math.max(THETA_GRID[0], Math.min(THETA_GRID[THETA_GRID.length - 1], t));
}

function logLik(theta, a, items) {
  let s = 0;
  for (const it of items) {
    const p = Math.min(1 - 1e-12, Math.max(1e-12, sig(a * (theta - it.x))));
    s += it.y ? Math.log(p) : Math.log(1 - p);
  }
  return s;
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
  let s = (seed * 2654435761) >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  return { theta, a, rnd, knows(r) { return rnd() < sig(a * (theta - LN[r])); } };
}

// ---------- 抽题 ----------
function pickNear(target, used, rnd) {
  let lo = 1, hi = N_RANK;
  while (lo < hi) { const m = (lo + hi) >> 1; if (LN[m] < target) lo = m + 1; else hi = m; }
  // 逐级放宽窗口，避免在词表两端因窗口内词都被用过而丢题
  for (const win of [60, 300, 2000, N_RANK]) {
    const cands = [];
    for (let d = 0; d < win; d++) {
      for (const c of [lo + d, lo - d]) if (c >= 1 && c <= N_RANK && !used.has(c)) cands.push(c);
      if (cands.length >= 12) break;
    }
    if (cands.length) return cands[Math.floor(rnd() * cands.length)];
  }
  return null;
}

function drawBand(learner, used, n, lo, hi) {
  const items = [];
  for (let i = 0; i < n; i++) {
    const target = lo + (hi - lo) * ((i + 0.5) / n) + (learner.rnd() - 0.5) * ((hi - lo) / n);
    const rank = pickNear(Math.max(LN[1], Math.min(LN_MAX, target)), used, learner.rnd);
    if (rank == null) continue;
    used.add(rank);
    items.push({ x: LN[rank], y: learner.knows(rank) ? 1 : 0, rank });
  }
  return items;
}

function drawStratified(learner, used, n, lo, hi) {
  return drawBand(learner, used, n, lo, hi);
}

// 方案 A：盲扫混合
function runBlindHybrid(L, nTotal, sweepFrac) {
  const used = new Set();
  const nSweep = Math.round(nTotal * sweepFrac);
  const items = drawStratified(L, used, nSweep, LN[1], LN_MAX);
  let f = fit(items);
  const nRef = nTotal - items.length;
  for (let i = 0; i < nRef; i++) {
    const prog = i / nRef;
    const spread = 0.9 * Math.pow(1 - prog, 0.6) + 0.15;
    const r = pickNear(f.theta + (L.rnd() * 2 - 1) * spread, used, L.rnd);
    if (r == null) break;
    used.add(r);
    items.push({ x: LN[r], y: L.knows(r) ? 1 : 0, rank: r });
    if (i % 6 === 5) f = fit(items);
  }
  f = fit(items);
  return { theta: f.theta, v: vocabAt(L.__tbl, f.theta), n: items.length };
}

// 方案 B：先验播种自适应
function runPriorSeeded(L, nTotal, priorBias) {
  const used = new Set();
  const prior = L.theta + priorBias;
  const items = drawBand(L, used, 12, prior - 1.1, prior + 1.1);
  let f = fit(items);
  for (let i = items.length; i < nTotal; i++) {
    const prog = (i - 12) / Math.max(1, nTotal - 12);
    const spread = 0.7 * Math.pow(1 - prog, 0.6) + 0.12;
    const r = pickNear(f.theta + (L.rnd() * 2 - 1) * spread, used, L.rnd);
    if (r == null) break;
    used.add(r);
    items.push({ x: LN[r], y: L.knows(r) ? 1 : 0, rank: r });
    if (i % 6 === 5) f = fit(items);
  }
  f = fit(items);
  return { theta: f.theta, v: vocabAt(L.__tbl, f.theta), n: items.length };
}

// 方案 C：混合双任务 —— nAdapt 题先验自适应定总量 + nBand 题窄带追踪
function runHybrid(L, nAdapt, nBand, bandOffset, bandHalfWidthLn) {
  const used = new Set();
  const ad = runPriorSeededSubset(L, used, nAdapt);
  const center = L.theta + bandOffset; // 真实场景用上次阈值；这里用真值+偏移
  const bandItems = drawBand(L, used, nBand, center - bandHalfWidthLn, center + bandHalfWidthLn);
  const hits = bandItems.reduce((s, it) => s + it.y, 0);
  return {
    theta: ad.theta, v: vocabAt(L.__tbl, ad.theta), n: nAdapt + bandItems.length,
    bandP: bandItems.length ? hits / bandItems.length : null, bandN: bandItems.length,
    bandKnownWords: bandItems.length ? (hits / bandItems.length) * bandItems.length : 0,
    bandWidthRanks: Math.round(Math.exp(center + bandHalfWidthLn) - Math.exp(center - bandHalfWidthLn)),
  };
}

function runPriorSeededSubset(L, used, nTotal) {
  const items = drawBand(L, used, Math.min(12, nTotal), L.theta - 1.1, L.theta + 1.1);
  let f = fit(items);
  for (let i = items.length; i < nTotal; i++) {
    const prog = (i - 12) / Math.max(1, nTotal - 12);
    const spread = 0.7 * Math.pow(1 - prog, 0.6) + 0.12;
    const r = pickNear(f.theta + (L.rnd() * 2 - 1) * spread, used, L.rnd);
    if (r == null) break;
    used.add(r);
    items.push({ x: LN[r], y: L.knows(r) ? 1 : 0, rank: r });
    if (i % 6 === 5) f = fit(items);
  }
  return fit(items);
}

// ---------- 统计 ----------
function sd(xs) { const m = mean(xs); return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1)); }
function mean(xs) { return xs.reduce((a, b) => a + b, 0) / xs.length; }

// ---------- 主扫描 ----------
const N_SIMS = parseInt(process.argv[2] || '400', 10);
const t0 = performance.now();

const VOCAB_LEVELS = [3000, 5000, 8000, 12000];
const SLOPES = [2.5, 4.0, 6.0];

console.log(`仿真次数/配置 = ${N_SIMS}`);
console.log(`\n${'#'.repeat(104)}`);
console.log('# 第一部分：总量估算精度 —— 题量 × 曲线陡度 a × 真实水平');
console.log('# 表中数字 = 95% 置信区间半宽（词）；括号内 = 相对误差');
console.log(`${'#'.repeat(104)}`);

const tables = new Map();
for (const a of SLOPES) tables.set(a, buildVocabTable(a));

function thetaForVocab(a, V) {
  const tbl = tables.get(a);
  let lo = 0, hi = tbl.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (tbl[m] < V) lo = m; else hi = m;
  }
  const frac = (V - tbl[lo]) / ((tbl[hi] - tbl[lo]) || 1);
  return THETA_GRID[lo] + frac * 0.02;
}

for (const a of SLOPES) {
  console.log(`\n${'─'.repeat(104)}`);
  console.log(`曲线陡度 a = ${a}  ${a <= 2.6 ? '(过渡平缓：从"几乎全认识"到"几乎全不认识"跨约 8 倍词频)' : a >= 5.9 ? '(过渡陡峭：跨约 2 倍词频)' : '(中等：跨约 3 倍词频)'}`);
  console.log('─'.repeat(104));
  console.log('方案'.padEnd(30) + VOCAB_LEVELS.map(v => `V=${v}`.padStart(16)).join(''));
  console.log('─'.repeat(104));

  const designs = [
    ['A 盲扫混合 n=80', L => runBlindHybrid(L, 80, 0.5)],
    ['A 盲扫混合 n=160', L => runBlindHybrid(L, 160, 0.5)],
    ['B 先验播种 n=80', L => runPriorSeeded(L, 80, 0)],
    ['B 先验播种 n=120', L => runPriorSeeded(L, 120, 0)],
    ['B 先验播种 n=160', L => runPriorSeeded(L, 160, 0)],
    ['B 先验播种 n=200', L => runPriorSeeded(L, 200, 0)],
    ['C 混合 60自适应+60窄带', L => runHybrid(L, 60, 60, 0, 0.55)],
    ['C 混合 40自适应+80窄带', L => runHybrid(L, 40, 80, 0, 0.55)],
  ];

  for (const [label, fn] of designs) {
    let line = label.padEnd(28);
    for (const V of VOCAB_LEVELS) {
      const theta = thetaForVocab(a, V);
      const ests = [];
      for (let i = 0; i < N_SIMS; i++) {
        const L = makeLearner(theta, a, i + 1);
        L.__tbl = tables.get(a);
        const r = fn(L);
        if (process.env.DEBUG && i < 5) console.error(`   [dbg ${label} V=${V} i=${i}] theta_true=${theta.toFixed(3)} theta_hat=${r.theta.toFixed(3)} v=${r.v.toFixed(1)} n=${r.n}`);
        ests.push(r.v);
      }
      const s = sd(ests);
      line += `  ±${String(Math.round(1.96 * s)).padStart(4)} (${(100 * s / V).toFixed(1)}%)`.padStart(18);
    }
    console.log(line);
  }
}

console.log(`\n${'#'.repeat(104)}`);
console.log('# 第二部分：变化检出力 —— 能否看出"这个难度带里多认识了 G 个词"？');
console.log('# 窄带中心 = 你当前阈值（该处命中率≈50%，对变化最敏感）');
console.log(`${'#'.repeat(104)}`);

for (const a of SLOPES) {
  console.log(`\n${'─'.repeat(104)}`);
  console.log(`曲线陡度 a = ${a}`);
  console.log('─'.repeat(104));
  console.log('带半宽(ln)'.padEnd(12) + '≈带内词数'.padStart(11) + '  ' + [60, 80, 100, 140].map(n => `n=${n}`.padStart(15)).join(''));
  console.log('        检出 +G 词所需的 z 值（≥1.96 才能在 95% 置信下确认不是噪声）'.padEnd(12));
  console.log('─'.repeat(104));

  for (const hw of [0.35, 0.55, 0.80]) {
    const theta = thetaForVocab(a, 6000);
    const rLo = Math.max(1, Math.round(Math.exp(theta - hw)));
    const rHi = Math.min(N_RANK, Math.round(Math.exp(theta + hw)));
    const bandW = rHi - rLo;
    let known0 = 0;
    for (let r = rLo; r < rHi; r++) known0 += sig(a * (theta - LN[r]));
    const p0 = known0 / bandW;

    for (const G of [300, 500, 800]) {
      const p1 = Math.min(0.999, (known0 + G) / bandW);
      let line = `${hw.toFixed(2)}`.padEnd(12) + `${bandW}`.padStart(9) + `  `;
      if (G === 500) line = `${hw.toFixed(2)} G=${G}`.padEnd(12) + `${bandW}`.padStart(9) + `  `;
      else line = `     G=${G}`.padEnd(12) + `${bandW}`.padStart(9) + `  `;
      for (const n of [60, 80, 100, 140]) {
        const sed = Math.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n);
        const z = (p1 - p0) / sed;
        line += `${z.toFixed(2)}${z >= 1.96 ? ' ✓' : ' ✗'}`.padStart(15);
      }
      console.log(line);
    }
    console.log('');
  }
}

console.log(`\n${'#'.repeat(104)}`);
console.log('# 第三部分：窄带中心放在哪里最灵敏？（V=6000, a=4.0, n=100, 检测 +500 词）');
console.log(`${'#'.repeat(104)}`);
{
  const a = 4.0, theta = thetaForVocab(a, 6000), n = 100, G = 500, hw = 0.55;
  console.log('带中心偏移(ln)'.padEnd(16) + '带内命中率'.padStart(12) + 'z 值'.padStart(10) + '  判定');
  console.log('─'.repeat(60));
  for (const off of [-0.6, -0.3, 0, 0.3, 0.6, 1.0]) {
    const c = theta + off;
    const rLo = Math.max(1, Math.round(Math.exp(c - hw))), rHi = Math.min(N_RANK, Math.round(Math.exp(c + hw)));
    const bw = rHi - rLo;
    let k0 = 0; for (let r = rLo; r < rHi; r++) k0 += sig(a * (theta - LN[r]));
    const p0 = k0 / bw, p1 = Math.min(0.999, (k0 + G) / bw);
    const sed = Math.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n);
    const z = (p1 - p0) / sed;
    console.log(`${off >= 0 ? '+' : ''}${off.toFixed(1)}`.padEnd(18) + (100 * p0).toFixed(1).padStart(10) + '%' + z.toFixed(2).padStart(10) + '   ' + (z >= 1.96 ? '✓ 可检出' : '✗ 淹没在噪声'));
  }
}

console.log(`\n总耗时 ${((performance.now() - t0) / 1000).toFixed(1)} 秒`);
