// v4 —— 决定性实验：增长"集中"还是"分散"，决定窄带追踪值不值得做
//
// v3 的结论是窄带追踪与总量估算打平，但 v3 假设增长 = θ 整体平移（分散在全词表）。
// 现实中如果你集中攻某一段词表（如 CET-6），增长是集中的。本脚本报两种增长模型：
//
//   模型 S（spread）    ：θ 整体平移，+G 词均匀落在过渡区   —— v3 已测，这里复现做对照
//   模型 C（concentrated）：+G 词全部落在指定的一个 3000 词宽排名区间内
//
// 对每种模型，比较两种测法的检出力：
//   测法 1：总量估算（先验播种自适应 n=100）
//   测法 2：窄带追踪（在增长发生的那个 3000 词带里抽 n=100）

import { performance } from 'node:perf_hooks';

const N_RANK = 20000;
const LN = new Float64Array(N_RANK + 1);
for (let r = 1; r <= N_RANK; r++) LN[r] = Math.log(r);
const LN_MAX = LN[N_RANK];
const THETA_LO = -1.0, THETA_HI = LN_MAX + 1.0, THETA_STEP = 0.02;
const N_GRID = Math.round((THETA_HI - THETA_LO) / THETA_STEP) + 1;
const sig = z => 1 / (1 + Math.exp(-z));
const thetaAt = i => THETA_LO + i * THETA_STEP;

function buildVocabTable(a) {
  const t = new Float64Array(N_GRID);
  for (let i = 0; i < N_GRID; i++) { const th = thetaAt(i); let v = 0; for (let r = 1; r <= N_RANK; r++) v += sig(a * (th - LN[r])); t[i] = v; }
  return t;
}
function vocabAt(tbl, th) {
  const i0 = (th - THETA_LO) / THETA_STEP;
  if (i0 <= 0) return tbl[0]; if (i0 >= N_GRID - 1) return tbl[N_GRID - 1];
  const i = Math.floor(i0), f = i0 - i; return tbl[i] * (1 - f) + tbl[i + 1] * f;
}
function thetaForVocab(tbl, V) {
  let lo = 0, hi = N_GRID - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (tbl[m] < V) lo = m; else hi = m; }
  return thetaAt(lo) + ((V - tbl[lo]) / ((tbl[hi] - tbl[lo]) || 1)) * THETA_STEP;
}

const A_GRID = []; for (let a = 1.0; a <= 9.0; a += 0.5) A_GRID.push(a);
function mleTheta(a, items) {
  let t = items.reduce((s, it) => s + it.x, 0) / items.length;
  for (let i = 0; i < 30; i++) {
    let g = 0, h = 0;
    for (const it of items) { const p = sig(a * (t - it.x)); g += it.y - p; h += p * (1 - p); }
    if (h < 1e-9) break;
    const st = Math.max(-1.2, Math.min(1.2, g / (a * h))); t += st;
    if (Math.abs(st) < 1e-8) break;
  }
  return Math.max(THETA_LO, Math.min(THETA_HI, t));
}
function logLik(th, a, items) {
  let s = 0;
  for (const it of items) { const p = Math.min(1 - 1e-12, Math.max(1e-12, sig(a * (th - it.x)))); s += it.y ? Math.log(p) : Math.log(1 - p); }
  return s;
}
function fit(items) {
  let b = null;
  for (const a of A_GRID) { const t = mleTheta(a, items); const l = logLik(t, a, items); if (!b || l > b.ll) b = { ll: l, theta: t, a }; }
  return b;
}
function seV(f, items, tbl) {
  let info = 0;
  for (const it of items) { const p = sig(f.a * (f.theta - it.x)); info += f.a * f.a * p * (1 - p); }
  const sT = info > 1e-9 ? 1 / Math.sqrt(info) : 5;
  const dV = (vocabAt(tbl, f.theta + 0.05) - vocabAt(tbl, f.theta - 0.05)) / 0.1;
  return Math.abs(dV) * sT;
}

// ---------- 考生 ----------
// extraKnown: 一个 Set，表示"额外学会的词"（模型 C 用）
function makeLearner(theta, a, seed, extraKnown) {
  let s = ((seed + 1) * 2654435761) >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  return {
    theta, a, rnd,
    knows(r) { if (extraKnown && extraKnown.has(r)) return true; return rnd() < sig(a * (theta - LN[r])); },
  };
}
function pickNear(target, used, rnd) {
  let lo = 1, hi = N_RANK;
  while (lo < hi) { const m = (lo + hi) >> 1; if (LN[m] < target) lo = m + 1; else hi = m; }
  for (const win of [60, 300, 2000, N_RANK]) {
    const c = [];
    for (let d = 0; d < win; d++) { for (const x of [lo + d, lo - d]) if (x >= 1 && x <= N_RANK && !used.has(x)) c.push(x); if (c.length >= 12) break; }
    if (c.length) return c[Math.floor(rnd() * c.length)];
  }
  return null;
}
function drawUniform(L, used, n, rLo, rHi) {
  const items = [];
  for (let i = 0; i < n; i++) {
    const t = rLo + (rHi - rLo) * ((i + 0.5) / n) + (L.rnd() - 0.5) * ((rHi - rLo) / n);
    const r = pickNear(LN[Math.max(1, Math.min(N_RANK, Math.round(t)))], used, L.rnd);
    if (r == null) continue;
    used.add(r);
    items.push({ x: LN[r], y: L.knows(r) ? 1 : 0, rank: r });
  }
  return items;
}
// 测法 1：总量估算（先验播种自适应）
function totalEstimate(L, n, tbl, prior) {
  const used = new Set();
  const rLo = Math.max(1, Math.round(Math.exp(prior - 1.1))), rHi = Math.min(N_RANK, Math.round(Math.exp(prior + 1.1)));
  const items = drawUniform(L, used, 12, rLo, rHi);
  let f = fit(items), th = prior;
  while (items.length < n) {
    const prog = items.length / n;
    const spread = 0.7 * Math.pow(1 - prog, 0.6) + 0.12;
    const r = pickNear(th + (L.rnd() * 2 - 1) * spread, used, L.rnd);
    if (r == null) break;
    used.add(r);
    items.push({ x: LN[r], y: L.knows(r) ? 1 : 0, rank: r });
    if (items.length % 6 === 5) { f = fit(items); th = f.theta; }
  }
  f = fit(items);
  return { v: vocabAt(tbl, f.theta), se: seV(f, items, tbl) };
}
// 测法 2：窄带追踪
function bandHitRate(L, n, rLo, rHi) {
  const used = new Set();
  const items = drawUniform(L, used, n, rLo, rHi);
  const p = items.reduce((s, it) => s + it.y, 0) / items.length;
  return { p, se: Math.sqrt(p * (1 - p) / items.length), n: items.length };
}

// ---------- 主实验 ----------
const N_SIMS = parseInt(process.argv[2] || '600', 10);
const t0 = performance.now();
const a = 4.0;
const tbl = buildVocabTable(a);
const V0 = 6000;
const theta0 = thetaForVocab(tbl, V0);
const THETA_RANK = Math.round(Math.exp(theta0));

// 集中增长的目标带：阈值上方的 3000 词（模拟"攻下一段新词表"）
const BAND_LO = THETA_RANK, BAND_HI = Math.min(N_RANK, THETA_RANK + 3000);
const BAND_W = BAND_HI - BAND_LO;
const N_ITEM = 100;

console.log(`仿真次数/配置 = ${N_SIMS}    曲线陡度 a=${a}    基线词汇量 V=${V0}`);
console.log(`阈值对应词频排名 ≈ ${THETA_RANK}；集中增长目标带 = 排名 ${BAND_LO}-${BAND_HI}（宽 ${BAND_W} 词）`);
console.log(`两种测法都用 ${N_ITEM} 题（约 13 分钟）\n`);

const GAINS = [0, 200, 300, 500, 800];

function runModel(modelName, makeT1) {
  console.log(`${'═'.repeat(100)}`);
  console.log(`${modelName}`);
  console.log(`${'═'.repeat(100)}`);
  console.log('测法'.padEnd(24) + GAINS.map(g => (g === 0 ? '假警报率' : `涨${g}词`).padStart(13)).join(''));
  console.log('─'.repeat(100));

  for (const [mName, mFn] of [
    ['总量估算 n=100', 'total'],
    ['窄带追踪 n=100', 'band'],
  ]) {
    let line = mName.padEnd(22);
    for (const g of GAINS) {
      let det = 0;
      for (let i = 0; i < N_SIMS; i++) {
        const L0 = makeLearner(theta0, a, i * 787 + 3, null);
        const L1 = makeT1(i, g);
        if (mFn === 'total') {
          const r0 = totalEstimate(L0, N_ITEM, tbl, theta0);
          const r1 = totalEstimate(L1, N_ITEM, tbl, theta0);
          const sed = Math.sqrt(r0.se ** 2 + r1.se ** 2);
          if (sed > 0 && Math.abs(r1.v - r0.v) > 1.96 * sed) det++;
        } else {
          const b0 = bandHitRate(L0, N_ITEM, BAND_LO, BAND_HI);
          const b1 = bandHitRate(L1, N_ITEM, BAND_LO, BAND_HI);
          const sed = Math.sqrt(b0.se ** 2 + b1.se ** 2);
          if (sed > 0 && Math.abs(b1.p - b0.p) > 1.96 * sed) det++;
        }
      }
      line += `${(100 * det / N_SIMS).toFixed(0).padStart(11)}%`;
    }
    console.log(line);
  }
  console.log('');
}

// 模型 S：分散增长（θ 整体平移）
runModel('模型 S：增长分散 —— θ 整体平移，+G 词均匀落在整个过渡区（自然阅读的典型模式）',
  (i, g) => makeLearner(thetaForVocab(tbl, V0 + g), a, i * 787 + 999, null));

// 模型 C：集中增长（+G 词全部落在目标带内）
// 知识是概率性的，所以"净增 G 词"定义为：Σ(1 - p_r) = G，即期望命中率增量折合 G 个词。
// 构造方法：在带内随机取词，累加 (1-p) 直到达到 G。
function makeConcentratedLearner(i, g) {
  const seed = i * 787 + 999;
  if (g === 0) return makeLearner(theta0, a, seed, null);
  let s = ((seed + 1) * 2654435761) >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const gained = new Set();
  let acc = 0, guard = 0;
  while (acc < g && guard++ < 200000) {
    const r = BAND_LO + Math.floor(rnd() * BAND_W);
    if (gained.has(r)) continue;
    gained.add(r);
    acc += 1 - sig(a * (theta0 - LN[r]));
  }
  return makeLearner(theta0, a, seed, gained);
}

runModel(`模型 C：增长集中 —— 折合 +G 词全部落在排名 ${BAND_LO}-${BAND_HI} 这一段（集中攻一个词表的典型模式）`,
  (i, g) => makeConcentratedLearner(i, g));

console.log(`${'═'.repeat(100)}`);
console.log('附：模型 C 下窄带命中率的绝对变化幅度（供换算成"这个带里多认识了多少词"）');
console.log(`${'═'.repeat(100)}`);
{
  let k0 = 0;
  for (let r = BAND_LO; r < BAND_HI; r++) k0 += sig(a * (theta0 - LN[r]));
  const p0 = k0 / BAND_W;
  console.log(`带内初始认识 ${Math.round(k0)} / ${BAND_W} 词，命中率 ${(100 * p0).toFixed(1)}%`);
  for (const g of GAINS.slice(1)) {
    const p1 = (k0 + g) / BAND_W;
    console.log(`  涨 ${g} 词 → 命中率 ${(100 * p1).toFixed(1)}%（+${(100 * (p1 - p0)).toFixed(1)} 个百分点），折合带内 ${(100 * g / BAND_W).toFixed(1)}% 的词`);
  }
}

console.log(`\n总耗时 ${((performance.now() - t0) / 1000).toFixed(1)} 秒`);
