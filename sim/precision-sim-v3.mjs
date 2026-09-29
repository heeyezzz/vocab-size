// 词汇量测试精度仿真 v3 —— 直接量"实证检出力"
//
// 前两轮的问题：
//   v1 太慢（9.5 分钟）；v2 的窄带对比有混淆变量（固定 ln 半宽 → 低频端带的词数更少）
// v3 的改法：
//   1. 窄带固定"词数宽度"，不再固定 ln 宽度
//   2. 不再用解析 z 公式，改为真实模拟两次测试：t0 的考生 → t1 涨了多少词的考生，
//      各跑一次完整测试，统计"有多少次能正确判定为显著变化"（实证检出力）
//   3. 判定标准用真实实现会用的那套：每次测试自己算出的渐近标准误，
//      两次差值 > 1.96 × SE(差) 才算显著
//
// 模型（2PL，有"不认识"按钮所以猜测参数 c=0）：
//   P(认识 | 词频排名 r) = sigmoid( a * (theta - ln r) )
//   真实词汇量 V = Σ_{r=1..20000} P(认识 | r)

import { performance } from 'node:perf_hooks';

const N_RANK = 20000;
const LN = new Float64Array(N_RANK + 1);
for (let r = 1; r <= N_RANK; r++) LN[r] = Math.log(r);
const LN_MAX = LN[N_RANK];
const THETA_LO = -1.0, THETA_HI = LN_MAX + 1.0, THETA_STEP = 0.02;
const sig = z => 1 / (1 + Math.exp(-z));

const N_GRID = Math.round((THETA_HI - THETA_LO) / THETA_STEP) + 1;
const thetaAt = i => THETA_LO + i * THETA_STEP;

function buildVocabTable(a) {
  const tbl = new Float64Array(N_GRID);
  for (let i = 0; i < N_GRID; i++) {
    const t = thetaAt(i);
    let v = 0;
    for (let r = 1; r <= N_RANK; r++) v += sig(a * (t - LN[r]));
    tbl[i] = v;
  }
  return tbl;
}
function vocabAt(tbl, theta) {
  const idx = (theta - THETA_LO) / THETA_STEP;
  if (idx <= 0) return tbl[0];
  if (idx >= N_GRID - 1) return tbl[N_GRID - 1];
  const i = Math.floor(idx), f = idx - i;
  return tbl[i] * (1 - f) + tbl[i + 1] * f;
}
function thetaForVocab(tbl, V) {
  let lo = 0, hi = N_GRID - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (tbl[m] < V) lo = m; else hi = m; }
  const frac = (V - tbl[lo]) / ((tbl[hi] - tbl[lo]) || 1);
  return thetaAt(lo) + frac * THETA_STEP;
}

// ---------- 极大似然 + 渐近标准误 ----------
const A_GRID = [];
for (let a = 1.0; a <= 9.0; a += 0.5) A_GRID.push(a);

function mleTheta(a, items) {
  let t = items.reduce((s, it) => s + it.x, 0) / items.length;
  for (let i = 0; i < 30; i++) {
    let g = 0, h = 0;
    for (const it of items) { const p = sig(a * (t - it.x)); g += it.y - p; h += p * (1 - p); }
    if (h < 1e-9) break;
    const step = Math.max(-1.2, Math.min(1.2, g / (a * h)));
    t += step;
    if (Math.abs(step) < 1e-8) break;
  }
  return Math.max(THETA_LO, Math.min(THETA_HI, t));
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
// 渐近 SE(theta) = 1/sqrt(Σ a²·p(1-p))，再链式换算成 SE(V)
function asymptoticSeV(f, items, tbl) {
  let info = 0;
  for (const it of items) { const p = sig(f.a * (f.theta - it.x)); info += f.a * f.a * p * (1 - p); }
  const seTheta = info > 1e-9 ? 1 / Math.sqrt(info) : 5;
  const dVdTheta = (vocabAt(tbl, f.theta + 0.05) - vocabAt(tbl, f.theta - 0.05)) / 0.1;
  return Math.abs(dVdTheta) * seTheta;
}

// ---------- 虚拟考生 ----------
function makeLearner(theta, a, seed) {
  let s = ((seed + 1) * 2654435761) >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  return { theta, a, rnd, knows(r) { return rnd() < sig(a * (theta - LN[r])); } };
}
function pickNear(target, used, rnd) {
  let lo = 1, hi = N_RANK;
  while (lo < hi) { const m = (lo + hi) >> 1; if (LN[m] < target) lo = m + 1; else hi = m; }
  for (const win of [60, 300, 2000, N_RANK]) {
    const c = [];
    for (let d = 0; d < win; d++) {
      for (const x of [lo + d, lo - d]) if (x >= 1 && x <= N_RANK && !used.has(x)) c.push(x);
      if (c.length >= 12) break;
    }
    if (c.length) return c[Math.floor(rnd() * c.length)];
  }
  return null;
}
function drawUniform(learner, used, n, rankLo, rankHi) {
  // 在 [rankLo, rankHi) 这个"词数区间"里等距抽 n 个未用过的词
  const items = [];
  for (let i = 0; i < n; i++) {
    const t = rankLo + (rankHi - rankLo) * ((i + 0.5) / n) + (learner.rnd() - 0.5) * ((rankHi - rankLo) / n);
    const r = pickNear(LN[Math.max(1, Math.min(N_RANK, Math.round(t)))], used, learner.rnd);
    if (r == null) continue;
    used.add(r);
    items.push({ x: LN[r], y: learner.knows(r) ? 1 : 0, rank: r });
  }
  return items;
}
function adaptiveFill(learner, used, items, nTotal, startTheta, seedSpread) {
  let f = fit(items);
  let th = startTheta != null ? startTheta : f.theta;
  while (items.length < nTotal) {
    const prog = items.length / nTotal;
    const spread = seedSpread * Math.pow(1 - prog, 0.6) + 0.12;
    const r = pickNear(th + (learner.rnd() * 2 - 1) * spread, used, learner.rnd);
    if (r == null) break;
    used.add(r);
    items.push({ x: LN[r], y: learner.knows(r) ? 1 : 0, rank: r });
    if (items.length % 6 === 5) { f = fit(items); th = f.theta; }
  }
  return fit(items);
}

// ---------- 测试方案（一次 sitting）----------
// 返回 { v, seV, band?: {p, n, width} }
function sittingA_blindHybrid(L, n, tbl) {
  const used = new Set();
  const nSweep = Math.round(n * 0.5);
  const items = drawUniform(L, used, nSweep, 1, N_RANK);
  const f = adaptiveFill(L, used, items, n, null, 0.9);
  return { v: vocabAt(tbl, f.theta), seV: asymptoticSeV(f, items, tbl), n: items.length };
}
function sittingB_priorSeeded(L, n, tbl, priorBias) {
  const used = new Set();
  const prior = L.theta + priorBias;
  const rLo = Math.max(1, Math.round(Math.exp(prior - 1.1))), rHi = Math.min(N_RANK, Math.round(Math.exp(prior + 1.1)));
  const items = drawUniform(L, used, 12, rLo, rHi);
  const f = adaptiveFill(L, used, items, n, prior, 0.7);
  return { v: vocabAt(tbl, f.theta), seV: asymptoticSeV(f, items, tbl), n: items.length };
}
function sittingC_hybrid(L, nAdapt, nBand, tbl, bandWidthRanks, bandOffsetLn) {
  const used = new Set();
  const prior = L.theta;
  const rLo0 = Math.max(1, Math.round(Math.exp(prior - 1.1))), rHi0 = Math.min(N_RANK, Math.round(Math.exp(prior + 1.1)));
  const items = drawUniform(L, used, 12, rLo0, rHi0);
  const f = adaptiveFill(L, used, items, nAdapt, prior, 0.7);
  // 窄带：固定"词数宽度"，中心 = 上次阈值 + 偏移（真实实现用历史 theta，这里用真值近似）
  const centerRank = Math.exp(L.theta + bandOffsetLn);
  const bLo = Math.max(1, Math.round(centerRank - bandWidthRanks / 2));
  const bHi = Math.min(N_RANK, bLo + bandWidthRanks);
  const bandItems = drawUniform(L, used, nBand, bLo, bHi);
  const hits = bandItems.reduce((s, it) => s + it.y, 0);
  return {
    v: vocabAt(tbl, f.theta), seV: asymptoticSeV(f, items, tbl), n: nAdapt + bandItems.length,
    band: { p: bandItems.length ? hits / bandItems.length : 0, n: bandItems.length, width: bHi - bLo },
  };
}

// ---------- 实证检出力 ----------
// 模拟 t0 和 t1 两次测试，t1 的考生词汇量涨了 gain 词
function power(label, nSims, runPair, gain, tbl) {
  let detected = 0, ok = 0, falseAlarmBase = 0;
  const deltas = [];
  for (let i = 0; i < nSims; i++) {
    const [r0, r1] = runPair(i);
    if (!r0 || !r1) continue;
    ok++;
    const sed = Math.sqrt(r0.seV ** 2 + r1.seV ** 2);
    const d = r1.v - r0.v;
    deltas.push(d);
    if (Math.abs(d) > 1.96 * sed) detected++;
  }
  const m = deltas.reduce((a, b) => a + b, 0) / deltas.length;
  const s = Math.sqrt(deltas.reduce((q, x) => q + (x - m) ** 2, 0) / (deltas.length - 1));
  return { label, power: 100 * detected / ok, biasVsTruth: m - gain, spread: s };
}

// 假警报率：gain=0 时误判为显著的比例（应当 ≈5%）
function falseAlarm(label, nSims, runPair, tbl) {
  let det = 0, ok = 0;
  for (let i = 0; i < nSims; i++) {
    const [r0, r1] = runPair(i);
    if (!r0 || !r1) continue;
    ok++;
    const sed = Math.sqrt(r0.seV ** 2 + r1.seV ** 2);
    if (Math.abs(r1.v - r0.v) > 1.96 * sed) det++;
  }
  return { label, fa: 100 * det / ok };
}

// ---------- 主程序 ----------
const N_SIMS = parseInt(process.argv[2] || '400', 10);
const t0 = performance.now();
const SLOPES = [2.5, 4.0, 6.0];
const tables = new Map();
for (const a of SLOPES) tables.set(a, buildVocabTable(a));

console.log(`仿真次数/配置 = ${N_SIMS}（每个数字 = 跑 ${N_SIMS} 次"两个月前后各测一次"）`);

console.log(`\n${'#'.repeat(100)}`);
console.log('# 第一部分：总量估算精度（单次测试的 95% CI 半宽，单位：词）');
console.log('# 括号内为相对误差。这是"我有多少词"这个数字本身有多准。');
console.log(`${'#'.repeat(100)}`);

const LEVELS = [3000, 5000, 8000, 12000];
const designs1 = [
  ['A 盲扫混合  n=80', (L, tbl) => sittingA_blindHybrid(L, 80, tbl)],
  ['A 盲扫混合  n=160', (L, tbl) => sittingA_blindHybrid(L, 160, tbl)],
  ['B 先验播种  n=80', (L, tbl) => sittingB_priorSeeded(L, 80, tbl, 0)],
  ['B 先验播种  n=120', (L, tbl) => sittingB_priorSeeded(L, 120, tbl, 0)],
  ['B 先验播种  n=160', (L, tbl) => sittingB_priorSeeded(L, 160, tbl, 0)],
  ['B 先验播种  n=200', (L, tbl) => sittingB_priorSeeded(L, 200, tbl, 0)],
];

for (const a of SLOPES) {
  const tbl = tables.get(a);
  console.log(`\n${'─'.repeat(100)}`);
  console.log(`曲线陡度 a = ${a}${a <= 2.6 ? '  (过渡平缓)' : a >= 5.9 ? '  (过渡陡峭)' : '  (中等)'}`);
  console.log('─'.repeat(100));
  console.log('方案'.padEnd(24) + LEVELS.map(v => `V=${v}`.padStart(17)).join(''));
  for (const [label, fn] of designs1) {
    let line = label.padEnd(22);
    for (const V of LEVELS) {
      const theta = thetaForVocab(tbl, V);
      const vs = [], ses = [];
      for (let i = 0; i < N_SIMS; i++) { const r = fn(makeLearner(theta, a, i * 3 + 7), tbl); vs.push(r.v); ses.push(r.seV); }
      const m = vs.reduce((x, y) => x + y, 0) / vs.length;
      const s = Math.sqrt(vs.reduce((q, x) => q + (x - m) ** 2, 0) / (vs.length - 1));
      line += `  ±${String(Math.round(1.96 * s)).padStart(4)} (${(100 * s / V).toFixed(1)}%)`.padStart(19);
    }
    console.log(line);
  }
}

console.log(`\n${'#'.repeat(100)}`);
console.log('# 第二部分：实证检出力 —— 你真的涨了 G 个词，测试有多大把握报警？');
console.log('# 数字 = 检出率%。基线 V=6000，两次测试间隔期内涨了 G 词。');
console.log('# 同时给"假警报率"（G=0 时误报的比例，理想值 5%）—— 校准这套判定标准是否可信。');
console.log(`${'#'.repeat(100)}`);

const GAINS = [0, 300, 500, 800, 1200];
for (const a of SLOPES) {
  const tbl = tables.get(a);
  const V0 = 6000;
  const theta0 = thetaForVocab(tbl, V0);
  console.log(`\n${'─'.repeat(100)}`);
  console.log(`曲线陡度 a = ${a}    基线词汇量 V=6000`);
  console.log('─'.repeat(100));
  console.log('方案'.padEnd(26) + GAINS.map(g => (g === 0 ? '假警报率' : `涨${g}词`).padStart(13)).join(''));

  const designs2 = [
    ['A 盲扫混合 n=80', g => [() => sittingA_blindHybrid(makeLearner(theta0, a, 1), tbl), () => sittingA_blindHybrid(makeLearner(thetaForVocab(tbl, V0 + g), a, 2), tbl)]],
    ['B 先验播种 n=80', g => [() => sittingB_priorSeeded(makeLearner(theta0, a, 11), tbl, 0), () => sittingB_priorSeeded(makeLearner(thetaForVocab(tbl, V0 + g), a, 12), tbl, 0)]],
    ['B 先验播种 n=120', g => [() => sittingB_priorSeeded(makeLearner(theta0, a, 21), tbl, 0), () => sittingB_priorSeeded(makeLearner(thetaForVocab(tbl, V0 + g), a, 22), tbl, 0)]],
    ['B 先验播种 n=160', g => [() => sittingB_priorSeeded(makeLearner(theta0, a, 31), tbl, 0), () => sittingB_priorSeeded(makeLearner(thetaForVocab(tbl, V0 + g), a, 32), tbl, 0)]],
    ['B 先验播种 n=200', g => [() => sittingB_priorSeeded(makeLearner(theta0, a, 41), tbl, 0), () => sittingB_priorSeeded(makeLearner(thetaForVocab(tbl, V0 + g), a, 42), tbl, 0)]],
  ];

  for (const [label, mk] of designs2) {
    let line = label.padEnd(24);
    for (const g of GAINS) {
      const [f0, f1] = mk(g);
      let det = 0, ok = 0;
      for (let i = 0; i < N_SIMS; i++) {
        // 每次重复都换随机种子，否则每次结果一样
        const s0 = i * 977 + 5, s1 = i * 977 + 999;
        const r0 = g === 0
          ? sittingRun(label, theta0, a, s0, tbl)
          : sittingRun(label, theta0, a, s0, tbl);
        const r1 = sittingRun(label, thetaForVocab(tbl, V0 + g), a, s1, tbl);
        const sed = Math.sqrt(r0.seV ** 2 + r1.seV ** 2);
        ok++;
        if (Math.abs(r1.v - r0.v) > 1.96 * sed) det++;
      }
      line += `${(100 * det / ok).toFixed(0).padStart(11)}%`;
    }
    console.log(line);
  }
}

function sittingRun(label, theta, a, seed, tbl) {
  const L = makeLearner(theta, a, seed);
  const n = parseInt(label.match(/n=(\d+)/)[1], 10);
  return label.startsWith('A') ? sittingA_blindHybrid(L, n, tbl) : sittingB_priorSeeded(L, n, tbl, 0);
}

console.log(`\n${'#'.repeat(100)}`);
console.log('# 第三部分：窄带追踪 vs 总量估算 —— 同样时间，谁更能看出进步？');
console.log('# 窄带固定 3000 词宽（修正 v2 的混淆变量），中心放在阈值上方的不同位置');
console.log('# 数字 = 检出率%（该窄带内命中率变化被判定为显著的比例）');
console.log(`${'#'.repeat(100)}`);

{
  const a = 4.0, tbl = tables.get(a), V0 = 6000;
  const theta0 = thetaForVocab(tbl, V0);
  const BAND_W = 3000;
  console.log(`\n曲线陡度 a=4.0, 基线 V=6000, 窄带宽 3000 词, 每次 100 题（~13 分钟）`);
  console.log('─'.repeat(100));
  console.log('带中心相对阈值'.padEnd(20) + '带内初始命中率'.padStart(16) + GAINS.slice(1).map(g => `涨${g}词`.padStart(13)).join(''));
  console.log('─'.repeat(100));

  for (const offRanks of [-2500, -1500, 0, 1500, 3000, 5000]) {
    const centerRank = Math.round(Math.exp(theta0)) + offRanks;
    const bLo = Math.max(1, centerRank - BAND_W / 2), bHi = Math.min(N_RANK, bLo + BAND_W);
    let k0 = 0;
    for (let r = Math.round(bLo); r < Math.round(bHi); r++) k0 += sig(a * (theta0 - LN[r]));
    const p0 = k0 / BAND_W;

    let line = `${offRanks >= 0 ? '+' : ''}${offRanks} 名`.padEnd(18) + `${(100 * p0).toFixed(1)}%`.padStart(16);
    for (const g of GAINS.slice(1)) {
      const theta1 = thetaForVocab(tbl, V0 + g);
      let det = 0;
      for (let i = 0; i < N_SIMS; i++) {
        const L0 = makeLearner(theta0, a, i * 613 + 3), L1 = makeLearner(theta1, a, i * 613 + 77);
        const u0 = new Set(), u1 = new Set();
        const b0 = drawUniform(L0, u0, 100, Math.round(bLo), Math.round(bHi));
        const b1 = drawUniform(L1, u1, 100, Math.round(bLo), Math.round(bHi));
        const h0 = b0.reduce((s, it) => s + it.y, 0) / b0.length;
        const h1 = b1.reduce((s, it) => s + it.y, 0) / b1.length;
        const sed = Math.sqrt(h0 * (1 - h0) / b0.length + h1 * (1 - h1) / b1.length);
        if (sed > 0 && Math.abs(h1 - h0) > 1.96 * sed) det++;
      }
      line += `${(100 * det / N_SIMS).toFixed(0).padStart(11)}%`;
    }
    console.log(line);
  }
  console.log('\n对照：总量估算 B 先验播种 n=100 的检出力见第二部分（n=80 与 n=120 之间）');
}

console.log(`\n总耗时 ${((performance.now() - t0) / 1000).toFixed(1)} 秒`);
