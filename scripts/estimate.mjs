#!/usr/bin/env node
// vocab-size · IRT 估计核心
//
// 模型（2PL，因为有"不认识"按钮所以猜测参数 c = 0）：
//     P(认识 | 词频排名 r) = sigmoid( a · (theta - ln r) )
//
//   theta = 命中率跌到 50% 的那个位置（对数词频坐标），是个人水平的唯一刻度
//   a     = 曲线陡峭度。a 大 = 会的就是会的、不会的就是不会的；a 小 = 半生不熟的词多
//
//   词汇量 V(theta) = Σ_{r=1..20000} P(认识 | r)
//
// 两种模式的差别，全在 a 上：
//   level 模式题目横跨整个词表，theta 和 a 能分开认出来 → a 由本次作答拟合出来
//   track 模式题目挤在一个 3000 词的窄带里，ln r 变化太小，theta 和 a 会互相顶替、
//     认不出谁是谁 → 必须把 a 钉死在上一次 level 测出的值上
//
// 参考实现已在 sim/precision-sim-v3.mjs 里跑过 800×多组蒙特卡洛验证，
// 渐近标准误的假警报率实测 4~7%（名义值 5%），所以这里沿用渐近 SE 而不是自助法。

import { loadWordlist, TARGET_SIZE } from './wordlist.mjs';
import { isMain, makeRnd, median } from './util.mjs';

const N_RANK = TARGET_SIZE;
const LN = new Float64Array(N_RANK + 1);
for (let r = 1; r <= N_RANK; r++) LN[r] = Math.log(r);

const THETA_LO = -1.0;
const THETA_HI = LN[N_RANK] + 1.0;
const THETA_STEP = 0.02;
const N_GRID = Math.round((THETA_HI - THETA_LO) / THETA_STEP) + 1;
const thetaAt = i => THETA_LO + i * THETA_STEP;

const A_GRID = [];
for (let a = 0.5; a <= 9.0; a += 0.5) A_GRID.push(Math.round(a * 10) / 10);

export const DEFAULT_A = 4.0;
export const THETA_BOUNDS = { lo: THETA_LO, hi: THETA_HI };

const sigmoid = z => 1 / (1 + Math.exp(-z));

// ---------- theta <-> 词汇量 换算表 ----------
const vocabTables = new Map();

function vocabTable(a) {
  let tbl = vocabTables.get(a);
  if (tbl) return tbl;
  tbl = new Float64Array(N_GRID);
  for (let i = 0; i < N_GRID; i++) {
    const t = thetaAt(i);
    let v = 0;
    for (let r = 1; r <= N_RANK; r++) v += sigmoid(a * (t - LN[r]));
    tbl[i] = v;
  }
  vocabTables.set(a, tbl);
  return tbl;
}

export function vocabOf(theta, a) {
  const tbl = vocabTable(a);
  const idx = (theta - THETA_LO) / THETA_STEP;
  if (idx <= 0) return tbl[0];
  if (idx >= N_GRID - 1) return tbl[N_GRID - 1];
  const i = Math.floor(idx), f = idx - i;
  return tbl[i] * (1 - f) + tbl[i + 1] * f;
}

export function thetaOf(vocab, a) {
  const tbl = vocabTable(a);
  let lo = 0, hi = N_GRID - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (tbl[m] < vocab) lo = m; else hi = m; }
  const frac = (vocab - tbl[lo]) / ((tbl[hi] - tbl[lo]) || 1);
  return thetaAt(lo) + frac * THETA_STEP;
}

// ---------- 极大似然 ----------
// Newton-Raphson 解 theta。全对/全错时逻辑斯谛似然没有有限极大值，
// 会一路跑飞出词表 —— 所以每步都夹住。
function mleTheta(a, items, start) {
  let t = start !== undefined ? start
    : items.reduce((s, it) => s + LN[it.rank], 0) / items.length;
  for (let i = 0; i < 40; i++) {
    let g = 0, h = 0;
    for (const it of items) {
      const p = sigmoid(a * (t - LN[it.rank]));
      g += it.correct - p;
      h += p * (1 - p);
    }
    if (h < 1e-9) break;
    const step = Math.max(-1.2, Math.min(1.2, g / (a * h)));
    t += step;
    t = Math.max(THETA_LO, Math.min(THETA_HI, t));
    if (Math.abs(step) < 1e-8) break;
  }
  return t;
}

function logLik(theta, a, items) {
  let s = 0;
  for (const it of items) {
    const p = Math.min(1 - 1e-12, Math.max(1e-12, sigmoid(a * (theta - LN[it.rank]))));
    s += it.correct ? Math.log(p) : Math.log(1 - p);
  }
  return s;
}

export function predict(theta, a, rank) {
  return sigmoid(a * (theta - LN[rank]));
}

// ---------- 渐近标准误 ----------
// SE(theta) = 1/sqrt(Σ a²·p(1-p))，再用 dV/dtheta 链式换算成 SE(V)
function seOf(theta, a, items) {
  let info = 0;
  for (const it of items) {
    const p = predict(theta, a, it.rank);
    info += a * a * p * (1 - p);
  }
  const seTheta = info > 1e-9 ? 1 / Math.sqrt(info) : THETA_HI - THETA_LO;
  const dV = (vocabOf(theta + 0.05, a) - vocabOf(theta - 0.05, a)) / 0.1;
  return { seTheta, seVocab: Math.abs(dV) * seTheta, info };
}

/**
 * 拟合一次作答记录。
 * @param {Array<{rank:number, correct:boolean|number}>} items
 * @param {object} [opts]
 * @param {number} [opts.fixA]        钉死 a（track 模式用）；不给则在 A_GRID 上搜
 * @param {number} [opts.priorTheta]  起始值（自适应测试中途重估时用，能加速收敛）
 * @returns {null|{theta,a,aFixed,ll,seVocab,vocab,ci95,shapeUncertain,nItems,nCorrect,atBound}}
 */
export function fit(items, opts = {}) {
  const usable = items.filter(it => it && Number.isFinite(it.rank) && it.rank >= 1 && it.rank <= N_RANK);
  if (usable.length === 0) return null;
  const y = usable.map(it => (it.correct ? 1 : 0));
  const nCorrect = y.reduce((s, v) => s + v, 0);

  let theta, a, ll, ci95, seVocab, vocab, shapeUncertain = false;
  const aFixed = opts.fixA !== undefined;

  if (aFixed) {
    a = opts.fixA;
    theta = mleTheta(a, usable, opts.priorTheta);
    ll = logLik(theta, a, usable);
    const { seTheta, seVocab: se } = seOf(theta, a, usable);
    seVocab = se;
    vocab = vocabOf(theta, a);
    ci95 = [
      Math.max(0, Math.round(vocabOf(theta - 1.96 * seTheta, a))),
      Math.min(N_RANK, Math.round(vocabOf(theta + 1.96 * seTheta, a))),
    ];
  } else {
    // a 经常认不出来（作答曲线平时，似然沿 a 是一条脊）。
    // "取网格最大似然"会被网格边界绑架（下界改一改总量就跳几百词），所以：
    // 点估计 = 轮廓似然加权的模型平均；区间 = 似然比集合(Δll≤1.92)内各 a 的渐近区间之并。
    const prof = A_GRID.map(cand => {
      const t = mleTheta(cand, usable, opts.priorTheta);
      return { a: cand, theta: t, ll: logLik(t, cand, usable), v: vocabOf(t, cand) };
    });
    const llmax = Math.max(...prof.map(p => p.ll));
    const ws = prof.map(p => Math.exp(p.ll - llmax));
    const wsum = ws.reduce((s, w) => s + w, 0);
    a = round(prof.reduce((s, p, i) => s + ws[i] * p.a, 0) / wsum, 2);
    theta = prof.reduce((s, p, i) => s + ws[i] * p.theta, 0) / wsum;
    vocab = prof.reduce((s, p, i) => s + ws[i] * p.v, 0) / wsum;
    ll = llmax;
    const S = prof.filter(p => p.ll >= llmax - 1.92);
    shapeUncertain = S.length > 1
      && (S[0].a === A_GRID[0] || S[S.length - 1].a === A_GRID[A_GRID.length - 1]);
    let lo = Infinity, hi = -Infinity;
    for (const p of S) {
      const { seTheta: se } = seOf(p.theta, p.a, usable);
      lo = Math.min(lo, vocabOf(p.theta - 1.96 * se, p.a));
      hi = Math.max(hi, vocabOf(p.theta + 1.96 * se, p.a));
    }
    ci95 = [Math.max(0, Math.round(lo)), Math.min(N_RANK, Math.round(hi))];
    seVocab = (ci95[1] - ci95[0]) / 3.92;
  }

  return {
    nItems: usable.length,
    nCorrect,
    theta: round(theta, 4),
    a,
    aFixed,
    ll: round(ll, 3),
    seVocab: Math.round(seVocab),
    vocab: Math.round(vocab),
    ci95,
    shapeUncertain,
    // 撞上词表边界 = 题目太简单/太难，这次测不准，报告里要如实说明
    atBound: theta <= THETA_LO + 1e-6 || theta >= THETA_HI - 1e-6
      || nCorrect === 0 || nCorrect === usable.length,
  };
}

const round = (v, n) => Math.round(v * 10 ** n) / 10 ** n;

// ---------- 自检 ----------
// 用已知的 theta/a 造虚拟作答，看能不能把它估回来。
// 单次抽样本来就有 5% 概率落空，所以覆盖率、偏差这类指标都要跑够次数才有意义。
async function selftest() {
  const REPS = Number(process.env.REPS || 60);
  const ws = loadWordlist();
  if (ws.length !== N_RANK) throw new Error(`词表长度 ${ws.length} ≠ ${N_RANK}`);

  const rnd = makeRnd(12345);

  let bad = 0;
  console.log(`每组 ${REPS} 次重复。level 模式：160 题均匀铺满全表，a 由作答拟合。`);
  console.log('真实a  真实词汇量 │ 词汇量偏差中位数   95%CI覆盖率   a估计中位数   a误差中位数');
  for (const trueA of [1.0, 2.5, 4.0, 6.0]) {
    for (const trueV of [3000, 5000, 8000, 12000]) {
      const trueTheta = thetaOf(trueV, trueA);
      const errs = [], aErrs = [], aEsts = [];
      let cover = 0;
      for (let rep = 0; rep < REPS; rep++) {
        const items = [];
        for (let k = 0; k < 160; k++) {
          const rank = 1 + Math.floor(rnd() * N_RANK);
          items.push({ rank, correct: rnd() < predict(trueTheta, trueA, rank) ? 1 : 0 });
        }
        const f = fit(items);
        errs.push(f.vocab - trueV);
        aEsts.push(f.a);
        aErrs.push(Math.abs(f.a - trueA));
        if (f.ci95[0] <= trueV && trueV <= f.ci95[1]) cover++;
      }
      const covRate = cover / REPS;
      const line = `${String(trueA).padStart(5)} ${String(trueV).padStart(10)} │`
        + `${String(median(errs).toFixed(0)).padStart(15)}`
        + `${(covRate * 100).toFixed(0).padStart(12)}%`
        + `${String(median(aEsts).toFixed(2)).padStart(13)}`
        + `${String(median(aErrs).toFixed(2)).padStart(13)}`;
      console.log(line);
      // 覆盖率低于 75% 说明 SE 被低估了，报告里的置信区间会骗人
      if (covRate < 0.75) { bad++; console.log('       ^ CI 覆盖率过低'); }
    }
  }

  // 极端：全对 / 全错，不能崩、不能跑飞出词表
  for (const [label, correct] of [['全对', 1], ['全错', 0]]) {
    const items = Array.from({ length: 40 }, (_, i) => ({ rank: 1 + i * 500, correct }));
    const f = fit(items);
    const ok = Number.isFinite(f.theta) && f.theta >= THETA_LO - 1e-3
      && f.theta <= THETA_HI + 1e-3 && f.atBound;
    if (!ok) bad++;
    console.log(`${label}: theta=${f.theta} vocab=${f.vocab} atBound=${f.atBound} ${ok ? '✓' : '✗'}`);
  }

  // track 模式：窄带（阈值之上 3000 词）+ 钉死 a
  {
    const trueA = 4.0, trueV = 6000;
    const trueTheta = thetaOf(trueV, trueA);
    const errs = [], covs = [];
    for (let rep = 0; rep < REPS; rep++) {
      const items = [];
      for (let k = 0; k < 100; k++) {
        const rank = trueV + Math.floor(rnd() * 3000);   // 阈值之上 3000 词
        items.push({ rank, correct: rnd() < predict(trueTheta, trueA, rank) ? 1 : 0 });
      }
      const f = fit(items, { fixA: trueA });
      errs.push(f.vocab - trueV);
      covs.push(f.ci95[0] <= trueV && trueV <= f.ci95[1] ? 1 : 0);
      if (!f.aFixed) bad++;
    }
    const cov = covs.reduce((s, v) => s + v, 0) / REPS;
    console.log(`\ntrack 模式：100 题挤在阈值之上 3000 词的窄带里，a 钉死在 ${trueA}`);
    console.log(`  词汇量偏差中位数 ${median(errs).toFixed(0)}   95%CI覆盖率 ${(cov * 100).toFixed(0)}%`);
    if (cov < 0.75) { bad++; console.log('  ^ CI 覆盖率过低'); }
  }

  console.log(bad === 0 ? '\n自检通过' : `\n自检失败 ${bad} 项`);
  process.exit(bad === 0 ? 0 : 1);
}

if (process.argv[2] === '--selftest' && isMain(import.meta.url)) await selftest();
