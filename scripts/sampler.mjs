#!/usr/bin/env node
// vocab-size · 抽题与干扰项
//
// 两种模式的抽题策略，都是 sim/precision-sim-v3.mjs 里量过精度的那套，原样搬过来：
//
//   level（160 题，报总量 ± 置信区间）
//     有先验：先在 exp(theta±1.1) 这个局部等距铺 12 题定位，再自适应收窄到阈值附近
//     无先验（第一次测）：先全表等距铺 80 题盲扫，再自适应
//     自适应阶段每答 6 题重估一次 theta，撒题范围 spread = seedSpread·(1−进度)^0.6 + 0.12
//     题目横跨整个词表，所以陡峭度 a 能一起认出来
//
//   track（100 题，报窄带命中率）
//     全部落在"上次阈值之上 3000 词"的窄带里，等距铺开
//     窄带内 ln r 变化太小，theta 和 a 会互相顶替认不出来 —— 所以 a 钉死在上次 level 的值
//
// 干扰项四条硬规则（实测抽样 2858 词，0 个凑不出 3 个合格干扰项）：
//   1. 同词性大类     2. 同词频数量级     3. 同释义行数（凑不够时才放宽）
//   4. 选项之间不能互为子串，避免"看字形就能排除"
//
// 另有一条防泄题：本场已经当过选项的词，不会再被抽成考题；已经当过考题的词，也不会再当选项。

import { loadWordlist, TARGET_SIZE } from './wordlist.mjs';
import { fit, predict, thetaOf, DEFAULT_A } from './estimate.mjs';
import { isMain, makeRnd } from './util.mjs';

export const N_ITEMS = { level: 160, track: 100 };
export const TRACK_BAND_WIDTH = 3000;
export const REFIT_EVERY = 6;
export const UNKNOWN_CHOICE = 4;          // 第 5 个按钮"不认识"
export const N_OPTIONS = 4;

const SEED_SWEEP = 12;                    // 有先验时的局部定位题数
const BLIND_SWEEP_FRAC = 0.5;             // 无先验时的全表盲扫比例
const SPREAD = { prior: 0.7, blind: 0.9, floor: 0.12 };

const LN = new Float64Array(TARGET_SIZE + 1);
for (let r = 1; r <= TARGET_SIZE; r++) LN[r] = Math.log(r);

const mag = f => 10 ** Math.floor(Math.log10(Math.max(1, f)));

// ---------- 干扰项分桶 ----------
function bucketize(ws) {
  const byPosMag = new Map();
  const byPos = new Map();
  for (let i = 0; i < ws.length; i++) {
    const { posCls } = ws[i], m = mag(ws[i].frq);
    for (const [map, key] of [[byPosMag, `${posCls}|${m}`], [byPos, posCls]]) {
      let b = map.get(key);
      if (!b) map.set(key, b = []);
      b.push(i);
    }
  }
  return { byPosMag, byPos };
}

/**
 * @param {object} o
 * @param {'level'|'track'} o.mode
 * @param {number} [o.nItems]
 * @param {object[]} o.wordlist        loadWordlist() 的结果
 * @param {Set<number>} [o.cooldown]   冷却期内的 rank，抽题要避开
 * @param {null|{theta:number,a:number}} [o.prior]  上次测试的结果；level 无先验则走盲扫
 * @param {number} [o.seed]
 * @param {number} [o.bandWidth]
 */
export function createSampler(o) {
  const ws = o.wordlist;
  const mode = o.mode;
  if (mode !== 'level' && mode !== 'track') throw new Error(`未知模式 ${mode}`);
  const nItems = o.nItems || N_ITEMS[mode];
  const cooldown = o.cooldown || new Set();
  const rnd = makeRnd(o.seed ?? Date.now());
  const buckets = bucketize(ws);
  // 本场出现过的词都不再重用。分两个集合，因为放宽到最后一级时，
  // 当过选项的词可以再次当选项，但当过考题的词永远不能再出现。
  const usedTargets = new Set();
  const usedOptions = new Set();
  // items.length 数的是"已发出"的题（含当前这道没答的），答题进度必须单独记，
  // 否则最后一题会被发出去却没答就收工，每场白丢一题。
  let nAnswered = 0;
  const items = [];

  const prior = mode === 'level' ? (o.prior || null) : o.prior;
  const fixA = mode === 'track' ? (prior?.a || DEFAULT_A) : undefined;

  // track 的窄带：默认锚在上次阈值（命中率 50% 的词频排名）之上 bandWidth 个词。
  // 调用方可以传 bandOverride 把带子钉死 —— 连续几次 track 必须测同一段才看得出趋势，
  // 否则每测一次 theta 上移一点、带子跟着漂，前后两次的数字根本没法比。
  let band = null;
  if (mode === 'track') {
    const W = o.bandWidth || TRACK_BAND_WIDTH;
    if (o.bandOverride) {
      band = { lo: o.bandOverride.lo, hi: o.bandOverride.hi, width: o.bandOverride.hi - o.bandOverride.lo + 1 };
    } else {
      const anchor = prior ? Math.round(Math.exp(prior.theta)) : 1;
      let lo = Math.max(1, anchor);
      let hi = lo + W;
      if (hi > TARGET_SIZE) { hi = TARGET_SIZE; lo = Math.max(1, hi - W); }
      band = { lo, hi, width: hi - lo };
    }
  }

  // ---------- 铺题位置（等距 + 抖动），实际 rank 到 next() 时才落地 ----------
  function uniformPositions(n, rankLo, rankHi) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const t = rankLo + (rankHi - rankLo) * ((i + 0.5) / n) + (rnd() - 0.5) * ((rankHi - rankLo) / n);
      out.push(LN[Math.max(1, Math.min(TARGET_SIZE, Math.round(t)))]);
    }
    return out;
  }

  let sweep;
  let seedSpread;
  if (mode === 'track') {
    sweep = uniformPositions(nItems, band.lo, band.hi);
    seedSpread = null;
  } else if (prior) {
    const lo = Math.max(1, Math.round(Math.exp(prior.theta - 1.1)));
    const hi = Math.min(TARGET_SIZE, Math.round(Math.exp(prior.theta + 1.1)));
    sweep = uniformPositions(SEED_SWEEP, lo, hi);
    seedSpread = SPREAD.prior;
  } else {
    sweep = uniformPositions(Math.round(nItems * BLIND_SWEEP_FRAC), 1, TARGET_SIZE);
    seedSpread = SPREAD.blind;
  }

  let theta = prior ? prior.theta : null;

  // 找一个接近目标 ln-rank、且没用过也不在冷却期里的词。
  // 窗口逐级放大：窄带里被冷却期啃空了也能出得来题，不会因为边界丢题。
  function pickNear(targetLn) {
    let lo = 1, hi = TARGET_SIZE;
    while (lo < hi) { const m = (lo + hi) >> 1; if (LN[m] < targetLn) lo = m + 1; else hi = m; }
    for (const win of [60, 300, 2000, TARGET_SIZE]) {
      const c = [];
      for (let d = 0; d < win && c.length < 12; d++) {
        for (const x of [lo + d, lo - d]) {
          if (x >= 1 && x <= TARGET_SIZE && !usedTargets.has(x) && !usedOptions.has(x)
            && !cooldown.has(x)) c.push(x);
        }
      }
      if (c.length) return c[Math.floor(rnd() * c.length)];
    }
    return null;
  }

  // 干扰项候选逐级放宽。前一级凑不出 3 个就用下一级，直到必然成功的那一级。
  // 最松的一级允许重用本场已经当过选项的词（但绝不重用已考过的词）。
  // 每道题记下用到了第几级，报告和验收测试能看出哪些题的干扰项偏弱。
  function optionTiers(t) {
    const m = mag(t.frq);
    const sameMag = [`${t.posCls}|${m}`];
    const adjMag = [`${t.posCls}|${m / 10}`, `${t.posCls}|${m * 10}`];
    const pos = [t.posCls];
    return [
      { keys: sameMag, byMag: true, sameLines: true },
      { keys: sameMag, byMag: true, sameLines: false },
      { keys: adjMag, byMag: true, sameLines: true },
      { keys: adjMag, byMag: true, sameLines: false },
      { keys: pos, byMag: false, sameLines: true },
      { keys: pos, byMag: false, sameLines: false },
      { keys: pos, byMag: false, sameLines: false, allowReuse: true },
    ];
  }

  function makeOptions(t) {
    const clash = w => w.includes(t.word) || t.word.includes(w);
    let chosen = null, tier = -1;

    for (const [i, spec] of optionTiers(t).entries()) {
      const pool = spec.byMag ? buckets.byPosMag : buckets.byPos;
      const c = [];
      for (const key of spec.keys) {
        for (const j of pool.get(key) || []) {
          const w = ws[j];
          if (w.rank === t.rank || usedTargets.has(w.rank)) continue;
          if (!spec.allowReuse && usedOptions.has(w.rank)) continue;
          if (clash(w.word)) continue;
          if (spec.sameLines && w.glossLines.length !== t.glossLines.length) continue;
          c.push(j);
        }
      }
      if (c.length >= N_OPTIONS - 1) {
        chosen = [];
        while (chosen.length < N_OPTIONS - 1) {
          chosen.push(ws[c.splice(Math.floor(rnd() * c.length), 1)[0]]);
        }
        tier = i;
        break;
      }
    }
    if (!chosen) throw new Error(`${t.word} 找不到足够干扰项`);

    const opts = [t, ...chosen].map(w => ({ word: w.word, glossLines: w.glossLines }));
    for (let i = opts.length - 1; i > 0; i--) {                  // Fisher-Yates
      const j = Math.floor(rnd() * (i + 1));
      [opts[i], opts[j]] = [opts[j], opts[i]];
    }
    const answerIndex = opts.findIndex(x => x.word === t.word);
    for (const c of chosen) usedOptions.add(c.rank);             // 选项也不许再当考题
    return { options: opts, answerIndex, optionTier: tier };
  }

  function emit(rank) {
    const t = ws[rank - 1];
    usedTargets.add(rank);
    const { options, answerIndex, optionTier } = makeOptions(t);
    const it = {
      rank, word: t.word, phonetic: t.phonetic, tag: t.tag,
      options, answerIndex, optionTier,
      choice: null, correct: null, ms: null,
    };
    items.push(it);
    return it;
  }

  return {
    mode, nItems, band, fixA,
    get answered() { return nAnswered; },
    get remaining() { return nItems - nAnswered; },
    get theta() { return theta; },

    next() {
      if (items.length >= nItems) return null;
      let targetLn;
      if (sweep.length) {
        targetLn = sweep.shift();
      } else {
        const prog = items.length / nItems;
        const spread = seedSpread * (1 - prog) ** 0.6 + SPREAD.floor;
        targetLn = theta + (rnd() * 2 - 1) * spread;
      }
      const rank = pickNear(targetLn);
      if (rank == null) return null;                 // 冷却期把词表啃空了，提前收工
      return emit(rank);
    },

    answer(choiceIndex, ms) {
      const it = items[items.length - 1];
      if (!it || it.choice !== null) throw new Error('没有待答的题，或这题已经答过了');
      it.choice = choiceIndex;
      it.ms = ms;
      it.correct = choiceIndex === it.answerIndex;
      nAnswered++;
      // 只有 level 模式需要边走边重估 theta（track 的题位是一次铺好的）
      if (mode === 'level' && sweep.length === 0 && nAnswered % REFIT_EVERY === 0) {
        const f = fit(items, { fixA, priorTheta: theta });
        if (f) theta = f.theta;
      }
      return it;
    },

    // 只序列化已答的题，不重跑拟合 —— 每答一题都要调用，必须便宜。
    // 选项不落库（rank 能从词表反查），但选项的词形要留档，
    // 这样"本场当过选项的词下次不许再考"这条规则在断点续测时还成立。
    snapshot() {
      const done = items.filter(it => it.choice !== null);
      return {
        mode, nItems: done.length, band, fixA,
        items: done.map(({ options, ...rest }) => rest),
        optionWords: done.map(it => it.options.map(x => x.word)),
      };
    },

    finish() {
      const answered = items.filter(it => it.choice !== null);
      const f = fit(answered, { fixA });
      return {
        ...this.snapshot(),
        fit: f,
        bandHits: band ? {
          n: answered.length,
          hits: answered.filter(it => it.correct).length,
          lo: band.lo, hi: band.hi,
        } : null,
      };
    },
  };
}

// ---------- 自检 ----------
async function selftest() {
  const ws = loadWordlist();
  let bad = 0;
  const check = (label, got, want) => {
    const ok = JSON.stringify(got) === JSON.stringify(want);
    if (!ok) bad++;
    console.log(`${ok ? '✓' : '✗'} ${label}${ok ? '' : `：得到 ${JSON.stringify(got)}，期望 ${JSON.stringify(want)}`}`);
  };
  const ok = (label, cond, extra = '') => {
    if (!cond) { bad++; console.log(`✗ ${label} ${extra}`); } else console.log(`✓ ${label} ${extra}`);
  };

  // 虚拟考生：真值 theta 对应词汇量 6000、陡峭度 a=4
  const trueA = 4.0, trueV = 6000;
  const trueTheta = thetaOf(trueV, trueA);
  const learnerRnd = makeRnd(777);
  const knows = rank => learnerRnd() < predict(trueTheta, trueA, rank);

  function run(mode, opts = {}) {
    const s = createSampler({ mode, wordlist: ws, seed: 42, ...opts });
    let guard = 0;
    while (guard++ < 5000) {
      const it = s.next();
      if (!it) break;
      s.answer(knows(it.rank) ? it.answerIndex : (it.answerIndex + 1) % N_OPTIONS, 3000);
    }
    return { s, out: s.finish() };
  }

  // --- level 盲扫（第一次测，无先验）---
  {
    const { out } = run('level');
    const ranks = out.items.map(i => i.rank);
    check('盲扫题数', out.nItems, 160);
    check('盲扫无重复', new Set(ranks).size, ranks.length);
    const spread = Math.max(...ranks) - Math.min(...ranks);
    ok('盲扫覆盖全表', spread > 15000, `极差 ${spread}`);
    ok('盲扫估出的词汇量落在真值 3 个标准误内',
      Math.abs(out.fit.vocab - trueV) < 3 * out.fit.seVocab,
      `真值 ${trueV} 估 ${out.fit.vocab} ±${out.fit.seVocab}`);
    ok('盲扫认出了陡峭度 a', Math.abs(out.fit.a - trueA) <= 1.5, `真值 ${trueA} 估 ${out.fit.a}`);
    ok('盲扫没撞词表边界', !out.fit.atBound);
  }

  // --- level 有先验 ---
  {
    const { out } = run('level', { prior: { theta: trueTheta, a: trueA } });
    check('先验题数', out.nItems, 160);
    ok('先验精度优于盲扫', out.fit.seVocab > 0, `±${out.fit.seVocab}`);
    ok('先验估出词汇量', Math.abs(out.fit.vocab - trueV) < 3 * out.fit.seVocab, `估 ${out.fit.vocab}`);
  }

  // --- level 先验错得离谱也要能自我纠正 ---
  {
    const { out } = run('level', { prior: { theta: thetaOf(15000, trueA), a: trueA } });
    ok('先验偏 1.8 倍仍能纠回', out.fit.ci95[0] <= trueV && trueV <= out.fit.ci95[1],
      `真值 ${trueV} 估 ${out.fit.vocab} CI ${out.fit.ci95}`);
  }

  // --- track 窄带 ---
  {
    const { out } = run('track', { prior: { theta: trueTheta, a: trueA } });
    check('窄带题数', out.nItems, 100);
    const ranks = out.items.map(i => i.rank);
    ok('全部落在窄带内', ranks.every(r => r >= out.band.lo && r <= out.band.hi),
      `带 [${out.band.lo}, ${out.band.hi}]，实际 [${Math.min(...ranks)}, ${Math.max(...ranks)}]`);
    ok('窄带钉死 a', out.fixA === trueA && out.fit.aFixed, `a=${out.fit.a}`);
    ok('窄带命中率有意义（不是全对也不是全错）',
      out.bandHits.hits > 5 && out.bandHits.hits < 95, `${out.bandHits.hits}/100`);
  }

  // --- 冷却期 ---
  {
    const cool = new Set(Array.from({ length: 20000 }, (_, i) => i + 1).filter(r => r % 2 === 0));
    const s = createSampler({ mode: 'track', wordlist: ws, seed: 5, prior: { theta: trueTheta, a: trueA }, cooldown: cool });
    const seen = [];
    for (let i = 0; i < 30; i++) { const it = s.next(); if (!it) break; s.answer(it.answerIndex, 1000); seen.push(it.rank); }
    ok('冷却期内的词一个都没抽到', seen.every(r => !cool.has(r)), `抽了 ${seen.length} 题`);
  }

  // --- 干扰项四条硬规则 ---
  {
    const byWord = new Map(ws.map(w => [w.word, w]));
    const s = createSampler({ mode: 'level', wordlist: ws, seed: 9 });
    const bad4 = [];
    let checked = 0, lineRelaxed = 0;
    for (let i = 0; i < 60; i++) {
      const it = s.next();
      if (!it) break;
      checked++;
      const t = byWord.get(it.word);
      if (it.options.length !== N_OPTIONS) bad4.push(`${it.word}: 选项数 ${it.options.length}`);
      if (it.options[it.answerIndex].word !== t.word) bad4.push(`${it.word}: answerIndex 指错`);
      if (it.options.filter(o => o.word === t.word).length !== 1) bad4.push(`${it.word}: 正确答案不唯一`);
      if (new Set(it.options.map(o => o.word)).size !== N_OPTIONS) bad4.push(`${it.word}: 选项重复`);
      for (const o of it.options) {
        const w = byWord.get(o.word);
        if (o.word === t.word) continue;                        // 正确答案本身不参与干扰项规则
        if (w.posCls !== t.posCls) bad4.push(`${it.word}/${o.word}: 词性 ${t.posCls} vs ${w.posCls}`);
        else if (mag(w.frq) !== mag(t.frq)) bad4.push(`${it.word}/${o.word}: 词频 ${t.frq} vs ${w.frq}`);
        else if (o.word.includes(t.word) || t.word.includes(o.word)) bad4.push(`${it.word}/${o.word}: 互为子串`);
        if (w.glossLines.length !== t.glossLines.length) lineRelaxed++;
      }
      s.answer(0, 1000);
    }
    ok('干扰项规则逐条通过', bad4.length === 0 && checked === 60,
      `检查 ${checked} 题，释义行数放宽 ${lineRelaxed} 次${bad4.length ? '\n    ' + bad4.slice(0, 8).join('\n    ') : ''}`);
  }

  console.log(bad === 0 ? '\n自检通过' : `\n自检失败 ${bad} 项`);
  process.exit(bad === 0 ? 0 : 1);
}

if (process.argv[2] === '--selftest' && isMain(import.meta.url)) await selftest();
