#!/usr/bin/env node
// vocab-size · 报告的四个维度
//
// 关键口径（设计时定死的）：
//   **总量只用首考词**。复现词（90 天冷却期之后又碰到的）不进总量，
//   单独算三个记忆指标。否则"背下了上次的答案"会把词汇量越测越高。
//
// 掌握曲线和考试标签维度都不数"这次答对了几题"——一次测试每段只有几题，
// 噪声压过信号。改成用拟合出来的曲线去推算整张词表：
//   第 k 段认识多少词 = Σ_{r∈段} P(认识|r)，其中 P 来自 theta 和 a。
// 这样 160 道题的信息全部用上了，而不是只用落在这一段的那 8 道。

import { fit, predict, DEFAULT_A } from './estimate.mjs';
import { memoryMetrics } from './state.mjs';
import { isMain } from './util.mjs';
import { TARGET_SIZE } from './wordlist.mjs';

export const BAND_SIZE = 1000;
export const EXAM_TAGS = ['zk', 'gk', 'cet4', 'cet6', 'ky', 'toefl', 'ielts', 'gre'];
export const TAG_LABEL = {
  zk: '中考', gk: '高考', cet4: '四级', cet6: '六级',
  ky: '考研', toefl: '托福', ielts: '雅思', gre: 'GRE',
};

/** 整张词表按 1000 词一段切开，用拟合曲线推算每段认识多少词。 */
export function masteryCurve(theta, a) {
  const out = [];
  for (let lo = 1; lo <= TARGET_SIZE; lo += BAND_SIZE) {
    const hi = Math.min(lo + BAND_SIZE - 1, TARGET_SIZE);
    let known = 0;
    for (let r = lo; r <= hi; r++) known += predict(theta, a, r);
    out.push({
      lo, hi,
      total: hi - lo + 1,
      known: Math.round(known),
      frac: Math.round((known / (hi - lo + 1)) * 100),
    });
  }
  return out;
}

/** 每个考试标签下，词表里有多少词、推算认识多少。 */
export function tagBreakdown(theta, a, wordlist) {
  const out = {};
  for (const t of EXAM_TAGS) out[t] = { label: TAG_LABEL[t], total: 0, known: 0 };
  for (const w of wordlist) {
    const p = predict(theta, a, w.rank);
    for (const t of (w.tag || '').split(/\s+/)) {
      if (out[t]) { out[t].total++; out[t].known += p; }
    }
  }
  for (const t of EXAM_TAGS) {
    const o = out[t];
    o.known = Math.round(o.known);
    o.frac = o.total ? Math.round((o.known / o.total) * 100) : null;
  }
  return out;
}

export function trendFrom(history) {
  return history
    .filter(r => r.complete && r.estimate?.vocab != null)
    .map(r => ({
      date: (r.startedAt || '').slice(0, 10),
      mode: r.mode,
      vocab: r.estimate.vocab,
      ci95: r.estimate.ci95,
      theta: r.estimate.theta,
      a: r.estimate.a,
    }));
}

/**
 * @param {object} o
 * @param {object} o.sitting   sampler.finish() 的结果，items 已标注 firstExposure/wasCorrect
 * @param {object[]} o.history 不含本次
 * @param {object[]} o.wordlist
 */
export function buildReport({ sitting, history, wordlist }) {
  const items = sitting.items;
  const first = items.filter(it => it.firstExposure);
  const repeated = items.filter(it => !it.firstExposure);

  // 总量只用首考词
  const fitFirst = fit(first, { fixA: sitting.fixA });
  const fitAll = fit(items, { fixA: sitting.fixA });

  const theta = fitFirst?.theta ?? fitAll?.theta ?? null;
  const a = fitFirst?.a ?? fitAll?.a ?? DEFAULT_A;

  const prev = history.filter(r => r.complete).slice(-1)[0] || null;
  const prevVocab = prev?.estimate?.vocab ?? null;

  // track 模式：窄带命中率的绝对变化，比总量数字灵敏得多
  let band = null;
  if (sitting.band && fitFirst) {
    const { lo, hi } = sitting.band;
    const n = items.length;
    const hits = items.filter(it => it.correct).length;
    const rate = n ? hits / n : 0;
    // 这次窄带认识多少词（模型推算），对比上次在同一区间上的推算
    let knownNow = 0, knownPrev = 0;
    for (let r = lo; r <= hi; r++) {
      knownNow += predict(fitFirst.theta, a, r);
      if (prev?.estimate?.theta != null) {
        knownPrev += predict(prev.estimate.theta, prev.estimate.a || a, r);
      }
    }
    const prevBand = prev?.band && prev.band.lo === lo && prev.band.hi === hi ? prev.band : null;
    band = {
      lo, hi, width: hi - lo + 1,
      items: n, hits, rate: Math.round(rate * 100),
      knownNow: Math.round(knownNow),
      // 只有上次测的是同一个带，delta 才有意义
      deltaKnown: prevBand ? Math.round(knownNow - prevBand.knownNow) : null,
      deltaRate: prevBand ? Math.round(rate * 100 - prevBand.rate) : null,
      comparable: !!prevBand,
    };
  }

  return {
    mode: sitting.mode,
    headline: fitFirst ? {
      vocab: fitFirst.vocab,
      ci95: fitFirst.ci95,
      seVocab: fitFirst.seVocab,
      theta: fitFirst.theta,
      a: fitFirst.a,
      aFixed: fitFirst.aFixed,
      atBound: fitFirst.atBound,
      nItems: fitFirst.nItems,
      nCorrect: fitFirst.nCorrect,
      nFirstExposure: first.length,
      nRepeated: repeated.length,
      deltaVocab: prevVocab != null ? fitFirst.vocab - prevVocab : null,
      deltaVs: prev?.estimate ? `${prev.startedAt.slice(0, 10)}` : null,
    } : null,
    // 首考词太少时总量不可信，页面要显式提醒
    thinSample: first.length < 30,
    mastery: theta != null ? masteryCurve(theta, a) : null,
    tags: theta != null ? tagBreakdown(theta, a, wordlist) : null,
    // 趋势要把本次也算进去，不然结果页上最后一个点永远是上一次的
    trend: [...trendFrom(history), ...(fitFirst ? [{
      date: new Date().toISOString().slice(0, 10),
      mode: sitting.mode,
      vocab: fitFirst.vocab,
      ci95: fitFirst.ci95,
      theta: fitFirst.theta,
      a: fitFirst.a,
    }] : [])],
    wrong: items.filter(it => !it.correct).map(it => {
      const w = wordlist[it.rank - 1];
      return { rank: it.rank, word: it.word, phonetic: w?.phonetic, tag: w?.tag, glossLines: w?.glossLines };
    }),
    memory: memoryMetrics(items),
    band,
    // 尺子量程：这个词表只有 COCA 频率前 2 万词，母语成人（2万–3.5万）已经超量程，
    // 所以接近 2 万的读数是被压扁的。报告必须把这句说出来，不能让数字自己装准。
    scale: { max: TARGET_SIZE, saturated: (fitFirst?.vocab ?? 0) > TARGET_SIZE * 0.8 },
  };
}

/** 给对话用的纯文字摘要 —— SKILL.md 会让 agent 把这段转述给用户。 */
export function textSummary(rep) {
  const L = [];
  const h = rep.headline;
  if (!h) return '这次测试没有可用的估算结果（答题数太少）。';

  L.push(rep.mode === 'level'
    ? `【level 全量测试】词汇量约 ${h.vocab} 词（95% 区间 ${h.ci95[0]}–${h.ci95[1]}，±${h.seVocab}）`
    : `【track 窄带追踪】词汇量约 ${h.vocab} 词（95% 区间 ${h.ci95[0]}–${h.ci95[1]}）`);

  if (h.atBound) L.push('⚠ 这次撞到了词表边界（几乎全对或全错），数字不可信，建议调整难度重测。');
  if (rep.thinSample) L.push(`⚠ 首考词只有 ${h.nFirstExposure} 个（其余是复现词），总量估算偏粗。`);

  L.push(`陡峭度 a = ${h.a}${h.aFixed ? '（沿用上次 level 的测量值）' : '（本次拟合）'}。a 越大会的越干脆，a 小则半生不熟的词多。`);

  if (rep.band) {
    const b = rep.band;
    L.push(`窄带 [第 ${b.lo}–${b.hi} 名]：答对 ${b.hits}/${b.items}（${b.rate}%），推算这段认识约 ${b.knownNow}/${b.width} 词。`);
    if (b.comparable && b.deltaKnown != null) {
      L.push(`与上次同一个带相比：${b.deltaKnown >= 0 ? '+' : ''}${b.deltaKnown} 词，命中率 ${b.deltaRate >= 0 ? '+' : ''}${b.deltaRate} 个百分点。`);
    } else {
      L.push('（上次测的不是同一个带，无法直接比。窄带要连续测同一段才有趋势。）');
    }
  } else if (h.deltaVocab != null) {
    L.push(`与上次（${h.deltaVs}）相比：${h.deltaVocab >= 0 ? '+' : ''}${h.deltaVocab} 词。`
      + `注意：这个差值的测量噪声约 ±${h.seVocab * 1.4 | 0} 词，小于这个量级的变化看不出来。`);
  }

  if (rep.memory.repeated > 0) {
    const m = rep.memory;
    L.push(`长期记忆（${m.repeated} 个 90 天前见过的复现词）：保持率 ${m.retention}%、遗忘率 ${m.forgetting}%、挽回率 ${m.recovery}%。`);
  }

  if (rep.tags) {
    const line = Object.values(rep.tags)
      .filter(t => t.total >= 200)
      .map(t => `${t.label} ${t.frac}%`)
      .join(' · ');
    L.push(`考试词覆盖率：${line}`);
  }

  const curve = rep.mastery?.filter(b => b.frac >= 5 && b.frac <= 95);
  if (curve?.length) L.push(`掌握曲线的过渡区在第 ${curve[0].lo}–${curve[curve.length - 1].hi} 名之间，这是最该投入的区间。`);

  if (rep.wrong.length) {
    L.push(`答错 ${rep.wrong.length} 词：${rep.wrong.slice(0, 12).map(w => w.word).join('、')}${rep.wrong.length > 12 ? ' …' : ''}`);
  }
  return L.join('\n');
}

if (process.argv[2] === '--demo' && isMain(import.meta.url)) {
  const { loadWordlist } = await import('./wordlist.mjs');
  const ws = loadWordlist();
  const rep = buildReport({
    sitting: {
      mode: 'level', fixA: undefined, band: null,
      items: Array.from({ length: 160 }, (_, i) => ({
        rank: 1 + i * 125, word: ws[i * 125]?.word, correct: i * 125 < 6000,
        firstExposure: true, choice: 0, ms: 3000,
      })),
    },
    history: [], wordlist: ws,
  });
  console.log(textSummary(rep));
  console.log('\n掌握曲线（每 1000 词一段）:');
  for (const b of rep.mastery) console.log(`  ${String(b.lo).padStart(5)}–${String(b.hi).padEnd(5)} ${'█'.repeat(Math.round(b.frac / 5)).padEnd(20)} ${String(b.frac).padStart(3)}%  ${b.known}词`);
}
