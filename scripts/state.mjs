#!/usr/bin/env node
// vocab-size · 状态、历史与冷却期
//
// 状态目录 ~/.vocab-test/（可用 VOCAB_TEST_STATE_DIR 覆盖）：
//   wordlist.tsv    词表（尺子），由 wordlist.mjs build 生成
//   history.jsonl   一行一次测试，append-only
//   .cache/         原始 ECDICT csv，提炼完可删；永远不进 git
//
// 历史是这个 skill 的全部记忆。三件事依赖它：
//   1. 90 天冷却期 —— 上次测过的词短期内不再出现（避免"背答案"把词汇量测高）
//   2. 三个记忆指标 —— 复现词的保持率 / 挽回率 / 遗忘率
//   3. level 模式的先验 —— 上次的 theta 决定这次从哪儿开始出题
//
// 冷却期是 90 天，所以"复现词"必然是 90 天以前见过的。
// 这三个指标量的是**长期记忆**，不是昨天刚背的东西。这是设计如此，报告里要说明。
//
// 约定：所有统计逻辑都是 (history 数组) -> 结果 的纯函数，文件读写只在最外面一层。
// 这样自检不需要碰磁盘，也不会误伤真实历史。

import { existsSync, readFileSync, appendFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { STATE_DIR, WORDLIST, CACHE_DIR } from './wordlist.mjs';
import { isMain } from './util.mjs';

export const HISTORY = join(STATE_DIR, 'history.jsonl');
export const COOLDOWN_DAYS = 90;
export const SCHEMA_VERSION = 1;
const DAY = 86400000;

export { STATE_DIR, WORDLIST, CACHE_DIR };

// ---------- 纯逻辑 ----------

/**
 * 最近 days 天内出过的词的 rank 集合，抽题时必须避开。
 * 未完成的测试也算 —— 题面已经见过了，再出就是背答案。
 */
export function computeCooldown(history, days = COOLDOWN_DAYS, until = Date.now()) {
  const cutoff = until - days * DAY;
  const seen = new Set();
  for (const rec of history) {
    const when = Date.parse(rec.startedAt || rec.finishedAt || '');
    if (!Number.isFinite(when) || when < cutoff) continue;
    for (const it of rec.items || []) seen.add(it.rank);
  }
  return seen;
}

/** 只有做完的测试才进趋势；中途关页面的标了 complete=false，不算。 */
export function onlyComplete(history) {
  return history.filter(r => r.complete);
}

// 只统计复现词（firstExposure=false）。首考词进总量，复现词进这三个数。
//   保持率 retention = 上次对、这次还对 / 上次对
//   遗忘率 forgetting= 上次对、这次错了 / 上次对
//   挽回率 recovery  = 上次错、这次对了 / 上次错
export function memoryMetrics(items) {
  const rep = items.filter(it => !it.firstExposure);
  const wasRight = rep.filter(it => it.wasCorrect);
  const wasWrong = rep.filter(it => !it.wasCorrect);
  const kept = wasRight.filter(it => it.correct).length;
  const regained = wasWrong.filter(it => it.correct).length;
  const pct = (n, d) => (d ? Math.round((n / d) * 100) : null);
  return {
    repeated: rep.length,
    firstExposure: items.length - rep.length,
    retention: pct(kept, wasRight.length),
    forgetting: pct(wasRight.length - kept, wasRight.length),
    recovery: pct(regained, wasWrong.length),
    nWasRight: wasRight.length,
    nWasWrong: wasWrong.length,
  };
}

/** 给抽出来的题标上 firstExposure / wasCorrect —— 落库前调用一次。 */
export function annotateExposure(items, history) {
  const seen = new Map();
  for (const rec of history) {
    for (const it of rec.items || []) seen.set(it.rank, !!it.correct);
  }
  return items.map(it => {
    const before = seen.get(it.rank);
    return { ...it, firstExposure: before === undefined, wasCorrect: before };
  });
}

/** 最近一次可用的测试（默认不分模式）。track 的窄带位置和 level 的先验都从它来。 */
export function pickLast(history, mode) {
  const h = onlyComplete(history).filter(r => !mode || r.mode === mode);
  return h.length ? h[h.length - 1] : null;
}

/** 尺子换了版本，老历史里的 rank 就指向别的词了 —— 必须挡下来。 */
export function findDrift(history, currentSha) {
  return onlyComplete(history).filter(r => r.wordlistSha && r.wordlistSha !== currentSha);
}

// ---------- 文件层 ----------
export function ensureDirs() {
  mkdirSync(STATE_DIR, { recursive: true });
  mkdirSync(CACHE_DIR, { recursive: true });
}

export function wordlistSha() {
  if (!existsSync(WORDLIST)) return null;
  return createHash('sha256').update(readFileSync(WORDLIST)).digest('hex').slice(0, 16);
}

export function loadHistory() {
  if (!existsSync(HISTORY)) return [];
  const out = [];
  for (const [i, line] of readFileSync(HISTORY, 'utf8').split('\n').entries()) {
    if (!line.trim()) continue;
    try { out.push(JSON.parse(line)); }
    catch { console.error(`history.jsonl 第 ${i + 1} 行解析失败，已跳过`); }
  }
  return out;
}

export function appendSitting(rec) {
  ensureDirs();
  appendFileSync(HISTORY, JSON.stringify({ v: SCHEMA_VERSION, wordlistSha: wordlistSha(), ...rec }) + '\n');
  return rec;
}

export const cooldownSet = (days, until) => computeCooldown(loadHistory(), days, until);
export const completedHistory = () => onlyComplete(loadHistory());
export const lastSitting = mode => pickLast(loadHistory(), mode);

// ---------- 分仓同步 ----------
// 代码仓公开、状态仓私有。冲突**绝不自动合并** —— 停下来问用户哪台机器的账本更新。
export class SyncConflict extends Error {
  constructor(detail) {
    super(`状态仓同步冲突，需要你决定保留哪一边。${detail}`);
    this.name = 'SyncConflict';
  }
}

function git(...args) {
  return execFileSync('git', ['-C', STATE_DIR, ...args], { encoding: 'utf8' });
}
function gitOk(...args) {
  try { git(...args); return true; } catch { return false; }
}
function commitLocal(msg) {
  git('add', '-A');
  if (gitOk('diff', '--cached', '--exit-code')) return false;   // 无改动
  git('commit', '-m', msg);
  return true;
}

export function syncPull() {
  if (!existsSync(join(STATE_DIR, '.git'))) return { skipped: '状态目录还不是 git 仓库，跳过 pull' };
  commitLocal(`chore: 本地账本快照 ${new Date().toISOString().slice(0, 16).replace('T', ' ')}`);
  if (!gitOk('pull', '--ff-only')) {
    throw new SyncConflict('pull 无法快进，说明两台机器各自都写了历史。请人工比对 history.jsonl 后决定保留哪一边。');
  }
  return { ok: true };
}

export function syncPush(message) {
  if (!existsSync(join(STATE_DIR, '.git'))) return { skipped: '状态目录还不是 git 仓库，跳过 push' };
  commitLocal(message);
  if (!gitOk('push')) {
    throw new SyncConflict('push 被拒绝，远端有本地没有的提交。先人工确认哪台机器是最新的。');
  }
  return { ok: true };
}

/** .cache 里的 62.9MB 原始 csv 永远不进 git。 */
export function writeGitignore() {
  const p = join(STATE_DIR, '.gitignore');
  const body = '.cache/\n';
  if (!existsSync(p) || readFileSync(p, 'utf8') !== body) writeFileSync(p, body);
}

// ---------- 自检 ----------
function selftest() {
  let bad = 0;
  const check = (label, got, want) => {
    const ok = JSON.stringify(got) === JSON.stringify(want);
    if (!ok) bad++;
    console.log(`${ok ? '✓' : '✗'} ${label}${ok ? '' : `：得到 ${JSON.stringify(got)}，期望 ${JSON.stringify(want)}`}`);
  };

  const now = Date.parse('2026-09-30T00:00:00Z');
  const at = d => new Date(now - d * DAY).toISOString();
  const sit = (daysAgo, items, extra = {}) => ({
    v: 1, startedAt: at(daysAgo), finishedAt: at(daysAgo), mode: 'level',
    complete: true, items, wordlistSha: 'aaaa', ...extra,
  });

  const history = [
    sit(200, [
      { rank: 100, word: 'old1', correct: true, firstExposure: true },
      { rank: 200, word: 'old2', correct: false, firstExposure: true },
      { rank: 300, word: 'old3', correct: true, firstExposure: true },
    ]),
    sit(10, [{ rank: 900, word: 'recent', correct: true, firstExposure: true }]),
    sit(5, [{ rank: 950, word: 'half', correct: false, firstExposure: true }], { mode: 'track', complete: false }),
  ];

  // 冷却期：200 天前的过期放行，10 天前的挡住，未完成的也挡住（题面见过了）
  check('冷却集合', [...computeCooldown(history, 90, now)].sort((x, y) => x - y), [900, 950]);
  check('冷却期可调', [...computeCooldown(history, 3, now)].sort((x, y) => x - y), []);
  check('冷却集合随时间清空', computeCooldown(history, 90, now + 100 * DAY).size, 0);
  check('冷却窗口边界', computeCooldown(history, 7, now).size, 1);   // 只剩 5 天前那次

  check('趋势排除未完成', onlyComplete(history).map(r => r.items[0].word), ['old1', 'recent']);
  check('最近一次不限模式', pickLast(history).items[0].word, 'recent');
  check('最近一次限模式', pickLast(history, 'track'), null);
  check('空历史', pickLast([], 'level'), null);

  const m = memoryMetrics([
    { firstExposure: false, wasCorrect: true, correct: true },
    { firstExposure: false, wasCorrect: true, correct: true },
    { firstExposure: false, wasCorrect: true, correct: false },
    { firstExposure: false, wasCorrect: false, correct: true },
    { firstExposure: true, correct: true },
  ]);
  check('保持率', m.retention, 67);
  check('遗忘率', m.forgetting, 33);
  check('挽回率', m.recovery, 100);
  check('复现/首考计数', [m.repeated, m.firstExposure], [4, 1]);
  check('无复现词时不编造百分比', memoryMetrics([{ firstExposure: true, correct: true }]).retention, null);

  check('标注复现',
    annotateExposure([{ rank: 300 }, { rank: 400 }], history).map(x => [x.rank, x.firstExposure, x.wasCorrect]),
    [[300, false, true], [400, true, undefined]]);

  check('尺子漂移检测', findDrift(history, 'aaaa').length, 0);
  check('尺子漂移检测（换了版本）', findDrift(history, 'bbbb').length, 2);

  console.log(bad === 0 ? '\n自检通过' : `\n自检失败 ${bad} 项`);
  process.exit(bad === 0 ? 0 : 1);
}

if (process.argv[2] === '--selftest' && isMain(import.meta.url)) selftest();
