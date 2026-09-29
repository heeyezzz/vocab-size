#!/usr/bin/env node
// vocab-size · 词表提炼
//
// 把 ECDICT 全量 ecdict.csv（77 万词条，62.9MB）提炼成本 skill 用的前 2 万词精简表。
// 提炼必须**完全确定性**：同样的输入 CSV 永远产出字节一致的 wordlist.tsv。
// 这是跨机器可比的前提 —— 用户在 Mac 和 Windows 上同步数据仓，两台机器的尺子必须一模一样。
//
// 用法：
//   node wordlist.mjs build [--yes]     下载（若无）+ 提炼；--yes 表示同意下载 62.9MB
//   node wordlist.mjs stats             打印已提炼词表的统计
//   node wordlist.mjs lookup <word>     查单个词（调试用）

import { createReadStream, existsSync, readFileSync, writeFileSync, mkdirSync, rmSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createInterface } from 'node:readline';
import { execFileSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const STATE_DIR = process.env.VOCAB_TEST_STATE_DIR || join(process.env.HOME, '.vocab-test');
const CACHE_DIR = join(STATE_DIR, '.cache');
const RAW_CSV = join(CACHE_DIR, 'ecdict.csv');
const WORDLIST = join(STATE_DIR, 'wordlist.tsv');

const SOURCE_URL = 'https://raw.githubusercontent.com/skywind3000/ECDICT/master/ecdict.csv';
const TARGET_SIZE = 20000;

// ---------- 词性 ----------
// ECDICT 的 pos 列在本版本里全为空，词性藏在 translation 的行首前缀（简明英汉惯例）
const POS_TOKEN = /([a-z]{1,7}\.)/g;
const DOMAIN_TAG = /^\s*(\[[^\]]{1,8}\]\s*)+/;

function parsePosClass(glossFirstLine) {
  if (!glossFirstLine) return null;
  // 剥掉领域前缀（[计] [医] [经] [化] 等）与 \r
  const s = glossFirstLine.replace(/\r/g, '').replace(DOMAIN_TAG, '');
  const m = s.match(/^\s*((?:[a-z]{1,7}\.)(?:\s*[&,]\s*[a-z]{1,7}\.)*)/);
  if (!m) return null;
  const raw = m[1].replace(/\s+/g, '');
  return { raw, cls: posClassOf(raw) };
}

// 干扰项要"同词性"，但 vt./vi./v. 应视为同一类，否则动词干扰项池太小
const POS_CLASS = {
  'n.': 'NOUN', 'pl.': 'NOUN',
  'v.': 'VERB', 'vt.': 'VERB', 'vi.': 'VERB', 'aux.': 'VERB', 'vt.&vi.': 'VERB',
  'a.': 'ADJ', 'adj.': 'ADJ', 'a.&n.': 'ADJ',
  'adv.': 'ADV', 'ad.': 'ADV',
  'prep.': 'FUNC', 'conj.': 'FUNC', 'pron.': 'FUNC', 'art.': 'FUNC',
  'num.': 'FUNC', 'int.': 'FUNC', 'interj.': 'FUNC',
  'abbr.': 'ABBR', 'pref.': 'ABBR', 'suf.': 'ABBR',
};
function posClassOf(raw) {
  if (POS_CLASS[raw]) return POS_CLASS[raw];
  // 组合词性如 "n.&v."：取第一个已知成分
  for (const part of raw.split(/[&,]/)) {
    const p = part.trim();
    if (POS_CLASS[p]) return POS_CLASS[p];
  }
  return 'OTHER';
}

// ---------- CSV ----------
function splitRow(s) {
  const out = []; let cur = '', inQ = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (inQ) { if (c === '"') { if (s[i + 1] === '"') { cur += '"'; i++; } else inQ = false; } else cur += c; }
    else if (c === '"') inQ = true;
    else if (c === ',') { out.push(cur); cur = ''; }
    else cur += c;
  }
  out.push(cur); return out;
}
async function* parseCsv(rl) {
  let buf = null;
  for await (const line of rl) {
    const raw = buf === null ? line : buf + '\n' + line;
    if ((raw.match(/"/g) || []).length % 2 === 1) { buf = raw; continue; } // 引号跨行
    buf = null; yield splitRow(raw);
  }
  if (buf !== null) yield splitRow(buf);
}

const sha256 = f => createHash('sha256').update(readFileSync(f)).digest('hex');

// ---------- build ----------
async function build(opts) {
  mkdirSync(CACHE_DIR, { recursive: true });

  if (!existsSync(RAW_CSV)) {
    if (!opts.yes) {
      console.error(`缺少 ECDICT 全量词表：${RAW_CSV}`);
      console.error(`需要下载 ${(62.9).toFixed(1)}MB：${SOURCE_URL}`);
      console.error('同意后重跑并加 --yes（或让 agent 代为下载）。');
      process.exit(3);
    }
    console.error(`下载 ECDICT（约 62.9MB）→ ${RAW_CSV}`);
    execFileSync('curl', ['-sL', '--fail', '-o', RAW_CSV, SOURCE_URL], { stdio: 'inherit' });
  }

  const bytes = statSync(RAW_CSV).size;
  const digest = sha256(RAW_CSV);
  console.error(`读取 ${(bytes / 1048576).toFixed(1)}MB / sha256 ${digest.slice(0, 16)}…`);

  const rl = createInterface({ input: createReadStream(RAW_CSV, 'utf8'), crlfDelay: Infinity });
  const rows = parseCsv(rl);
  await rows.next();

  const rejected = { noFrq: 0, phrase: 0, noGloss: 0, inflected: 0, notLowerAlpha: 0, noPos: 0, abbrev: 0 };
  const cand = [];

  for await (const r of rows) {
    const word = r[0], phonetic = r[1], glossRaw = r[3], collins = +r[5] || 0, oxford = +r[6] || 0, tag = (r[7] || '').trim(), frq = +r[9], ex = r[10] || '';

    if (!(frq > 0)) { rejected.noFrq++; continue; }
    if (/\s/.test(word)) { rejected.phrase++; continue; }
    // 只收全小写纯字母词：一条规则同时挡掉专有名词、国籍形容词、缩写（TV/PC/DNA）
    // 大写字母在四选一是视觉泄题，专有名词测的是文化暴露度而非词汇量
    if (!/^[a-z][a-z'\-]*$/.test(word)) { rejected.notLowerAlpha++; continue; }
    if (!glossRaw || !glossRaw.trim()) { rejected.noGloss++; continue; }
    if (/(^|\/)0:/.test(ex)) { rejected.inflected++; continue; } // 屈折形式，原形另有其行

    const gloss = cleanGloss(glossRaw);
    if (!gloss) { rejected.noGloss++; continue; }

    const firstLine = gloss.split('\\n')[0];
    const pos = parsePosClass(firstLine);
    if (!pos) { rejected.noPos++; continue; }
    if (pos.cls === 'ABBR') { rejected.abbrev++; continue; }

    cand.push({ word, phonetic: (phonetic || '').replace(/[\t\r\n]/g, ''), posRaw: pos.raw, posCls: pos.cls, tag, collins, oxford, frq, gloss });
  }

  // 确定性排序：frq 升序，同 frq 内按词形（约 950 个 frq 值被多词共用）
  cand.sort((x, y) => x.frq - y.frq || (x.word < y.word ? -1 : x.word > y.word ? 1 : 0));

  if (cand.length < TARGET_SIZE) {
    console.error(`候选不足：过滤后仅 ${cand.length} 词，需要 ${TARGET_SIZE}`);
    process.exit(4);
  }
  const top = cand.slice(0, TARGET_SIZE);

  const lines = [
    '# vocab-size · 前 2 万词精简词表（词汇量测试的唯一尺子）',
    '#',
    '# 数据来源：skywind3000/ECDICT — MIT License, Copyright (c) 2017-2025 skywind3000',
    '#   https://github.com/skywind3000/ECDICT',
    '#   本文件仅保留该词典的 word/phonetic/tag/collins/oxford/frq 六个字段与中文释义，',
    '#   并按下方规则做了筛选与重排。原始词典数据版权归 ECDICT 作者所有。',
    '#',
    `# 源文件 sha256：${digest}`,
    `# 源文件字节数：${bytes}`,
    `# 提炼时间：${new Date().toISOString().slice(0, 10)}`,
    `# 提炼后词数：${top.length}（原候选 ${cand.length}，取自 ECDICT 全量 77 万词条）`,
    `# frq 覆盖范围：${top[0].frq} .. ${top[top.length - 1].frq}（frq = COCA 语料库词频排名）`,
    '#',
    '# 筛选规则（顺序即拒绝原因统计的顺序）：',
    `#   1. frq > 0                      —— 剔除 ${rejected.noFrq} 个无 COCA 排名的词`,
    `#   2. 词形不含空格                  —— 剔除 ${rejected.phrase} 个短语（只考单词）`,
    `#   3. 匹配 ^[a-z][a-z'\\-]*$        —— 剔除 ${rejected.notLowerAlpha} 个专有名词/缩写/大写词`,
    `#   4. 中文释义非空                  —— 剔除 ${rejected.noGloss} 个`,
    `#   5. exchange 不含 "0:"           —— 剔除 ${rejected.inflected} 个屈折形式（banks/went 等）`,
    `#   6. 释义行首能解析出词性          —— 剔除 ${rejected.noPos} 个`,
    `#   7. 词性不是缩写类                —— 剔除 ${rejected.abbrev} 个 abbr./pref./suf.`,
    '#',
    '# rank 为本表内的位置（1..20000），是词汇量模型里的自变量 x = ln(rank)。',
    '# 注意 rank 不等于 frq：frq 有跳号且被多词共用，rank 是重排后的连续序号。',
    '#',
    '# 列格式（TAB 分隔，共 10 列）：',
    '#   rank  word  phonetic  pos  posCls  tag  collins  oxford  frq  gloss',
    '#   posCls ∈ NOUN|VERB|ADJ|ADV|FUNC|OTHER，用于干扰项同词性匹配',
    '#   gloss 内多行释义用字面两字符 \\n 分隔（沿用 ECDICT 原约定），最多保留前 3 行',
  ];

  for (let i = 0; i < top.length; i++) {
    const w = top[i];
    lines.push([i + 1, w.word, w.phonetic, w.posRaw, w.posCls, w.tag, w.collins, w.oxford, w.frq, w.gloss].join('\t'));
  }

  writeFileSync(WORDLIST, lines.join('\n') + '\n', 'utf8');
  const out = statSync(WORDLIST).size;
  console.error(`\n写出 ${WORDLIST}`);
  console.error(`  ${(out / 1048576).toFixed(2)}MB / ${top.length} 词 / frq ${top[0].frq}..${top[top.length - 1].frq}`);

  // 统计
  const byCls = new Map(), byTag = new Map(), byLines = new Map();
  for (const w of top) {
    byCls.set(w.posCls, (byCls.get(w.posCls) || 0) + 1);
    if (!w.tag) byTag.set('(无标签)', (byTag.get('(无标签)') || 0) + 1);
    for (const t of w.tag.split(/\s+/)) if (t) byTag.set(t, (byTag.get(t) || 0) + 1);
    const n = w.gloss.split('\\n').length;
    byLines.set(n, (byLines.get(n) || 0) + 1);
  }
  console.error(`\n词性类别: ${[...byCls.entries()].sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}=${v}`).join(' ')}`);
  console.error(`考试标签: ${[...byTag.entries()].sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}=${v}`).join(' ')}`);
  console.error(`释义行数: ${[...byLines.entries()].sort((a, b) => a[0] - b[0]).map(([k, v]) => `${k}行=${v}`).join(' ')}`);
  console.error(`collins>0: ${top.filter(w => w.collins > 0).length}  oxford=1: ${top.filter(w => w.oxford === 1).length}`);

  if (opts.prune) {
    rmSync(RAW_CSV);
    console.error(`\n已删除原始 CSV（${(bytes / 1048576).toFixed(1)}MB）`);
  } else {
    console.error(`\n原始 CSV 保留在 ${RAW_CSV}（加 --prune 可删除以释放 ${(bytes / 1048576).toFixed(0)}MB）`);
  }
}

// 释义清洗：去掉 \r 和真换行/制表符，最多保留前 3 行，行间用字面 \n 连接
function cleanGloss(raw) {
  return raw
    .replace(/\r/g, '')
    .split(/\\n|\n/)
    .map(s => s.replace(/[\t]/g, ' ').trim())
    .filter(Boolean)
    .slice(0, 3)
    .join('\\n');
}

// ---------- stats / lookup ----------
function loadWordlist() {
  if (!existsSync(WORDLIST)) {
    console.error(`词表不存在：${WORDLIST}\n先跑 node wordlist.mjs build --yes`);
    process.exit(3);
  }
  const out = [];
  for (const line of readFileSync(WORDLIST, 'utf8').split('\n')) {
    if (!line || line.startsWith('#')) continue;
    const f = line.split('\t');
    if (f.length < 10) continue;
    out.push({
      rank: +f[0], word: f[1], phonetic: f[2], pos: f[3], posCls: f[4],
      tag: f[5], collins: +f[6], oxford: +f[7], frq: +f[8], gloss: f[9],
      glossLines: f[9].split('\\n'),
    });
  }
  return out;
}

function stats() {
  const ws = loadWordlist();
  const byCls = new Map(), byTag = new Map();
  for (const w of ws) {
    byCls.set(w.posCls, (byCls.get(w.posCls) || 0) + 1);
    for (const t of w.tag.split(/\s+/)) if (t) byTag.set(t, (byTag.get(t) || 0) + 1);
  }
  console.log(JSON.stringify({
    path: WORDLIST, count: ws.length,
    rankRange: [ws[0].rank, ws[ws.length - 1].rank],
    frqRange: [ws[0].frq, ws[ws.length - 1].frq],
    posCls: Object.fromEntries([...byCls.entries()].sort((a, b) => b[1] - a[1])),
    tags: Object.fromEntries([...byTag.entries()].sort((a, b) => b[1] - a[1])),
    glossLineHist: (() => { const m = new Map(); for (const w of ws) m.set(w.glossLines.length, (m.get(w.glossLines.length) || 0) + 1); return Object.fromEntries([...m.entries()].sort((a, b) => a[0] - b[0])); })(),
  }, null, 2));
}

function lookup(word) {
  const ws = loadWordlist();
  const w = ws.find(x => x.word === word);
  console.log(w ? JSON.stringify(w, null, 2) : `未收录：${word}`);
}

// ---------- main ----------
export {
  STATE_DIR, CACHE_DIR, RAW_CSV, WORDLIST, SOURCE_URL, TARGET_SIZE,
  parsePosClass, posClassOf, cleanGloss, loadWordlist, build,
};

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [, , cmd, ...rest] = process.argv;
  const opts = { yes: rest.includes('--yes'), prune: rest.includes('--prune') };

  if (cmd === 'build') await build(opts);
  else if (cmd === 'stats') stats();
  else if (cmd === 'lookup') lookup(rest.filter(a => !a.startsWith('--'))[0]);
  else {
    console.error('用法: wordlist.mjs build [--yes] [--prune] | stats | lookup <word>');
    process.exit(2);
  }
}
