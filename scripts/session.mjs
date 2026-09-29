#!/usr/bin/env node
// vocab-size · 一次测试的编排：起临时服务、驱动抽题、落库、同步、打印摘要
//
//   node session.mjs level            全量测试，160 题，报词汇量 ± 置信区间
//   node session.mjs track            窄带追踪，100 题，报这一段涨了多少
//
// 服务只活在这一次测试期间：随机空闲端口，测完自动退出。
// 答案的正确判定全在服务端，浏览器只拿到释义文本，拿不到正确答案。
//
// 中途关页面 / Ctrl-C：已答的题会落进 history.jsonl 并标 complete=false，
// 不进历史趋势，但**进冷却期**（题面见过了，再出就是背答案）。

import { createServer } from 'node:http';
import { readFileSync, existsSync, writeFileSync, rmSync } from 'node:fs';
import { execFile } from 'node:child_process';
import { join, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadWordlist, WORDLIST, STATE_DIR } from './wordlist.mjs';
import { createSampler, N_ITEMS, TRACK_BAND_WIDTH, UNKNOWN_CHOICE } from './sampler.mjs';
import { TARGET_SIZE } from './wordlist.mjs';
import {
  loadHistory, appendSitting, annotateExposure, computeCooldown,
  pickLast, syncPull, syncPush, writeGitignore, SyncConflict,
  COOLDOWN_DAYS, wordlistSha, ensureDirs,
} from './state.mjs';
import { buildReport, textSummary } from './report.mjs';
import { fit } from './estimate.mjs';
import { isMain } from './util.mjs';

const ASSETS = join(dirname(fileURLToPath(import.meta.url)), '..', 'assets');
const PARTIAL = join(STATE_DIR, 'session.partial.json');

const MIME = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.svg': 'image/svg+xml' };

function parseArgs(argv) {
  const [mode, ...rest] = argv;
  const get = (flag, dflt) => {
    const i = rest.indexOf(flag);
    return i >= 0 && rest[i + 1] ? rest[i + 1] : dflt;
  };
  return {
    mode,
    items: Number(get('--items', N_ITEMS[mode] || N_ITEMS.level)),
    seed: get('--seed') ? Number(get('--seed')) : (Date.now() & 0x7fffffff),
    bandWidth: Number(get('--band-width', TRACK_BAND_WIDTH)),
    // 窄带命中率超过这个值就认为"啃完了"，下次往上挪一带。
    // 这是个手感参数，没有实测最优值，保守取 0.6（还有 40% 不认识，信息量最大）。
    bandAdvanceAt: Number(get('--band-advance-at', 0.6)),
    cooldownDays: Number(get('--cooldown-days', COOLDOWN_DAYS)),
    browser: !rest.includes('--no-browser'),
    sync: !rest.includes('--no-sync'),
  };
}

/** 上次测试留下的先验。track 必须有先验，level 没有就走盲扫。 */
function resolvePrior(mode, history) {
  const last = pickLast(history);
  if (!last?.estimate?.theta) return { prior: null, source: null };
  const lastLevel = pickLast(history, 'level');
  return {
    prior: { theta: last.estimate.theta, a: lastLevel?.estimate?.a || last.estimate.a },
    source: { date: (last.startedAt || '').slice(0, 10), mode: last.mode, vocab: last.estimate.vocab },
  };
}

/**
 * track 的窄带该放哪儿。
 * 规则：上次 track 的命中率还没到 bandAdvanceAt，就**继续测同一段**（否则前后没法比）；
 * 已经到了就把带子往上挪一格；从来没有 track 记录就用 exp(theta) 定位。
 */
function resolveTrackBand(history, opts) {
  const lastTrack = pickLast(history, 'track');
  const b = lastTrack?.band;
  if (!b?.lo) return { override: null, note: '第一次 track，窄带按上次阈值定位' };
  if (b.rate < opts.bandAdvanceAt * 100) {
    return { override: { lo: b.lo, hi: b.hi }, note: `沿用上次同一段（上次命中率 ${b.rate}% < ${opts.bandAdvanceAt * 100}%），这样才能比出增量` };
  }
  const lo = Math.min(b.hi + 1, TARGET_SIZE - 1);
  const hi = Math.min(lo + opts.bandWidth - 1, TARGET_SIZE);
  return { override: { lo, hi: Math.max(hi, lo + 1) }, note: `上次那一段命中率已到 ${b.rate}%，往上挪一带` };
}

export function runSession(opts) {
  return new Promise((resolveSession, rejectSession) => {
    ensureDirs();
    writeGitignore();

    if (!existsSync(WORDLIST)) {
      rejectSession(new Error(
        `词表不存在：${WORDLIST}\n先跑一次：node scripts/wordlist.mjs build --yes\n`
        + `（需要下载 62.9MB 的 ECDICT 原始词典，提炼成 1.9MB 的小表后就删掉）`));
      return;
    }

    // 上一次没跑完的会话：补记为 incomplete，别让它烂在临时文件里
    if (existsSync(PARTIAL)) finalizePartial('上次测试没做完，已按 incomplete 归档');

    let history;
    try { if (opts.sync) syncPull(); history = loadHistory(); }
    catch (e) {
      if (e instanceof SyncConflict) { rejectSession(e); return; }
      console.error(`拉取状态仓失败（${e.message}），本次改用本地历史继续。`);
      history = loadHistory();
    }

    const wordlist = loadWordlist();
    const cooldown = computeCooldown(history, opts.cooldownDays);
    const { prior, source } = resolvePrior(opts.mode, history);

    if (opts.mode === 'track' && !prior) {
      rejectSession(new Error(
        'track 模式需要上次的测试结果来定位窄带，但历史里没有。\n先跑一次：node session.mjs level'));
      return;
    }

    const drift = history.filter(r => r.complete && r.wordlistSha && r.wordlistSha !== wordlistSha());
    if (drift.length) {
      console.error(`⚠ 有 ${drift.length} 条历史记录是在另一版本的词表下测的，rank 含义已变，趋势对比会失真。`);
    }

    const track = opts.mode === 'track' ? resolveTrackBand(history, opts) : null;
    if (track) console.error(track.note);
    const sampler = createSampler({
      mode: opts.mode, nItems: opts.items, wordlist,
      cooldown, prior, seed: opts.seed, bandWidth: opts.bandWidth,
      bandOverride: track?.override || undefined,
    });

    const startedAt = new Date().toISOString();
    let finished = false;
    let report = null;

    console.error(`模式 ${opts.mode} · ${opts.items} 题 · 冷却期 ${opts.cooldownDays} 天内测过的 ${cooldown.size} 个词已排除`);
    if (source) console.error(`先验来自 ${source.date} 的 ${source.mode} 测试（词汇量 ${source.vocab}）`);
    else console.error('没有历史先验，本次走全表盲扫（第一次测就是这样）');
    if (opts.mode === 'track') console.error(`窄带：第 ${sampler.band.lo}–${sampler.band.hi} 名，a 钉死在 ${sampler.fixA}`);

    // 每答一题就落一次盘。只序列化已答的题，不重跑拟合。
    function persistPartial() {
      const snap = sampler.snapshot();
      writeFileSync(PARTIAL, JSON.stringify({
        mode: opts.mode, startedAt, band: sampler.band, fixA: sampler.fixA,
        seed: opts.seed, nItems: opts.items,
        items: snap.items, optionWords: snap.optionWords,
      }));
    }

    function finish(complete) {
      if (finished) return;
      finished = true;
      const raw = sampler.finish();
      const items = annotateExposure(raw.items, history);
      report = buildReport({ sitting: { ...raw, items }, history, wordlist });
      if (existsSync(PARTIAL)) rmSync(PARTIAL);
      appendSitting({
        startedAt,
        finishedAt: new Date().toISOString(),
        mode: opts.mode,
        complete,
        note: complete ? null : '中途退出',
        seed: opts.seed,
        nItemsPlanned: opts.items,
        cooldownDays: opts.cooldownDays,
        prior: source,
        band: report.band ? { lo: report.band.lo, hi: report.band.hi, width: report.band.width,
          hits: report.band.hits, n: report.band.items, rate: report.band.rate,
          knownNow: report.band.knownNow } : null,
        items,
        optionWords: raw.optionWords,
        estimate: report.headline && {
          vocab: report.headline.vocab, ci95: report.headline.ci95, seVocab: report.headline.seVocab,
          theta: report.headline.theta, a: report.headline.a, aFixed: report.headline.aFixed,
          atBound: report.headline.atBound, nItems: report.headline.nItems,
        },
        memory: report.memory,
      });
      if (opts.sync) {
        try { syncPush(`test: ${opts.mode} ${new Date().toISOString().slice(0, 10)} 词汇量 ${report.headline?.vocab ?? '?'}`); }
        catch (e) {
          console.error(e instanceof SyncConflict
            ? `⚠ ${e.message}\n历史已存在本地，没丢，但没推上远端。`
            : `⚠ 推送状态仓失败：${e.message}`);
        }
      }
      console.log('\n' + (complete ? textSummary(report) : '测试未完成，已按 incomplete 归档：已答的题不进历史趋势，但进 90 天冷却期（题面见过了）。'));
      server.close();
      resolveSession({ report, complete });
    }

    function sendJson(res, code, obj) {
      const body = JSON.stringify(obj);
      res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': Buffer.byteLength(body), 'Cache-Control': 'no-store' });
      res.end(body);
    }

    /** 发给浏览器的题目：只有释义，没有正确答案，也没有干扰项的词形。 */
    const publicItem = it => it ? {
      index: sampler.answered + 1, total: opts.items,
      word: it.word, phonetic: it.phonetic, tag: it.tag,
      options: it.options.map(o => o.glossLines),
      unknownChoice: UNKNOWN_CHOICE,
    } : null;

    let current = sampler.next();

    const server = createServer((req, res) => {
      const url = new URL(req.url, 'http://localhost');
      const p = url.pathname;

      if (req.method === 'GET' && (p === '/' || p === '/index.html')) {
        const body = readFileSync(join(ASSETS, 'index.html'));
        res.writeHead(200, { 'Content-Type': MIME['.html'], 'Content-Length': body.length, 'Cache-Control': 'no-store' });
        return res.end(body);
      }
      if (req.method === 'GET' && (p === '/style.css' || p === '/app.js')) {
        const f = join(ASSETS, p.slice(1));
        if (!existsSync(f)) { res.writeHead(404); return res.end(); }
        const body = readFileSync(f);
        res.writeHead(200, { 'Content-Type': MIME[extname(f)], 'Content-Length': body.length, 'Cache-Control': 'no-store' });
        return res.end(body);
      }

      if (req.method === 'GET' && p === '/api/state') {
        return sendJson(res, 200, {
          mode: opts.mode, answered: sampler.answered, total: opts.items,
          band: sampler.band, prior: source, item: publicItem(current), done: finished,
        });
      }

      if (req.method === 'POST' && p === '/api/answer') {
        let body = '';
        req.on('data', c => { body += c; if (body.length > 1e5) req.destroy(); });
        req.on('end', () => {
          if (finished) return sendJson(res, 409, { error: '测试已结束' });
          if (!current) return sendJson(res, 409, { error: '没有待答的题' });
          let choice;
          try { choice = JSON.parse(body).choice; } catch { return sendJson(res, 400, { error: '请求体不是合法 JSON' }); }
          if (!Number.isInteger(choice) || choice < 0 || choice > UNKNOWN_CHOICE) {
            return sendJson(res, 400, { error: `choice 必须是 0..${UNKNOWN_CHOICE} 的整数` });
          }
          const answered = current;
          sampler.answer(choice, Number(JSON.parse(body).ms) || null);
          current = sampler.next();
          try { persistPartial(); } catch (e) { console.error(`写入 session.partial.json 失败：${e.message}`); }

          if (!current || sampler.answered >= opts.items) {
            const payload = {
              correct: answered.correct, answerIndex: answered.answerIndex,
              chosen: choice === UNKNOWN_CHOICE ? null : answered.options[choice].word,
              answerWord: answered.word, answerPhonetic: answered.phonetic,
              answerGloss: answered.options[answered.answerIndex].glossLines,
              answered: sampler.answered, total: opts.items, done: true,
            };
            finish(true);
            payload.report = report;
            return sendJson(res, 200, payload);
          }
          sendJson(res, 200, {
            correct: answered.correct, answerIndex: answered.answerIndex,
            chosen: choice === UNKNOWN_CHOICE ? null : answered.options[choice].word,
            answerWord: answered.word, answerPhonetic: answered.phonetic,
            answerGloss: answered.options[answered.answerIndex].glossLines,
            answered: sampler.answered, total: opts.items, done: false,
            item: publicItem(current),
          });
        });
        return;
      }

      if (req.method === 'POST' && p === '/api/quit') {
        finish(false);
        return sendJson(res, 200, { ok: true });
      }

      if (req.method === 'GET' && p === '/api/result') {
        return sendJson(res, 200, { report, done: finished });
      }

      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('404');
    });

    server.on('error', e => rejectSession(e));
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      const link = `http://127.0.0.1:${port}/`;
      console.error(`\n测试页：${link}\n`);
      console.log(link);
      if (opts.browser) openBrowser(link);
    });

    // Ctrl-C / 终端关闭：把已答的题按 incomplete 归档，别丢
    const onExit = () => { if (!finished) { try { finish(false); } catch { /* 已经没救了 */ } } process.exit(0); };
    process.on('SIGINT', onExit);
    process.on('SIGTERM', onExit);
  });
}

function finalizePartial(reason) {
  try {
    const p = JSON.parse(readFileSync(PARTIAL, 'utf8'));
    const items = annotateExposure(p.items || [], loadHistory());
    appendSitting({
      startedAt: p.startedAt, finishedAt: new Date().toISOString(),
      mode: p.mode, complete: false, seed: p.seed, nItemsPlanned: p.nItems,
      band: p.band, items, optionWords: p.optionWords,
      estimate: fit(items.filter(it => it.firstExposure), { fixA: p.fixA }),
      note: reason,
    });
    rmSync(PARTIAL);
    console.error(`⚠ ${reason}（${items.length} 题已归档为 incomplete）`);
  } catch (e) {
    console.error(`⚠ 无法归档上次未完成的会话：${e.message}`);
    try { rmSync(PARTIAL); } catch { /* 忽略 */ }
  }
}

function openBrowser(link) {
  const cmd = process.platform === 'darwin' ? ['open', [link]]
    : process.platform === 'win32' ? ['cmd', ['/c', 'start', '', link]]
      : ['xdg-open', [link]];
  execFile(cmd[0], cmd[1], err => { if (err) console.error(`打不开浏览器，请手动访问 ${link}`); });
}

if (isMain(import.meta.url)) {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.mode !== 'level' && opts.mode !== 'track') {
    console.error('用法: node session.mjs level|track [--items N] [--seed S] [--band-width N] [--cooldown-days N] [--no-browser] [--no-sync]');
    process.exit(2);
  }
  try {
    const { report } = await runSession(opts);
    process.exit(report?.headline ? 0 : 1);
  } catch (e) {
    console.error(`\n${e.message}`);
    process.exit(1);
  }
}
