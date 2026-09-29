// 崩溃恢复冒烟：真起一个 session，答两题，SIGTERM 它，
// 断言：历史里多了一条 complete:false、已答 2 题；session.partial.json 被清掉。
import { spawn } from 'node:child_process';
import { mkdtempSync, copyFileSync, readFileSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const SKILL = join(dirname(fileURLToPath(import.meta.url)), '..');
const SESSION = join(SKILL, 'scripts', 'session.mjs');
const REAL_WL = join(process.env.HOME || '', '.vocab-test', 'wordlist.tsv');

if (!existsSync(REAL_WL)) {
  console.log('跳过崩溃冒烟：默认状态目录里没有 wordlist.tsv（先跑 wordlist.mjs build）');
  process.exit(0);
}

const dir = mkdtempSync(join(tmpdir(), 'vocab-accept-'));
copyFileSync(REAL_WL, join(dir, 'wordlist.tsv'));

let fail = 0;
const check = (ok, label) => { console.log(`${ok ? '  ✓' : '  ✗'} ${label}`); if (!ok) fail++; };

const child = spawn(process.execPath, [SESSION, 'level', '--items', '6', '--seed', '3', '--no-browser', '--no-sync'], {
  env: { ...process.env, VOCAB_TEST_STATE_DIR: dir },
  stdio: ['ignore', 'pipe', 'pipe'],
});
let out = '';
child.stdout.on('data', c => { out += c; });
child.stderr.resume();

const url = await new Promise((resolve, reject) => {
  const t = setTimeout(() => reject(new Error('8 秒内没等到测试页 URL')), 8000);
  const iv = setInterval(() => {
    const m = out.match(/^(https?:\/\/\S+)$/m);
    if (m) { clearInterval(iv); clearTimeout(t); resolve(m[1]); }
  }, 100);
});
const base = url.replace(/\/+$/, '');

let item = (await (await fetch(`${base}/api/state`)).json()).item;
for (let i = 0; i < 2; i++) {
  const r = await fetch(`${base}/api/answer`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ choice: 0 }),
  });
  const j = await r.json();
  item = j.item ?? null;
}
check(item != null, '答完 2 题后还有下一题');

child.kill('SIGTERM');
await new Promise(resolve => child.on('exit', resolve));

const hist = join(dir, 'history.jsonl');
check(existsSync(hist), 'SIGTERM 后写入了 history.jsonl');
if (existsSync(hist)) {
  const lines = readFileSync(hist, 'utf8').trim().split('\n');
  const last = JSON.parse(lines.at(-1));
  check(last.complete === false, '归档为 incomplete');
  check(last.items?.length === 2, `已答题数落盘为 2（实际 ${last.items?.length}）`);
  check(last.note === '中途退出', 'note 标注中途退出');
}
check(!existsSync(join(dir, 'session.partial.json')), 'session.partial.json 已清理');

rmSync(dir, { recursive: true, force: true });
console.log(fail === 0 ? '崩溃冒烟通过' : `崩溃冒烟失败 ${fail} 项`);
process.exit(fail === 0 ? 0 : 1);
