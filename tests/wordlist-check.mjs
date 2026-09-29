// 词表完整性：rank 连续、frq（COCA 频率排位，越小越常见）单调不降、词形合法且唯一（lemma 去重的残留检查）。
import { existsSync } from 'node:fs';
import { loadWordlist, WORDLIST, TARGET_SIZE } from '../scripts/wordlist.mjs';

if (!existsSync(WORDLIST)) {
  console.log('跳过词表完整性：词表未构建（先跑 wordlist.mjs build）');
  process.exit(0);
}

const ws = loadWordlist();
let fail = 0;
const check = (ok, label) => { console.log(`${ok ? '  ✓' : '  ✗'} ${label}`); if (!ok) fail++; };

check(ws.length === TARGET_SIZE, `词数 = ${TARGET_SIZE}（实际 ${ws.length}）`);

let rankOk = true, frqOk = true, formOk = true, glossOk = true;
const seen = new Set();
for (let i = 0; i < ws.length; i++) {
  const w = ws[i];
  if (w.rank !== i + 1) rankOk = false;
  if (i && w.frq < ws[i - 1].frq) frqOk = false;
  if (!/^[a-z][a-z'\-]*$/.test(w.word) || /\s/.test(w.word)) formOk = false;
  if (seen.has(w.word)) formOk = false;
  seen.add(w.word);
  if (!w.glossLines.length || w.glossLines.some(g => !g.trim())) glossOk = false;
}
check(rankOk, 'rank 从 1 连续递增');
check(frqOk, 'frq（COCA 频率排位）随 rank 单调不降');
check(formOk, '词形合法且无重复（lemma 去重无残留）');
check(glossOk, '每行都有非空释义');

console.log(fail === 0 ? '词表完整性通过' : `词表完整性失败 ${fail} 项`);
process.exit(fail === 0 ? 0 : 1);
