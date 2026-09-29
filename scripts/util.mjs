// vocab-size · 共用小工具
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

/** 只有直接被 node 执行时才算主模块；被 import 时不跑自检。 */
export function isMain(metaUrl) {
  try {
    return realpathSync(process.argv[1]) === realpathSync(fileURLToPath(metaUrl));
  } catch { return false; }
}

/** 可复现随机数。种子会写进历史记录，验收测试才能重放同一套题。 */
export function makeRnd(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

export const median = xs => {
  const s = [...xs].sort((p, q) => p - q);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
