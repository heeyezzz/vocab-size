#!/usr/bin/env bash
# vocab-size 验收：估计核心 / 状态层 / 抽题 三个自检 + 词表完整性 + 崩溃恢复冒烟
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 估计核心自检（偏差 / CI 覆盖率 / 全对全错撞边界）=="
node scripts/estimate.mjs --selftest
echo
echo "== 状态层自检（冷却期窗口 / 记忆指标 / incomplete 语义）=="
node scripts/state.mjs --selftest
echo
echo "== 抽题自检（干扰项硬规则 / 题量 / 窄带定位）=="
node scripts/sampler.mjs --selftest
echo
echo "== 词表完整性 =="
node tests/wordlist-check.mjs
echo
echo "== 崩溃恢复冒烟（答两题后 SIGTERM）=="
node tests/crash-smoke.mjs
echo
echo "验收全部通过"
