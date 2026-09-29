---
name: vocab-size
description: Use when the user wants to measure or track their English vocabulary size — 测词汇量 / 词汇量测试 / 我的词汇量 / 测测我认识多少词 / vocab size / vocabulary test. Runs a local adaptive four-option test (English word → pick the Chinese gloss) and reports total vocab ± 95% CI, mastery curve by frequency band, exam-tag coverage, history trend and the wrong-word list.
---

# vocab-size

本地自适应词汇量测试。两种模式：

- `level`（全量定标）：160 题 / 约 21 分钟。先验播种自适应抽题，报总量 ± 95% CI。**第一次必须跑它**——顺便测出用户个人的陡峭度 `a`，之后所有精度数字都依赖它。建议每 6–12 个月一次。
- `track`（窄带追踪）：100 题 / 约 13 分钟。在上次阈值上方一个 3000 词窄带里抽题，`a` 钉死用上次 level 的值，对窄带内的涨跌灵敏约 3 倍。默认模式，建议每季度一次。

## 何时用

用户说"测词汇量 / 词汇量测试 / 看看我认识多少词 / vocab size"，或直接 `/vocab-size [level|track]`。
没给模式时：历史为空 → 跑 `level`（并告诉用户第一次只能全量）；有历史 → 跑 `track`，用户明说"全量 / 重新定标 / level"才跑 `level`。

## 一次性准备：词表

状态目录 `~/.vocab-test/`（可用 `VOCAB_TEST_STATE_DIR` 覆盖）。尺子是 `wordlist.tsv`（前 2 万词，1.9MB）。

若 `wordlist.tsv` 不存在：

1. **先问用户同意**下载 62.9MB 的 ECDICT 原始 CSV（来源 skywind3000/ECDICT，MIT）。没同意就不要下载。
2. 同意后跑 `node scripts/wordlist.mjs build --yes`（ETL 是确定性的：Mac / Windows 产出的尺子逐字节一致）。
3. `node scripts/wordlist.mjs stats` 核对：20000 词、rank 连续、frq（COCA 频率排位，越小越常见）单调不降。

## 跑一次测试

```bash
node scripts/session.mjs level|track [--items N] [--seed S] [--band-width N] \
     [--cooldown-days N] [--band-advance-at F] [--no-browser] [--no-sync]
```

- **stdout 第一行是测试页 URL**（进程自己开随机端口、自己开浏览器）。立刻把 URL 转告用户；浏览器没起来时让用户手动打开。
- 诊断信息在 stderr；**进程退出时把文字摘要打到 stdout**，把它转述给用户（别只甩 URL）。
- 后台跑，等进程自己退出。**不要替用户答题，不要中途 kill**——中途退出会归档成 `incomplete`：不进趋势，但进 90 天冷却期（题面已经见过了）。
- 结束后读 `~/.vocab-test/history.jsonl` 最后一条核对落盘。

旗标都是手感参数，保守默认即可，用户要调再调：`--items` 题量；`--band-width` 窄带宽度（默认 3000）；`--cooldown-days` 冷却期（默认 90）；`--band-advance-at` 上次窄带命中率超过这个比例才往上挪一带（默认 0.6，没有实测最优值）；`--no-browser` / `--no-sync` 用于调试。

## 结果怎么读（照实说，别吹）

- 总量 = IRT 2PL 拟合出 50% 命中阈值后对前 2 万词积分；`a` 未知时（第一次 level 之前）用默认 4.0，**精度数字在那之前都别当回事**。
- `a` 钉不死是常态而不是故障：作答曲线扁的时候似然沿 `a` 是一条脊。估计器对 `a` 网格做轮廓似然加权的模型平均，95% 区间取似然比集合（Δll≤1.92）内各 `a` 的渐近区间之并；`shapeUncertain=true` 时结果页和摘要都会明说"曲线扁 = 词汇分布不均匀"。转述时别把这种宽区间说成测失败了。
- level 之间比涨跌要谨慎：仿真（`sim/precision-sim-v4.mjs`）里 +500 词级别的真实增长，总量测法的检出力只有约 27%——小于置信区间的起伏看不出是真涨还是抖动。盯小步进展用 `track`。
- 总量**只用首考词**算；复现词只进三个记忆指标（保持率 / 遗忘率 / 挽回率），否则"记住了上次的答案"会把词汇量越测越高。
- 结果页出现黄色警告（撞边界 / 首考词样本偏薄 / 超出 2 万量程）时，原样转告用户，不要替它圆。

## 双仓同步

- 代码仓 `heeyezzz/vocab-size`（公开）；数据仓 `heeyezzz/vocab-size-data`（私有，只放 `history.jsonl` + `wordlist.tsv`）。
- session 结束自动 `pull --ff-only` 再 push。**冲突时抛 SyncConflict：停下来问用户哪台机器的记录更新，绝不自动合并。**
- 两个仓尚未创建；建仓、首次 push 之前必须先问用户。

## 自检与验收

```bash
bash tests/acceptance.sh
```

覆盖：IRT 估计自检（偏差 / CI 覆盖率 / 全对全错撞边界）、状态层自检（冷却期窗口、记忆指标、incomplete 归档语义）、抽题自检（干扰项硬规则、题量、窄带定位）、词表完整性（若已构建）、以及一次真实的"答两题被 SIGTERM"崩溃恢复冒烟。

## 精度依据

`sim/precision-sim*.mjs`（v1→v4）是蒙特卡洛仿真：题量怎么定、要不要做窄带追踪、增长集中 vs 分散时两种测法的检出力。改题量或改模式之前先读 v4 的结论。

## 校准记录

- 2026-09-30 首次真人 level（160 题，67 对）：`a ≈ 0.71`、`shapeUncertain=true` —— 曲线很扁（前 2000 名命中 76%，1 万名后仍有约 20% 命中），总量 6842 ±882。此后 track 钉死这个 `a`；若用户词汇变得"整齐"，下次 level 会把 `a` 更新上去。
