# vocab-size

本地自适应英语词汇量测试：英词 → 四选一中文释义（外加一个"不认识"按钮），IRT 2PL 拟合出 50% 命中阈值，对 COCA 频率前 2 万词积分得到词汇量 ± 95% 置信区间。

两种模式：

| 模式 | 题量 / 时长 | 抽题 | 用途 |
|---|---|---|---|
| `level` | 160 题 / 约 21 分钟 | 先验播种 + 全表自适应 | 定总量、测个人陡峭度 `a`；每 6–12 个月一次 |
| `track` | 100 题 / 约 13 分钟 | 阈值上方 3000 词窄带，`a` 钉死 | 盯窄带内涨跌，灵敏约 3 倍；每季度一次 |

## 用法

```bash
node scripts/session.mjs level      # 第一次必须跑这个
node scripts/session.mjs track      # 之后默认
```

进程会开一个随机端口的本地 HTTP 服务并打开浏览器；stdout 第一行是测试页 URL。答完自动出结果页、写 `~/.vocab-test/history.jsonl`、把文字摘要打到 stdout。

状态目录默认 `~/.vocab-test/`，可用环境变量 `VOCAB_TEST_STATE_DIR` 覆盖。词表缺失时先征得同意再 `node scripts/wordlist.mjs build --yes`（要下载 62.9MB 原始 CSV）。

## 模型与诚实的精度

- `P(认识 | rank r) = sigmoid(a · (θ − ln r))`；有显式"不认识"按钮，所以猜测参数 `c = 0`。
- 词汇量 `V = Σ_{r≤20000} P(认识|r)`，查表 + 线性插值；CI 由 θ 的渐近标准误链式传过来。
- 总量**只用首考词**算；复现词单独报保持率 / 遗忘率 / 挽回率三个记忆指标。
- 90 天冷却期：测过的词 90 天内不再出现（中途退出的题也算见过）。
- 第一次 level 之前 `a` 未知（默认 4.0），所有精度数字在那之前都不可信。
- 蒙特卡洛仿真（`sim/precision-sim-v4.mjs`）：+500 词的真实增长，总量测法检出力约 27%；窄带追踪对集中在某一段的增长检出力显著更高。**小于置信区间的起伏不要当真相。**

## 数据与许可

词表提炼自 [skywind3000/ECDICT](https://github.com/skywind3000/ECDICT) — MIT License, Copyright (c) 2017-2025 skywind3000。ETL 只保留词频 > 0、无空格、可解析词性、非缩写、原形（靠 `exchange` 字段去掉变形）的词，取前 20000；产出 `wordlist.tsv` 约 1.9MB，文件头记录源文件 sha256 与各步淘汰数，Mac / Windows 构建逐字节一致。

## 仓库与隐私

代码公开（`heeyezzz/vocab-size`），答题历史与词表走私有仓（`heeyezzz/vocab-size-data`）。同步只 `pull --ff-only` + push，冲突时停下来问人，绝不自动合并。

## 自检

```bash
bash tests/acceptance.sh
```
