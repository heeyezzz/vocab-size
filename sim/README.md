# 精度仿真（蒙特卡洛）

决策依据，改题量 / 改模式之前先读 v4 结论。

- `precision-sim.mjs`（v1）：盲扫 vs 先验播种自适应，定 level 的题量下限。
- `precision-sim-v2.mjs`：窄带钉 `a` 的偏差与 CI 覆盖率，定 track 的题量。
- `precision-sim-v3.mjs`：总量估算 vs 窄带追踪的检出力对照（增长 = θ 整体平移）。
- `precision-sim-v4.mjs`：决定性实验——增长"集中"vs"分散"时两种测法的检出力；结论：集中增长下窄带追踪显著更灵敏，分散增长下两者打平，所以双模式都保留。

跑法：`node sim/precision-sim-v4.mjs`（纯 Node，无依赖，约几十秒）。
