# E3 Final Cohort 冻结规格（2026-08-27；与实现同一提交冻结；运行须外部静态
# 复审 PASS——本包为送审版，未运行）

## 配置（不变，非 winner-selection）
主配置 δ=0.15, root_lr_scale=0.3, q0(low)=0.05, q0(high)=0.9, ε=0.05,
σ_obs=0.30, 400 updates。comparator rls=1.0（次级敏感性，无门）。

## 种子（6000 段 final，首次启封；全部全新）
k2-low 与 k64-low：6000-6099（同 seeds，CRN）；k2-high 与 k64-high：
6100-6199（同 seeds，CRN；k64-high 为 DiD 所需）；comparator-low：6000-6029。
共 430 runs。

## 功效（精确二项，冻结）
F2 门 CP 下界>0.8：n=100 需 ≥89/100；在 pilot-v2 点估计 p=0.933 下功效 0.965
（n=30/40/80 分别为 0.394/0.493/0.913——n=100 为功效正当选择）。

## 门（机械；分层禁池化；UNRESOLVED 规则沿用）
- F1 competence（**endpoint-conditioned validity**，v2 修订：init-defined
  low-S 门与 F2 冲突——S-MSE 即 trap 指示量，95/100 在 p=.933 下功效仅
  ~34%）：trapped k2-low 的 S-MSE ≤1.10×0.23511 比例 CP 下界 >0.8；
  escaped k2-high 的 E-MSE 同标准；其余分支均值全报不设门。
- F2 trap：Pr(trap|k2-low) CP 下界 >0.8。
- F3 escape：Pr(escape|k2-high) CP 下界 >0.8。
- F4 容量对照：k64-low 与 k64-high 各 trapped ≤1/100。
- F5/F6 中介与 gap 双侧：trapped x12>x23 与 gap<0、escaped(high) 反序与
  gap>0，各 CP 下界 >0.5。
- F7 return 三件套（seed 级 bootstrap；CRN 仅限同初始化的 k2/k64 配对——
  low 与 high 为不同 seed 段，其间为独立样本，bootstrap 分开重采样）：
  a) matched rank-capacity removal return gap：(k64−k2|low) CI 下界 >0；
  b) 盆地差：k2 (high−low) return CI 下界 >0；
  c) **basin-specific rank interaction（DiD）**：(k64−k2|low) −
     (k64−k2|high) CI 下界 >0——扣除共同 rank gap，但依赖普通容量效应在
     low/high 间可比的假设，不宣称唯一中介路径识别。
- tail stability 全臂 ≤0.1。
判读：全过 = E3 存在性主张达确认级（仍非 prevalence）；任一失败 = 保留停止。

## v2 修订（第十二轮审查，运行前；环境/PPO/配置/种子数不变）
F1 改 endpoint-conditioned 并实现 high-E 门；admission 绑定本文件哈希
（deploy 复制 + stamp.final_spec_sha256 + Pool 前核验）；CRN 与 DiD 措辞
精确化；合成测试改用 pilot-v2 实际观察结构（逃逸行 S-MSE 劣化）。
