# E3 Pilot-v2 冻结规格（2026-08-27；与实现同一提交冻结，执行在后）
# 依据第十轮裁定 FRESH_PILOT_REQUIRED；实验主责的三处实例化选择见下并预声明。

## 配置
主配置：δ=0.15, root_lr_scale=0.3, q0=0.05, ε=0.05, σ_obs=0.30, 400 updates。
次级稳健配置（预声明，不参与主门救援）：δ=0.30, rls=0.1, q0=0.05。
Matched comparator（实验主责选择 #1，预声明次级、无门）：主配置同 seeds、
rls=1.0——保留时标叙事的可测试性；仅报告 trap 率对比，不作机制主张。

## 种子（全新 5200 段；final 6000 段继续封存）
主配置：低端 k2 5200-5229（30）；高端 k2（q0=0.9）5230-5259（30）；
k64 对照（q0=0.05）5200-5229（与低端 k2 同 seeds，CRN 配对）；
comparator 低端 5200-5229（同 seeds）。次级配置：低/高/k64 各 15（5260-5274 /
5275-5289 / 5260-5274）。n=30 为实验主责选择 #2（单格约 15 秒）。

## 测量与门（机械判定；分支分报；无池化）
- Q1 competence（分支分报）：低端 runs favored 分支 MSE ≤ 1.10×0.23511
  （构造参考，非天花板）≥28/30；disfavored 分支如实报告（不设门——其劣化
  即碰撞代价，属现象）。
- Q2 trap 可重复性：Pr(trap|low) 的双侧 CP 下界 > 0.8（30/30→0.884，
  29/30→0.828，28/30 不过）。
- Q3 escape：Pr(escape|high) CP 下界 > 0.8。
- Q4 容量对照：k64 trapped ≤ 1/30。
- Q5 中介双侧：trapped x12>x23 与 escaped-high x23>x12 各 CP 下界 > 0.5。
- Q6 forced gap 双侧符号：同 Q5 标准。
- Q7 return 链（实验主责选择 #3）：CRN 配对容量赤字——同 seed 同 q0 的
  return(k64) − return(k2)（deterministic deployment return），seed 级
  bootstrap CI 下界 > 0；sampled return 同报。
- tail stability：全臂 tail_drift ≤ 0.1。
- 全部按臂分层，禁池化；UNRESOLVED 规则沿用；admission（census/哈希/拒覆盖）
  沿用 pilot.py 机制，census = 本文件所列 job 全集。

## 判读边界
全过 = 允许设计 final cohort（不代表 E3 主张成立）；任一失败 = 保留停止，
调整开新 development version。
