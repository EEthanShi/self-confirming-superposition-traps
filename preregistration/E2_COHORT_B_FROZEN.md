# E2 Cohort Block B 冻结规格（2026-08-27；本文件+代码+靶同一提交冻结，执行在后）

## 相对 Block A 的变更（依第三轮外部审查）
1. 全部靶预先写死（含 δ=0.6 无约束靶，源 du_authority_ext，已在冻结时存在）。
2. p0 网格：各预测 ±0.03 内步长 0.0075 的并集 + 外扩锚点（39 点/δ）；
   任何 UNRESOLVED（边界非内点或 bootstrap 非有限率>5%）= 该格自动 FAIL，
   edge 不可能通过。
3. C2/C3 改真等价门：err = boundary − pred 的 95% CI 整体落入
   constr ±0.02 / unconstr ±0.03。
4. floor 主门 = 配对 shift CI 下界 >0；幅度 vs 预测为次级报告（无门）。
   floor 对改为 ε∈{0.05, 0.25}（预测 shift 0.008–0.140，全部 ≥ 网格步长）。
5. balanced forced evaluation 补全五级链并设链门（C5–C7）。
6. fullrank3 全文改称 **capacity control**（3×3 vs 2×3 参数量不等；
   matched-rank projector 留给 E3）。

## 种子与执行
Block B seeds = 3000..3049（此前从未使用，无任何 tuning 接触）。
超参与 Block A 相同（T=3000, burn 200, m=2, K=256, η=0.02, σ=0.1, ema=0.98），
评估段扩展：on-policy 1000 + forced-E 1000 + forced-S 1000。
等级由报告器机械判定：执行后补靶→DEV-CHECK；非有限率>5%→UNRESOLVED。

## 靶（bundle/cohort_targets_B.json，冻结）
constr=解析；unconstr=solver-stable numerically optimized population
reference（du_authority(+ext)；非全局证书，如实标注）。

## 门（冻结）
- B1 capacity control：fullrank3 trapped ≤0.5%。
- B2 constr2 等价：6 格 err CI ⊂ [−0.02, 0.02]。
- B3 unconstr2 等价：6 格 err CI ⊂ [−0.03, 0.03]。
- B4 floor 方向：12 格（3δ×2 臂×…按 δ×臂 共 6 对）配对 shift CI 下界 >0；
  幅度偏差仅报告。
- B5 碰撞中介：trapped 的 2D runs 中 g12>g23 比例 ≥95%；escaped 中 g23>g12 ≥95%。
- B6 gap 链：trapped 的 forced gap<0 比例 ≥95%；escaped 的 >0 ≥95%。
- B7 回报链：各确认 δ（臂内、并 ε 池化）trapped 与 escaped 的 balanced return
  差的 bootstrap CI 上界 <0（trap 严格次优；两组各 ≥10 runs 才判，否则 N/A）。
