# E2 冻结附加（append-only，2026-08-27；原 E2_FROZEN.md 与其判定不变，G6 永久 FAIL）

## G6'（append-only，正确估计量：盆地存在性）
估计量：盆地宽度 s(λ) = 低盆地边界（p0 二分，lo=0.002, hi=0.6），m=80, T=8000。
冻结 λ 点（与 post-hoc 诊断点尽量不重叠）：
- δ=0.3（λ_crit=2.7978）：Λ' = {1.5, 2.0, 2.3, 2.5, 2.65, 2.72, 2.77, 2.785}
- δ=0.6（λ_crit=1.5235，全新）：Λ' = {0.8, 1.1, 1.3, 1.42, 1.48, 1.51}
门 G6'：(i) s 在 Λ' 上严格递减；(ii) 末三点线性外推零点 λ̂ 满足 |λ̂−λ_crit|≤0.05；
(iii) s(Λ'末点) ≤ 0.015。G6' 通过不覆盖 G6 的 FAIL。

## E2c 四臂拆分（observed，无门；分离优化时标 vs 估计噪声）
臂：{population encoder, SGD encoder} × {exact gap(由 M=W^T W 解析), sampled EMA gap}。
δ∈{0.3, 0.6}；p0 = 含 floor 预言边界 ±{0.02,0.05,0.10,0.15} 及边界点（9 点）；
30 seeds（A 臂确定性，1 次）；sampled 臂 K∈{256,1024}；其余超参同 E2_FROZEN 神经段。
报告：各臂经验边界与跨臂差分。命名纪律：E2c 全文称
minimal learned-representation bridge，不称一般 neural RL 证据。

## Repeated-solve 对照（observed，无门；解除 cold-restart 的未收敛混杂）
每外步从随机初始化求解至黎曼梯度范数 ≤1e-8（上限 5000 步），不继承 V_t。
δ∈{0.3,0.6}，p0 21 点，T=4000（T 不敏感性已由 dev B 建立，披露此差异）。
报告：与 sign(p0−p_δ) 一致率，对照 warm(1.000)/cold(0.814)。
