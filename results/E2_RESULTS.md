# E2 Final 结果账本

## CURRENT AUTHORITATIVE STATUS（2026-08-27 第五轮审查后；本节为唯一权威摘要，
## 以下全部为 append-only 历史，含已收回的陈述，写论文时禁止从历史节直接摘取）

- **E2 core（可解类）**：G1–G5、G7 注册 PASS；G6 注册 FAIL（estimand 错误）+
  G6' append-only PASS（盆地宽度在 λ_crit 坍缩）；κ 系列（~1/κ 收敛）、
  cold-restart/repeated-solve 对照、多初始化稳健性 = observed/development。
- **Cohort Block A**：强 development evidence（16 pre-frozen PASS；其余重标）。
- **Cohort Block B（fresh seeds，权威确认层）**：边界/表示类/capacity control/
  floor 方向 = PASS；collision-gap 链 = PASS_WITH_QUALIFICATION（分层 12 层
  两侧 ≥0.99，唯一例外 unconstr/δ=0.6/ε=0.25 的 trap 侧 gap 0.718）；
  **确认级回报链 = HOLD**（B7 estimand 错误：balanced return 为理论零量；
  on-policy 回报差 12/12 命中 −(1−2ε)δ，等级 DEV-CHECK）。
- **D_u**：solver-stable numerically optimized population reference（无全局证书）。
- **κ=80 全网格**：outcome-informed visualization supplement，非确认性证据。
- **旧 E2c（30 seeds）**：被 Block B 取代，仅作历史。
- **E3：FINAL ALL_PASS（c0aca73，430/430，seeds 6000-6199）**——标准 PPO
  栈存在性达确认级：trap 96/100（CP 下界 0.901）、escape 100/100、双容量
  对照零 trap、endpoint-conditioned competence 双侧全过、中介/gap 双侧
  196/196、return 三件套全正（rank gap 0.225 / 盆地差 0.131 / DiD 0.133）。
  主张上限：受控标准 PPO 系统中的机制一致存在性证据（非 prevalence、非唯一
  中介识别）。历史：pilot-v1 FAIL（oracle 参照错配）、pilot-v2 FAIL（Q2 功效）
  均保留在册。
- **已收回、禁止摘取的历史陈述**："19/19 frozen PASS"、"unconstr balanced
  return 真实不对称"、"精确跟随自身理论/完全归因"、旧 E2c 的"CRN 配对/位级
  复现"、"重组带确认为真实一阶相变"。

---（2026-08-27 凌晨；门定义见 E2_FROZEN.md，冻结未改）

## 注册门判定
- G1 对照（dim3 无 trap）：779/779 PASS
- G2 域归属：**披露**——冻结文档 G2 行误写 m=80，网格节按 m=20 注册并执行；
  按实际执行网格判定：746/746 = 100%（33 格在排除带内）PASS
- G3 分离点（m=80，19δ）：max |err| = 0.0044 PASS
  - κ 系列（observed）：max|err| {κ=1: 0.152, 5: 0.055, 20: 0.019, 80: 0.0044}
  - cold-restart 对照（observed）：一致率 0.814（闭环历史依赖性有实质贡献）
- G4 熵：τ* ∈ 开区 3/3 PASS（1.724/1.393/1.115）；sink 误差 ≤6e-16 PASS
- G5 floor：4/4 |err| ≤ 0.0006 PASS
- G6 ortho：**FAIL（如实入册）** |λ*(80)−λ_crit| = 0.169/0.101 > 0.1；
  δ=0.6 low-endpoint gap 检查因轨迹逃逸测到 −s_λ 侧（该侧精确 1e-15）。
  **Post-hoc 诊断**（单独标注，不改门）：门把观测量编码错了——测的是
  "盆地是否包含 p0=0.03"，命题断言的是"盆地是否存在"。正确观测量：盆地宽度
  s(λ) 随 λ 单调收缩 0.461→0.0058，在 λ_crit=2.798 处坍缩到零
  （results_final/posthoc_G6_basin_width.json）。iff 预言在正确观测量下锐利成立；
  如需门级 headline，须预注册 G6' 重跑。
- G7 replay：|λ*(80)−pred| = 0.012/0.005 PASS

## 神经臂（observed 层，预注册无门）
- dim2 吸引域边界 vs 含 floor 修正预言：δ=0.3 时 0.455 vs 0.456（命中）；
  δ 增大时边界系统性低于绝热预言（0.15: 0.455/0.478; 0.45: 0.410/0.435;
  0.6: 0.275/0.415）——方向与 κ 系列一致（远离绝热时 trap 盆地收缩）。
- dim3 对照：2520 个 run 中 1 个 trap（δ=0.15 最弱驱动处），2519/2520 干净。

## 计时
core 510s（CPU）/ interventions 754s（CPU）/ neural 8 cells 41s（GPU）。

## 跟进轮（2026-08-27，冻结见 E2_FROZEN_ADDENDUM.md，提交 868883d 先于运行）

- **G6'（append-only）双 δ PASS**：s(λ) 在全新冻结 λ 点上严格单调收缩；外推
  坍缩点 λ̂=2.824/1.526 vs 解析 2.798/1.523；末点宽度 0.0052/0.0046。
  原 G6 保持 FAIL 不覆盖。
- **repeated-solve 对照 40/40=1.000**：解除 cold-restart 混杂——0.814 的退化
  源于内环未收敛而非历史移除。发现修订：warm-start 的价值是计算性的
  （携带历史使 m=20/外步即贴住绝热预言；无历史同预算失败，百倍预算才恢复）。
- **四臂拆分**：δ=0.6 边界收缩在全确定性 pop/exact 臂同样出现，K 256→1024 不
  变——排除估计噪声。post-hoc 时标检查 m∈{2,8,32} 不恢复——排除时标。
  post-hoc 类别检查：无约束类自身绝热分离点（人口优化可计算）含 floor 映射
  预言 0.323/0.479，实测 [0.315,0.335]/0.476——**偏离被完全归因于预言的
  表示类错配，类别正确后预言落点**。神经臂结论升级：最小学习系统跟随其
  自身表示类的可计算理论。

## 独立复算与估计量修订（08-27，审计员报告 results_final/independent_recheck.md）

- 独立复算：全部门判定与分母精确复现，时序账本核验通过，零判定级分歧；
  6 条呈现级修正已采纳（rebuild 脚本改为全量重算并补 G6 端点子检查；
  repeated-solve 报 42/42=1.000）。
- 神经边界估计量统一为 50% 过零线性插值。修订表（vs 无约束类绝热预言，
  D_u 以 12 trials×8000 步精算）：
  δ=0.15: 0.480 vs ~0.490；δ=0.30: 0.480 vs ~0.480（命中）；
  δ=0.45: 0.435 vs ~0.475（−0.04，唯一外点）；δ=0.60: 0.315 vs ~0.323（命中）。
- **开放点**：D_u(p) 在 p∈[0.47,0.50] 陡降（几何重组带），δ∈(0,0.5) 的类别
  分离点全部挤在带内；δ=0.45 的 −0.04 错位恰在带上，候选解释为动力系统穿越
  重组带的滞后/选择效应。留给中等神经桥 cohort（密集 δ∈[0.4,0.55] 网格）。

## 外部复审修正（08-27，第二轮 GPT 审查，四条全部采纳）

1. **跟进轮标签降级**：868883d 仅冻结了 addendum 文本，G6'/四臂/repeated-solve
   的实现代码与结果同入 1cd30f8——按实改称 **credible development follow-ups**，
   非严格冻结后确认实验。严格冻结（代码+命令+receipt 先于结果）自中等 cohort 起执行。
2. **收回 CRN 与位级复现声明**（神经臂）：neural.py 中 randn_like 未绑定 run
   generator（噪声流不可由记录 seed 复现），且 dim2/dim3 从同一 generator 消耗的
   随机数数量不同，两臂并非真正共享随机数。原始观察保留为 observed；下一轮实现
   按四流分绑（init/data/noise/action 各自 generator）修正。
3. **类别归因措辞降级**：改为"三个边界（δ=0.15/0.3/0.6）接近表示类匹配的人口
   预测（≤0.01），几何重组带（δ=0.45）仍存在 −0.04 的系统性偏离"。不称"精确
   跟随/完全归因"。D_u 生成代码此前仅存在于会话内，已补交
   bundle/scripts_unconstrained_class.py。
4. **单初始化缺口**：主 E2 全部 run 用 seed=0，多初始化稳健性检查
   （10 seeds × 2 m × 5 δ，development 标签）已补跑，结果见
   results_final/multiseed_robustness.json。

## 多初始化稳健性（development 标签，10 V-init seeds × κ∈{20,80} × 5δ）

- **κ=80（门所在档）：结论对表示初始化稳健**——分离点误差 sd 0.002–0.004，
  全部 100 个 (seed,δ) 点最差 |err|=0.0133，逐 seed 均满足 G3 的 0.02 门。
- κ=20：种子间散布 sd≈0.016、最差 −0.057，与该档的系统性偏移同量级——
  κ 收敛图应加 seed 带（图套件重做时执行）；seed=0 的原 κ 系列数值在
  seed 分布之内，不改变收敛结论。

## 中等 cohort Block A（08-27 晨；冻结 3174071 先于执行；gate 表可由
cohort_report.py 从 raw 一键重建）

**19/19 门 PASS，9900 记录零失败。** C1 对照 0/3300；C2 constr2 六格 CI 贴
解析预测（最大点偏差 0.007）；C3 unconstr2 六格贴 du_authority 预测（点偏差
≤0.012，δ=0.6/ε=0.15 为 grid-edge 哨兵格，PASS 按冻结规则）；C4 floor 配对
shift 六格方向全对、量级在冻结容差内。**诚实注记**：C4 有三格 bootstrap CI
不含点预测（constr2@0.3 实测 shift 0.001–0.004 vs 预测 0.0125 最显著）——
方向正确、量级偏小，候选解释为采样学习器中的有限速率/估计抹平，不掩饰。
探索带（outcome-informed，无门）：两个 2D 臂边界随 δ∈[0.40,0.55] 压缩
（0.45→0.39/0.41），fullrank3 全程零 trap；作为重组带研究的观察数据单独报告。

## 论文状态（08-27 晨）
图 v2 两张（closed-loop 三联 / learned-bridge 双联）已插入 main_v2.tex 实验节，
一句话 caption，claims-safe 引图句 + TODO 注释占位，E2 完整散文待作者对图
签字后撰写。编译零错误。

## 账目更正（08-27，第三轮外部审查，append-only；Block A 数据不动，只改等级）

外部审查裁定成立，"19/11 frozen PASS" 收回。机械化规则已写入报告器：
**靶晚于执行时刻写入的门自动标 DEV-CHECK；bootstrap 非有限率 >5% 的格自动
UNRESOLVED；等级由代码判，不再手写。** 更正后账目：

- **11 格严格 pre-frozen PASS**（C1 + C2×6 + C3×4）；
- **4 格 C4 重标 DIRECTION-ONLY**（冻结容差宽于信号，等价性检验不成立；
  四格方向全部为正，δ=0.15 两格的预测 shift 本就小于网格分辨率，设计缺陷）；
- **2 格 DEV-CHECK**（δ=0.6 无约束靶执行后补写：边界格 consistent，
  shift 格 direction+ 且 CI 不含点预测）；
- **1 格 UNRESOLVED**（grid-edge；修正统计后暴露 99% bootstrap 非有限，
  原"CI [0.2713,0.2730]"是幸存者过滤伪精度，作废）。
- D_u 全文改称 solver-stable numerically optimized population reference；
  "重组带为真实特征"降级为"在该求解协议下稳定的陡峭特征"。
- 图 2 判定为旧数据草图（画的是 E2c 30-seed，非 Block A 50-seed 三臂），
  终稿图待 Block B。

Block A 的科学结论按审查员措辞定格：学习盆地边界随表示约束系统性移动、
接近各自表示类的人口参考、容量对照干净、方向性 floor 响应一致——
强 development evidence，确认等级留给 Block B。

## Block B（08-27；冻结 7bacaee 先于执行；seeds 3000-3049 首次使用；
门表由 cohort_report_B.py 从 raw 一键重建）

**35,100 记录零失败；27 门中 24 PASS，3 FAIL 如实入册。**
- B1 capacity control：0/11,700。
- B2 constr2 等价 6/6：err CI 最宽 ±0.007，最窄 ±0.002（真等价门通过）。
- B3 unconstr2 等价 6/6：err CI 全部落于 ±0.023 内（margin 0.03）。
- B4 floor 方向 6/6，且次级幅度三个 constr δ 全部压点：0.0177/[0.0177,0.0209]、
  0.0351/[0.0345,0.0399]、0.0676/[0.0670,0.0716]。
- B5 碰撞中介 1.000/0.999；B6 gap 链 0.992/1.000。
- **B7 constr2 三格 FAIL（留账）+ 诊断**：balanced 50/50 回报在约束类端点
  对称是可证性质（两端点最优码 D_E+D_S 均为 1），门测了一个理论为零的量，
  实测 ≈0 恰与理论一致——estimand 编码错误（与 G6 同型，第二次）。
  正确回报链的 DEV-CHECK：on-policy 回报差在全部 12 格命中 floor 推论
  预言 −(1−2ε)δ（例 δ=0.3/ε=0.05：预言 −0.2700，CI [−0.2709,−0.2684]）。
  unconstr2 的 B7 3/3 PASS 反映无约束最优的真实不对称（observed）。
- 终稿图两张（e2_core_final / e2_bridge_final）已按第三轮图评重构并替换入
  main_v2.tex（log-κ+1/κ 参考线+seed ribbon；阈值处 s(λ) 坍缩；Block B 边界
  vs 双类预测曲线+CI；floor shift y=x 配对图；颜色语义分离），编译零错误。

## 第四轮外部审查执行（08-27，append-only）

两条数值指控逐位核实成立并采纳裁决：
- **B7 重定级**：constr2 三格 = 注册 FAIL；unconstr2 三格 = nominal PASS 但
  因 ε-pooling（Simpson 伪影，within-ε 差值 +0.0001~+0.008，池化后 −0.009~
  −0.023）与错误实验单位（record 级 bootstrap 非 seed 聚类）**不可解释**。
  收回上轮"unconstr 真实不对称"说法——分层显示两类 balanced return 均近似
  端点对称（理论零量）。**confirmatory return link：HOLD**；on-policy 12/12
  命中 −(1−2ε)δ 保持 DEV-CHECK。
- **B6 重定级**：pooled PASS（注册口径）；分层 11/12 层 ≥0.999、一层 0.718
  （unconstr/δ=0.6/ε=0.25）——链结论为 PASS_WITH_QUALIFICATION。该异常层与
  B7 唯一异常层重合（训练 occupancy 底 0.25 逼近该类陡降带），假设记录不外推。
- 分层×seed 聚类重报器已入库（cohort_report_B.py 追加节，向量化聚类 bootstrap）。
- 图 v3：两张 2-panel 主图（κ=80 统一时标；真实点估计+非对称 CI；capacity
  数字入 caption；y=x 降为次级参考并写入 caption；标题降级为
  "Representation constraints shift basin boundaries"）；阈值坍缩图移附录并
  与注册 G6 FAIL 同图展示。编译零错误。
- **E3 保持 HOLD**：预注册前置件 = 逐门"理论量→观测量→对比量→实验单位"
  推导审计（含理论零量子群检查），已写入 E3 规格（见 E3_SPEC_DRAFT.md 附表）。

Block B 科学结论定格（审查员措辞）：边界/表示类/capacity control/floor 方向
PASS；collision-gap 链 PASS_WITH_QUALIFICATION；确认级回报链 HOLD。

## E3 competence pilot（08-27，commit 57170e1e9049，seeds 5000-5019）

**机械判定 all_pass=FALSE，按预注册边界：结果保留、就地停止、零调整。**
200/200 jobs 零失败；admission 放行；receipt 完整（raw sha 7756b62b…）。
- P1 FAIL（结构性）：k=64 两分支 20/20（比值 0.95）；k=2 两分支 0/20
  （比值 0.678，受训分支）。
- P2 FAIL：split=0.18（CI [0.045,0.36]，方向正确但低于 0.5 margin）；
  控制臂 split≈0.00002（极干净）；drift 全过。
- P3 FAIL（技术性）：低端 n=4<6 不可能过精确区间门；但两侧方向比例均
  1.000（低端 4/4，高端 20/20，高端 CP 下界 0.832）。
运维披露：首次启动因 20×24 线程超订中止（无结果产生），线程钉扎补丁为
commit 57170e1；单格 10-20 秒。
后续（按协议属新 development version，须外部审查）：P1 的 0.85×oracle 门
把 rank-2 系统对到了全信息参照——与 E2c 的类错配教训同种；候选修正 =
类匹配 competence 参照。此判读为解释性备注，非本轮结论。

## E3 开发筛选（08-27，outcome-informed，commit 1a918346，dev seeds 5100-5117）

216 runs 零失败。三旋钮方向全部与 E2 理论同构：
- **时标比是主导旋钮**：root_lr_scale 1.0→0.3→0.1（actor 变慢=更绝热）在
  每个 (δ,q0) 单调抬升 trap 率——E2 的 κ 故事在标准 PPO 栈复现；
- **小 δ 加宽低盆地**（E2 预言方向）：δ=0.15 全面高于 0.3；
- 低 q0 更深入盆地。
不变量全程保持：escape_high 12/12 配置 = 1.00；k64 对照 36 runs 零 trap；
类参考 competence 0.97-0.99（rank-2 类参考度量有效）；中介方向在所有
有 trap 的配置 = 1.00。
多个配置同时满足全部筛选目标（如 δ=0.15/rls=0.3/q0=0.05：
1.00/1.00/0.00/0.98/1.00）。选型与 fresh pilot 冻结待外部审查。

## 第十轮审查执行（08-27）：三项收回 + 筛选定级

裁定 PASS_DEVELOPMENT_SELECTION / HOLD_TIMESCALE_CLAIM / FRESH_PILOT_REQUIRED
/ FINAL_HOLD 全部接受。收回：(1)"E2 时标在 PPO 复现"→ 改称 root-head
learning-rate sensitivity（共享干线仍全速更新且 root 梯度经干线反传，非干净
两时标干预）；(2)"整体 competence 0.98"→ min(S,E) 美化，改分支分报：favored
分支 MSE≈0.2395（构造参考 2% 内），disfavored≈0.744——不对称即碰撞代价，
应作现象呈现；(3)"216 独立 runs"→ 高端 60 条中 30 条为确定性重复计算（cfg.q0
不进高端运行），已披露；推荐配置 10/10 的 CP 下界仅 0.69，可重复性待 fresh
pilot。MANIFEST 在 1a918346 提交时未再生成（遗漏属实），本轮修复。
筛选定级：配置发现完成（40 个配对序列零反转、79/79+41/41 中介与 gap 方向、
k64 零 trap），非可重复性证明。

## E3 Pilot-v2（08-27，冻结 2dd30994，seeds 5200-5289，165/165 零失败）

**机械判定 all_pass=FALSE（6/7 门 PASS；按协议保留停止）。**
- Q1 PASS 30/30：favored 分支 0.2399 vs 构造参考 0.2351（2% 内），
  disfavored 0.743 如实呈现（碰撞代价）。
- **Q2 FAIL**：trap 28/30 = 0.933，CP 下界 0.779 vs 门 >0.8（差 0.021；
  冻结文本已预写 "28/30 不过"，判定在预定分辨率之内）。
- Q3 PASS：escape 30/30（CP 下界 0.884）。Q4 PASS：对照 0/30。
- Q5/Q6 PASS：中介与 gap 双侧全对（trapped n=28 全部 x12>x23 且 gap<0）。
- Q7 PASS：CRN 配对容量 return 赤字 0.223，CI [0.209, 0.232]——
  **标准栈的 trap→return 链首次以预注册门确立**。
- Comparator（次级）：rls=1.0 同 seeds trap 0.60 vs 0.93——LR 敏感性在
  fresh seeds 复现。次级配置一致（0.933/escape 1.0/对照 0）。
后续属新 development version（外部审查）：候选 = 加大 n（功效计算后冻结）
或 rls=0.1（筛选数据 20/20）。此判读为解释性备注。

## 第十一轮审查执行（08-27）：PILOT_V2_FAILED_AS_REGISTERED 定级 + 三修正

- **功效重述**：门在自身点估计 p=0.933 下仅 39.7% 通过功效（n=30 需 29/30），
  28/30 近模态——"差 0.021" 框架收回；FAIL 是设计功效性质，非运气边缘，
  亦非机制反证。不拼接补种。
- **Q1 min() 复发**：pilot2 门重新引入了筛选轮已纠正的 min(S,E)——两个逃逸
  seed 被静默改取 E 分支。工程教训入册：**修正必须全库 grep 清扫，不止
  被点名处**。final 改语义化分支（init-defined：low→S、high→E），双分支
  全报。
- **Q7 改名** matched rank-capacity removal return gap（rank 干预同时动容量/
  重构/trap/策略）；确认级 return 链需 low-vs-high k2 + rank gap + DiD 三件套。
- rls=0.1 切换建议收回（outcome-informed winner selection 风险）。
- 定级：在全新 seed 上出现高频（28/30）、机制一致（58/58 方向）、有显著
  回报后果（rank gap 0.223）的标准栈 trap；重复率门未过，确认待 final。

## E3 FINAL（08-27，冻结 c0aca73，seeds 6000-6199 唯一一次使用，430/430）

**机械判定 ALL_PASS = TRUE（7/7 门 + tail）。**
- F1 endpoint-conditioned competence：trapped 96/96 S 达标、escaped 100/100
  E 达标；跨分支报告呈完美对称（low: S=0.258/E=0.722；high: S=0.740/
  E=0.238）——被占分支学会、被弃分支退化，表示牺牲双向可见。
- F2 trap 96/100，CP [0.901, 0.989]（门 0.8，余量充分）。F3 escape 100/100。
- F4 双容量对照 0/100 + 0/100。F5/F6 中介与 gap 双侧 96/96 与 100/100。
- F7 三件套全正：rank gap(low) 0.225 [0.219,0.230]；盆地差 0.131
  [0.125,0.135]；basin-specific rank interaction 0.133 [0.127,0.137]。
- comparator（次级）：rls=1.0 trap 0.667——LR 敏感性第三次复现。
主张上限（审查裁定措辞，冻结）：**受控标准 PPO 系统中的机制一致存在性证据**；
不 claim 一般 PPO prevalence、不 claim 唯一中介识别。
