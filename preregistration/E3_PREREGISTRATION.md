# E3 预注册（送审版 v5，2026-08-27）——标准栈存在性实验

## v4→v5 变更日志（第四次规格审计 HOLD_FOR_PRE_RUN_CLOSURE，发布边界闭合；
## 环境与 PPO 零改动）
1. **Admission 前置门**：在任何 Pool/seed/训练之前执行；缺 stamp、dirty 部署、
   spec 或任一 E3 payload 哈希不符、文件 census 不精确、报告已存在（拒绝覆盖）
   均立即退出；输出原子写入（tmp + os.replace）。此前"缺戳拒跑"仅在 receipt
   阶段检查——审计判定正确，声明与实现不符，已改为真前置门。
2. **部署改 fresh commit-keyed 远端目录**（~/sct_e2/runs/<commit12>/），
   已存在即拒；dirty worktree 拒绝部署（非仅记录）；stamp 含逐文件 payload
   哈希（e3/*.py + tests/test_e3.py + 已部署 spec）。
3. **P3 改预冻结双侧精确二项区间**（Clopper–Pearson，α=0.05）：全成功时
   n=5 下界 0.478<0.5，故 P3 门蕴含每侧全成功至少 n≥6；bootstrap 的 [1,1]
   假区间问题消除。
4. E3_SPEC_DRAFT 的 E3-G2 行改为 Jacobian cross-talk operational analogue
   （probe-Gram 仅诊断）——清除最后的 Gram 残留。
5. **q0 与部署概率显式区分**：根头 bias 设定的是 softmax 偏好 q0；实际初始
   部署概率为 π_E(0) = ε + (1−2ε)·q0（ε=0.05 时 q0=0.1 → π_E(0)=0.14）。
   pilot 的 p0 标签指 q0。
6. 新增测试：admission 五种拒绝情形、exact job census（80+120=200 唯一）、
   精确二项区间边界值（n=5 全成功必败 / n=6 可过）。测试 11/11。
7. 声明修正："deploy.sh 部署 + 手动启动 pilot"（不声称一键即跑）。

## v3→v4 变更日志（第三次规格审计 HOLD_FOR_FINAL_GATE_ALIGNMENT，零设计扩展）
1. P1 改逐条件判定：四个 (k × 受训分支) 条件各自 ≥18/20，主门只考核受训
   分支的 competence；跨分支 competence 降为次级稳健性报告（原池化 72/80
   允许单条件失败被补偿，且跨分支 min 与协议不符——审计正确）。
2. P2 控制臂改双边 |csplit| ≤ 0.15（原单边可被 −0.9 穿过）；tail-stability
   覆盖 k=64 控制臂；主指标改按 seed 配对的 high−low 差及 bootstrap CI。
3. P3 实现预注册全文：低端 x12>x23 且高端反序 x23>x12，各 ≥0.9 且 seed 级
   bootstrap CI 下界 >0.5；文档残留的 probe-Gram 表述清除；cross-talk 明确
   标注为"理论启发的 operational analogue"，非定理直接预测量。
4. root-freeze 措辞修正："root policy loss/entropy disabled and root head
   fixed；共享 encoder/trunk 仍更新，root logits 可随表示漂移"。forced
   checkpoint 不得直接继承做自由策略实验（须重新初始化 root actor）。
5. receipt 改部署戳机制：deploy.sh 从真实 git 状态生成 stamp（commit、
   dirty 计数、spec/manifest SHA），pilot 运行时重算已部署 spec 哈希并断言
   一致（缺戳/漂移拒跑），receipt 含 raw-output SHA、seeds/job census、
   失败数、环境版本；GIT_HEAD 环境变量路径废除。
6. 新增 gate reducer 合成测试（高端反序失败 / 单条件 17/20 / csplit=−0.9 /
   控制臂不稳定四情形均正确拒绝）+ 1-seed schema-only E2E；测试 8/8。
7. 主张措辞：E3 "保留 masked-reconstruction 目标并检验相同机制方向"，
   不声称逐字复现 E2 假设类（branch-conditioned 解码偏离共享解码器假设）。

## v2→v3 变更日志（第二次规格审计裁决 HOLD_BEFORE_PILOT_V2，六项全部修正）
1. **主中介改为部署接口 cross-talk**（Jacobian |∂a_i/∂x_j|，行为层可识别、
   隐层重参数化不变）；probe-Gram 降为诊断（E3 网络类 ≠ E2 tied-linear，
   cos² 不可识别——审计正确）。
2. **branch 上下文绕行 projector**：仅 branch one-hot 在瓶颈后拼接，感官维
   （特征+nuisance）不可旁路（测试双向锁定）；披露：branch-conditioned
   解码偏离理论共享解码器假设，稀缺性主张只针对感官瓶颈。
3. **oracle 改解析式** MSE = 2σ²/(1+σ²)（被计分即 active，最优响应
   x/(1+σ²)；σ=0.30 → 0.16514）；MC 版删除；模拟-公式一致性测试锁定。
4. **forced 训练冻结 root actor**（forced 动作非策略样本，root surrogate 与
   root entropy 全部屏蔽；root head 权重不变性测试锁定）。
5. **checkpoint/resume 声明整体删除**（分钟级运行；按审计建议弃复杂恢复
   协议）；pilot 升级为完整 runner + 机械 gate reducer + receipt + CLI；
   deployment return 明确两口径（deterministic / sampled）分别命名。
6. 文档修正：floor 正文改理论式、六流 RNG、测试计数、E3_SPEC_DRAFT 的
   final 种子段冲突（final=6000 段）。

## v1→v2 变更日志（第一次规格审计裁决 HOLD_BEFORE_PILOT，六项全部修正）
1. **任务改为连续 masked reconstruction**（审计概念级发现：pair-classification
   把碰撞方向反转——S 重训练迫使区分 1/2，且 Gram² 在分类损失下非伤害中介，
   反对齐碰撞反而可分类）。终端动作 a∈R³，奖励 = −Σ_{i∈G}(a_i−Z_i)² + δ·1{E}，
   逐字保留 E2 人口目标；Gaussian-terminal + categorical-root 标准 PPO。
2. 观测加入 agent 自己根动作的 one-hot（branch memory）。
3. oracle 改为 branch-conditioned Bayes 重构 MSE（后验混合均值，冻结 MC
   seed=97/n=1e6）；先验缺失问题消除。
4. floor 改为理论原形 π_E = ε + (1−2ε)·softmax_E（floor_pi_E，测试锁定端值）。
5. 六流 RNG（init/data/noise/action/opt/eval）完全绑定：终端动作采样、
   minibatch 置换均不触全局 RNG；测试 test_bound_streams_immune_to_global_rng
   用不同全局种子扰动验证逐位一致。声明 CPU 执行（GPU 非确定性不在范围）。
6. 补全可执行组件：forced Low/High runner、competence/probe-Gram/forced-gap/
   on-policy-return 评估器（全部走 eval 流）、checkpoint/resume、receipt。
   P2 增加数值 margin（终态 P(E) 差 ≥0.5 且 tail_drift ≤0.1）、P3 中介方向
   用 probe-Gram g12/g23 序 + CI（重构任务下 E2 的 g12 预测恢复有效）。
# 主张上限：可重复存在性；不 claim prevalence / benchmark 泛化。
# 状态：规格独立审计中；审过后仅运行 competence pilot；三信号同现才冻结 final。

## 环境（bundle/e3/env.py v2，逐行可审计）
两步状态机 POMDP。t=0 固定起始观测，动作 {S,E} 选分支；t=1 恰两特征共激活
（S: {1,3}/{2,3} 各 1/2；E: {1,2}/{1,3} 各 1/2），观测 = [Z·mask +
σ_obs·N(0,1)（3 维）, nuisance（5 维）, 根动作 one-hot（2 维）]；终端动作
a∈R³ 连续重构；奖励 = −Σ_{i∈G}(a_i−Z_i)² + δ·1{E}（reward-only）。
σ_obs 默认 0.30，branch-conditioned Bayes MSE 由 oracle_recon_mse 冻结；
pilot 可调 σ_obs，调整须在 final 冻结前披露。

## 学习器（bundle/e3/ppo.py）
标准 PPO+GAE(λ=0.95)+clip 0.2+entropy 0.01+Adam 3e-4；感官编码后接**冻结
随机 rank-k 投影**（k=2 主臂 / k=64 对照，同尺寸网络同预算），branch one-hot
在投影后拼接（唯一合法旁路，测试双向锁定）；终端 Gaussian 头 + 根 categorical
头。根策略理论 floor：π_E = ε + (1−2ε)·softmax_E（floor_pi_E，端值测试锁定）；
初始偏好 p0 由根头 bias 设置。六流 RNG（init/data/noise/action/opt/eval）
按 seed 显式绑定；全局 RNG 免疫 + 全模型 state 哈希一致性测试锁定；CPU 执行。

## 臂
k=2 主臂 / k=64 capacity 对照（同 seeds CRN）；floor 干预对 ε∈{0.05, 0.25}。

## Pilot（审过规格后唯一允许的第一步）
seeds 5000–5019（tuning 段，与 final 永不相交）；两臂 × 强制占用调度
（Low/High source 训练各 200 updates）+ 自由训练小网格 p0∈{0.1,0.5,0.9}。
三信号门（全部同现才进 final）：
  P1 competence：强制分支下重构 MSE ≤ oracle_MSE/0.85（等价 ratio_to_oracle
     ≥0.85），两分支、两臂、逐 seed，20 seeds 中 ≥18 达标；
  P2 双盆地迹象：k=2 自由训练（400 updates）中 p0=0.1 与 p0=0.9 的终态
     P(E) 差的 seed 均值 ≥0.5 且逐 run tail_drift（后 1/4 训练内 P(E) 极差）
     ≤0.1；k=64 同指标差 ≤0.15（无分裂 margin）；
  P3 中介方向（部署接口 cross-talk，理论启发的 operational analogue）：
     k=2 低端 runs（p_E_end<0.5）x12 > x23 比例 ≥0.9 且双侧 Clopper–Pearson
     （α=0.05）下界 >0.5；高端 runs（p_E_end>0.5）反序 x23 > x12 同标准。
     n≥5 为收集下限；精确区间蕴含全成功时每侧至少 n=6 才可能通过。
任一不齐：停，修环境或收窄主张，不扩算力。

## Final（pilot 通过后另行冻结数值，先于运行提交）
seeds 6000+（fresh）；p0 网格 × ≥30 seeds × 两臂 × 两 ε；门按 estimand 审计表
（E3-G0..G5，见下）；统计对象全冻结（50% crossing、seed 聚类 bootstrap、
分层报告禁池化、UNRESOLVED 规则沿用 Block B 修正版）。

## Estimand 推导审计表（审计对象）
见 E3_SPEC_DRAFT.md 附表（G0 competence / G1 双盆地 / G2 碰撞中介双侧 /
G3 gap 链禁池化 / G4 floor 方向 / G5 on-policy 回报链；balanced return
禁用于回报链；每门列理论量→观测量→对比量→实验单位→理论零量检查）。

## 审计资产
测试 11/11（bundle/tests/test_e3.py，服务器 venv 运行）：环境分布、解析
oracle 模拟一致性、理论 floor 端值、感官 no-bypass + branch 旁路存在、
forced 模式 root loss disabled 且 head 固定（共享表示仍漂移）、全局 RNG
免疫 + state 哈希确定性、gate reducer 四情形合成拒绝、schema E2E、
admission 六种拒绝（含 spec 漂移）、exact job census、精确二项区间边界值。
