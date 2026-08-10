# OnlineSPEC baseline

[English](../en/onlinespec-baseline.md) · [首页](../../README_zh-CN.md)

## 状态与边界

OnlineSPEC 是一个重要的、已注册的对比 baseline，但不属于 Static/TTS/L0 核心假设，
也不进入其正式速度 gate。它拥有独立的 tuning 数据、selection artifact、配对
confirmation queue、性能表、GPU attestation 与诊断分析。其结果不能选择核心方法，也
不能改变核心 gate 的 `PASS`、`BLOCKED` 或 `UNMEASURED` 判定。

本实现针对论文中的测试时 drafter 更新抽象进行 clean-room 重写。依据是
[OnlineSPEC 论文](https://arxiv.org/abs/2603.12617v2)，并将官方仓库固定在 commit
[`e58f82e`](https://github.com/ZinYY/OnlineSPEC/tree/e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b)。
审计时该 commit 没有项目级许可证文件，部分单独文件带有各自声明；LightCone-Spec
没有复制任何源文件、checkpoint 工具或训练脚本。

机器可读审计记录为
[`manifests/provenance/onlinespec_source_audit_v2.json`](../../manifests/provenance/onlinespec_source_audit_v2.json)。
它绑定上游 commit、Git tree、本次对照使用的全部实现文件 SHA-256、三套架构相关流程以及
保留/拒绝的设计决策。OnlineSPEC 实验 manifest 通过内容哈希绑定该审计；一旦源码解释
发生变化，实验身份也必须变化。

Online-LR 的 reasoning-level DPO 流程不会被描述成 token-level drafter baseline。
LightCone-Spec 实现的是能够在同一 speculative-decoding 反馈与 exactness 合同下比较的
三种在线学习规则：OGD、optimistic OGD，以及 OGD 专家上的 Hedge。

GPU 状态为 `UNMEASURED`。本文描述协议与实现，不提供性能结果。

## 源码审计

| 上游思路或实现细节 | LightCone-Spec 的处理 |
|---|---|
| 先对一个 chunk 预测，得到反馈，再为下一个 chunk 更新 | 保留为严格的 prequential 生命周期 |
| OGD learner | 根据已发表的投影梯度公式重新实现 |
| Optimistic online learning | 实现论文的双状态投影更新，不用 momentum 冒充 |
| 不同学习率 learner 的 ensemble | 保留为相互独立 learner 与 cumulative-loss Hedge |
| target-to-draft KL/CE 监督 | 复用核心运行时的 semantic mask 与 frozen target |
| 每个 chunk 启动子进程并写磁盘 checkpoint | 改为 GPU-resident transactional candidate 与固定地址发布 |
| 在 CPU 合并 checkpoint 用于 ensemble inference | 改为在一种显式绑定 parameter class 中形成设备驻留加权 decision |
| 硬编码 device 与数据路径 | 改为不可变 schema、model lock 与渲染的 launch plan |
| 更新失败或 OOM 后跳过并继续实验 | 拒绝；安全失败会被记录并使对应证据失效 |
| 未计入的同步与训练时间 | 改为异步 CUDA 计时、exposed-update 账本与独立 profiler |

固定版本的官方仓库中存在不止一套 EAGLE ensemble pipeline。与论文一致的
`pipeline_ens.py` 和 `pipeline_eagle3_ens.py` 会保留相互独立的 base learner，并根据
累计 loss 形成下一轮 ensemble；较旧的 `pipeline_hedge.py` 变体只使用上一 chunk 的
loss，而且每轮训练前都从当前 meta checkpoint 重置所有 learner。后者不等价于
cumulative-loss Hedge，因此本项目不复刻它。在审计 commit 上，README 示例和两个
`eagle-ens.sh` 入口仍调用旧的 `*_hedge.py`，而不是累计版本的 `*_ens.py`。类似地，
README 把 momentum 称为“optimism”，官方 Hydra 路径实际实现的也是 momentum SGD，
而论文给出的是使用历史 gradient hint 的双状态 optimistic update。LightCone-Spec 实现
已发表的状态转移，不会把 momentum 重新命名为 optimistic online learning。

论文 recipe 与固定源码默认值之间还存在多处差异：附录写 Online-LR global batch 为 16，
而 `LR/pipeline.py` 启动 12；附录称 EAGLE 使用 Adam，而 `EAGLE/train.py` 实际构造
AdamW；附录把 Hedge meta rate 设为 10，而论文对齐的源码 pipeline 在另一种累计 loss
尺度下默认 0.2。这些是 provenance 事实，不能在看到 confirmation 数据后静默选择。
LightCone-Spec 只在注册的 tuning window 上选择归一化 loss 对应的 learner/meta rate。

### 上游三种实例不能互换

官方项目是由三套架构相关系统组成的工作区，而不是一个可以直接替换的 optimizer。下表
记录了在固定 commit 上审计到的完整 test-time 更新边界。

| 实例 | 预测对象 | 固定版本源码中的反馈与更新 recipe | 在线单位与发布 | LightCone-Spec 的处理 |
|---|---|---|---|---|
| Online-LR | Lookahead Reasoning 使用的独立 reasoning draft LM | 从 verifier 生成 chosen/rejected response；DPO 使用 AdamW、β=(0.9, 0.95)、学习率 5e-7、DPO β=0.1 和 3 个 epoch。launcher 使用 global batch 12、micro batch 2。 | 每 25 个样本为一个 chunk；DeepSpeed 子进程重写模型目录。 | 记录 reasoning-level preference pipeline，但不把它重命名为 token-level OGD。它需要独立 judge、数据与显存协议，不属于本注册 baseline。 |
| Opt-Hydra | 多头 Hydra draft head | feature reconstruction 加 teacher/token loss。发布源码中的“optimistic”路径实际使用 momentum 0.9、学习率 0.1、3 个 epoch 的 SGD，而不是论文双状态转移。 | 每 80 个样本为一个 chunk；跨 chunk 从磁盘加载 trainer checkpoint 与 optimizer state。 | 在受支持 drafter 参数上实现论文双状态 optimistic learner；不把 momentum 当成数学意义上的 optimism。 |
| Ens-EAGLE / EAGLE3 | 三个相互独立的 EAGLE 或 EAGLE3 draft head | 不同学习率分别训练。注册脚本中 EAGLE 为 3e-5/6e-5/1.2e-4、5 个 epoch，EAGLE3 为 1e-4/2e-4/4e-4、2 个 epoch；不同源码变体对累计 loss 与仅上一 chunk loss 的处理并不一致。 | 每 40 个样本为一个 chunk；评估前在 CPU 加载并合并 checkpoint。 | 保留相互独立的 projected-OGD expert 与 cumulative-loss Hedge，但在设备上更新并形成同一 decision class 内的加权 decision。expert backward 逐个流式执行，因此同时只保留一份 expert gradient scratch。 |

因此，本项目中“完整实现”具有精确定义：用于同一 speculative-decoding 对比的三种已发表
online learner 转移——OGD、双状态 optimistic OGD 与 cumulative-loss Hedge——均具有完整
schema、runtime、tuning、confirmation、telemetry 与安全实现。它不表示 Online-LR、Hydra
与 EAGLE 在架构上被强行做成相同系统，因为那会同时改变模型对、监督信号和系统预算，
无法再单独比较 online learner。

### 论文、固定源码与 LightCone-Spec 的逐项对比

下表是决定保留或舍弃哪些设计时采用的实现级审计；它只讨论 test-time drafter 更新，
不包含任何 benchmark 结果。

| 关注点 | 论文 | 固定版本官方源码 | LightCone-Spec |
|---|---|---|---|
| 在线单位 | 一轮 predict-feedback-update | 数据集 chunk：EAGLE 40、Hydra 80、reasoning 25；最大序列长度 2,048 | 一轮带版本的 speculative update；stride 显式配置并调优 |
| OGD 状态 | 一次 projected gradient step | Online-LR 使用 AdamW；EAGLE 训练同样使用 AdamW | 精确 projected SGD 状态，每次更新只产生一个事务式 proposal |
| Optimism | anchor 加上一轮 gradient hint | Hydra 使用 momentum 0.9 的 SGD | 精确实现论文 anchor/hint 转移，并记录 hint error |
| Ensemble | 独立 OGD expert 与累计 loss 指数加权 | `*_ens.py` 符合该结构；较旧 `*_hedge.py` 不符合 | GPU 常驻独立 expert；各自在自己的参数点计算 loss/gradient；累计 loss Hedge |
| EAGLE objective | 理论损失为 cross entropy | feature SmoothL1 加 token-distribution loss，配合 AdamW 与多 epoch | 对 exact semantic mask 计算 target-to-draft KL/CE，使各后端共享同一反馈合同 |
| Gradient clipping | 有界 gradient 假设 | EAGLE 源码使用 value clipping，各后端 recipe 不一致 | 每个 expert 独立做 global-norm clipping，配置与证据绑定 |
| 参数发布 | 抽象的下一轮 decision | 子进程训练并替换 checkpoint | functional candidate，在固定显存地址原子发布 |
| Ensemble merge | 参数 decision 的加权组合 | CPU 加载、平均、保存 checkpoint | GPU 上在注册的 Full 或 LoRA coordinate class 内形成加权 decision；不经过磁盘或 CPU merge |
| 历史 KV | 未定义 | 没有为更新前 KV 给出显式版本合同 | 历史 KV frozen、detach 且带版本；只有未来 KV 使用新发布版本 |
| Exact sampling | 按 target/draft likelihood ratio 验证 | 已发布实验大多使用 greedy decoding | 保留实际 proposal distribution，用于 exact rejection sampling |
| 失败处理 | 未定义 | 部分训练路径会跳过坏样本、NaN 或 OOM batch | 对受影响 cohort fail closed，并使不安全证据失效 |
| 计时 | 加速过程包含在线演化 | 逐问题 TPS；部分脚本另加 train time | 配对 clean-server goodput，并同时记录 update/barrier/overlap/HBM |

本项目不会直接复制源码 recipe 中 optimizer 或 Hedge meta learning rate 的数值。官方的
累计 epoch loss 与本项目归一化的逐轮 loss 尺度不同，直接复制 epsilon 并不能保持相同的
指数权重。learner rate 与 meta rate 都只能在已注册 tuning window 上选择。

由于每个 OnlineSPEC gradient 都经过 global clipping，其 learning rate 表示参数空间位移
尺度，而不是 Adam 风格的逐坐标步长。注册的 OGD/optimistic 网格因此使用
`1e-4, 1e-3, 1e-2, 1e-1` 和 stride `20, 40, 80, 160`；Hedge 使用从
`1e-4`、`1e-3` 或 `1e-2` 开始的有序三元组，stride 为 `40, 80, 160`。这些只是
tuning-only 协议边界，不是已报告的最优配置。

## 在线 learner

令 $K$ 为允许的参数集合，Π 为其 Euclidean 投影，$w_t$ 为 proposal 第 $t$ 轮使用的
decision，$g_t=\nabla\ell_t(w_t)$ 为 verification 后获得的 loss gradient。可执行合同中，
每个 learner 会先独立对 $g_t$ 应用配置绑定的 global-norm clip，再执行下述转移。

Projected OGD 为

\[
w_{t+1}=\Pi_K(w_t-\eta g_t).
\]

Optimistic learner 保存 anchor \(\hat w_t\) 与 hint \(h_t\)。其发布 decision 与反馈后
转移为

\[
w_t=\Pi_K(\hat w_t-\eta h_t),\qquad
\hat w_{t+1}=\Pi_K(\hat w_t-\eta g_t),\qquad
h_{t+1}=g_t.
\]

下一轮 decision 为

\[
w_{t+1}=\Pi_K(\hat w_{t+1}-\eta h_{t+1}).
\]

对于 Hedge，专家 $i$ 拥有各自的参数、gradient、学习率 ηᵢ 和累计 loss
$L_{t,i}$。每个专家都在自己的 decision 上接受评估并更新后，

\[
p_{t+1,i}=\frac{\exp(-\gamma L_{t+1,i})}
{\sum_j\exp(-\gamma L_{t+1,j})},\qquad
w_{t+1}=\sum_i p_{t+1,i}w_{t+1,i}.
\]

专家之间绝不复用 gradient。使用 `weight_update_mode=full` 时，$w$ 是稠密 drafter
parameter vector，decision 对应加权 parameter averaging；使用
`weight_update_mode=lora` 时，$w=(A,B)$ 是相同 rank 下已注册的 factor-coordinate
vector。后者在更低显存的 decision class 上执行相同 Hedge 转移，但它不等价于对稠密
update $BA$ 求平均，因此 manifest、layout 与 evidence 会严格区分两种 class。

Target-to-draft cross entropy 与
\(D_{\mathrm{KL}}(p_{\mathrm{target}}\Vert q_{\mathrm{draft}})\) 只相差不依赖
drafter 参数的 target entropy，因此二者的 drafter gradient 相同；这个共同加法项也不
改变 Hedge 权重。

## 运行时生命周期

每个 OnlineSPEC 方法都遵守同一顺序：

1. 使用当前已发布 decision 解码，并保留真正用于采样 draft token 的 proposal
   distribution。
2. Verification 给出 target 监督与 semantic valid mask。
3. 在 update round 上，通过 side CUDA stream 计算 learner-specific candidate；candidate
   不修改 active parameter 或 active learner state。
4. 在下一次 prediction boundary 校验 cohort epoch、slot generation、source version、
   数值有限性与 ready 状态，再原子提交或完整丢弃 candidate。
5. 下一次 proposal 使用已提交 decision；exact target rejection sampling 继续使用原先
   记录的 proposal distribution。

这一边界刻意不同于 TTS 和 L0 的发布时间。OnlineSPEC 不会被静默映射到 TTS stride
barrier，也不能在产生监督的 proposal 尚未结束时发布。

历史 drafter KV 与核心运行时相同，保持 frozen 且带版本。历史是 detach 的状态输入，
只有 current canvas 可微。发布新权重不会重写旧 KV。这是 truncated-online baseline，
遥测会记录每个新 KV segment 使用的版本。

## 后端与参数支持

- DFlash 支持 drafter-scope `full`、`lora` 以及共享 tail 消融；只有 update round 会运行
  differentiable current-canvas 路径；整个同质 cohort 会作为一个 batched SDPA/MLP graph
  重建，而不是用 Python 按请求循环。
- DSpark 与 EAGLE/EAGLE3 使用 cache-safe tail 路径及其既有后端限制，不伪装提供
  drafter-wide gradient。
- OGD 与 optimistic OGD 在已注册 DFlash 网格中支持 Full 与 LoRA。
- Hedge 支持显式区分的 Full 与 LoRA decision class，并至少使用两个有序专家学习率；
  所有 LoRA expert 使用相同的注册 rank。
- TP 与 DP 都必须为一；不支持、量化或 graph 不兼容的路径会在分配 OnlineSPEC 状态
  前 fail closed。

OnlineSPEC 使用自身的 plain projected SGD 数学，不会通过配置 alias 继承 AdamW、Muon
或其他核心 optimizer。

## 显存与遥测

HBM 预检包含 active decision、初始 projection anchor、candidate state、optimistic
anchor 与 hint、全部常驻 Hedge expert 参数、累计 loss、加权 merge staging、一份可复用的
expert gradient scratch、training activation、KV gather scratch、graph buffer 与有界
telemetry。Hedge expert 逐个计算梯度：保留已更新的 expert candidate，并在计算下一个
expert 前释放当前 differentiable leaf 与 gradient。因此它保持相同的 Hedge decision，
但不需要为每个 expert 各预留一份完整 gradient。这些 tensor 不可驱逐，也不会静默
offload 或降档；KV pool 只能使用扣除这些开销后的剩余显存。
当常驻 expert 与所需 long-context KV 无法同时容纳时，Full Hedge 可以合法地在该预检
失败。LoRA Hedge 把 learner state 保存在 factor coordinate 中，是已注册的单卡替代；
它绝不会被包装成一次成功的 Full 运行。

Update telemetry 记录 learner step、source/published version、loss、gradient norm、可微路径
与推理 logits 的重建诊断、更新时间、发布时间及安全处置。Optimistic 额外记录 hint
error；Hedge 记录逐 expert gradient norm、expert probability、cumulative loss、ensemble
entropy 与 effective expert count。Headline 收集不会每轮把这些值拷回 CPU；诊断在计时
hot path 外排空有界 device buffer。

## 已注册实验

跟踪的协议是 `manifests/speed-study/onlinespec_baseline_v2.json`。它与核心实验使用相同
的 controlled greedy sampling 语义和 DFlash 模型对。Greedy confirmation 让各 learner
遵循相同 target-token 轨迹；任何计时比较前都会验证配对输出摘要，且不保留原始生成文本。
Stochastic exactness 单独验证。协议仍保持独立证据身份：

1. 只在 tuning window 执行 successive halving。OGD、optimistic OGD 与 Hedge 各自
   独立减半，避免一个 learner 淘汰另一个 learner。
2. 为每个 learner 选择一个安全 candidate。CLI 强制接收核心 Static/TTS/L0
   selection，继承其已选择并发量，并递归绑定其 SHA-256；同时绑定完整 terminal tuning
   artifact、model lock、sampling profile、manifest 与 patched SGLang tree。手工指定或
   不匹配的负载会被拒绝。
   范围较窄的复现也可以通过 `select-onlinespec-anchor-config` 为每个 learner 锁定恰好
   一个已注册 terminal candidate。该路径必须同时接收完整 terminal tuning window 上的
   配对 Static 和三个 learner slice，并把 artifact 标记为 `heldout_anchor`；它绝不声称
   exhaustive grid optimum。
3. 每个 method/block 计时窗都把 32 个不重叠 prompt 各提交一次，并组成一个有序的原生
   batch 请求。正式 admission limit 不得超过这 32 个唯一 prompt；SGLang 锁定的 admission
   limit 在不 reset cohort 的情况下排空队列。随后在已注册的 16K–40,928 安全 long
   region（40,960 checkpoint 上限之外保留两个 block-16 speculative KV reservation）
   上运行八个独立随机 block 的配对
   Static/OGD/optimistic/Hedge confirmation。Headline 值是整个 batch active decode
   区间的并集；request 级行只用于诊断。
4. 对每个 learner 与配对 Static 生成一项诊断比较，并将证据绑定到 GPU attestation。
5. Profiler 单独运行；同步 trace 不能进入 headline timing。

命令族为：

```text
build-onlinespec-study        list-onlinespec-candidates
verify-onlinespec-source
render-onlinespec-tuning-runtime
run-onlinespec-tuning-slice  advance-onlinespec-tuning-stage
select-onlinespec-config     select-onlinespec-anchor-config
render-onlinespec-runtime
build-onlinespec-queue       run-onlinespec-confirmation
collect-onlinespec-study     attest-onlinespec-study
analyze-onlinespec-study
```

准确参数以 `lightcone-spec COMMAND --help` 为准。生成的 selection、性能数据与
attestation 必须保存在 ignored artifact root。该比较是重要证据，但
`analyze-onlinespec-study` 会显式输出 `core_speed_gate_affected=false`、
`selection_protocol` 与 `optimized_grid_claim`。

## 复现声明

LightCone-Spec 声明的是：在同一工业化 speculative-decoding runtime 中，对 OnlineSPEC
在线 drafter learner 的论文公式进行 clean-room 实现。它不声明与官方脚本逐字节一致，
不重新分发其代码，也不把 Online-LR 的 reasoning DPO 流程当作 token-level draft-model
adaptation。未来若扩展该流程，必须另行定义 objective、数据、显存合同与已注册比较协议。

若要重复源码审计，应把官方仓库克隆到本项目之外，并 detached 到已记录 commit：

```bash
git clone https://github.com/ZinYY/OnlineSPEC.git OnlineSPEC-upstream
git -C OnlineSPEC-upstream checkout --detach e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b
lightcone-spec verify-onlinespec-source \
  --checkout OnlineSPEC-upstream \
  --audit manifests/provenance/onlinespec_source_audit_v2.json \
  --output /path/to/ignored-artifacts/onlinespec-source-verification.json
```

只有 checkout clean，且 commit、tree、18 个审计文件哈希和许可证清单全部匹配时，
verifier 才会通过。该 checkout 与 receipt 只作为审计输入，不得复制、vendor 或提交到
LightCone-Spec。
