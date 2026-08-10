# 实验协议

[English](../en/experiment-protocol.md) · [首页](../../README_zh-CN.md)

## 注册问题

专项实验检验严格按论文实现的 TTS 与 first-ready L0 是否分别比未修改的 Static 获得
更高 decode goodput。实验不预设 acceptance 收益足以覆盖训练与调度成本。只有完整
协议生成内容绑定的 attestation 后，GPU 状态才会离开 `UNMEASURED`。

主模型对为 Qwen3-8B + DFlash。正式长区间从已生成 16K token 开始，到 checkpoint
安全上限结束，且不超过 40,960。TTS 与 L0 必须共享 candidate 计算、optimizer、更新
模式、参数范围、rank、学习率、stride、监督、sampling 与 load；唯一差别是发布策略。

## 数据隔离

Controlled adapter 从有限词表确定性生成无外部版权依赖的 prompt。三个内容不重叠且由
hash 绑定的窗口分别包含八个 load prompt、十六个 tuning prompt 和三十二个 confirmation
prompt。Confirmation 数据不得影响负载或超参数选择。

Context 按每次 proposal 前真实 `prefix_len_before` 记录。生成位置 bucket 为 0、2K、
4K、8K、16K、24K、32K 与安全上限。自然 EOS 副表使用锁定 revision 的 LiveCodeBench
与 Math500，各三十二个 prompt，并在每个 bucket 报告仍处于 at-risk 的请求数。它们用于
现实任务复现，不决定正式 gate。

## 负载与调参

Static 独立扫描 concurrency 1、2、4、8、16、32、48。只有 OOM 和 retraction 均为零，
KV capacity 足以容纳 `concurrency × 40,960`，且 p99 ITL 不超过 Static-c1 两倍的负载
才可选。满足条件后选择 decode goodput 最高者，并对三种方法固定。选择还要求 32-prompt
confirmation window 能提供足够的 active request；c48 screen 仍是容量诊断，不能成为只含
32 个请求的不满载正式负载。

调参对 drafter Full/LoRA 搜索 Adam、AdamW、SGDm、NAG、Muon 与 Lion，并为不同
optimizer 使用各自的 learning-rate 范围，同时覆盖 rank、stride、weight decay 与全局
gradient clipping。Muon 的 matrix step 与辅助 AdamW 字段绑定为一个 candidate identity，
run 后不能拆开选择。Successive halving 只能读取 tuning 窗口。共享配置最大化

\[
\min\left(V_{\mathrm{TTS}}/V_{\mathrm{Static}},
V_{\mathrm{L0}}/V_{\mathrm{Static}}\right),
\]

并先满足 exactness 与稳定性检查。并列时依次选择较低 peak HBM、较低 p99 ITL、较低
exposed update time。不可变 selection artifact 绑定 grid、tuning window、model lock、
patched tree、load 与 tuning evidence。

## 独立 Confirmation

Confirmation 包含八个 repetition block。每个 block 都在每种方法前独立 reset cohort，
随机排列 Static/TTS/L0，并将 32 个不同 prompt 以一次 start-gated burst 全部提交。
SGLang 锁定的 admission limit 负责 continuous batching，队列排空期间不 reset
engine/cohort。Warmup 位于计时区间之外；每个 method/block batch 只拥有 active decode
区间的并集，排除 request queue/prefill 空档；不合并 repetition，也不把 batch 指标复制到
prompt 或 bucket 行。

Request 级 ITL、TTFT、输出 identity 与单请求 decode 诊断独立记录，不能冒充系统聚合
goodput。Load screen 和早期 tuning stage 只有在 prompt 数少于 concurrency 时，才按完整
窗口 round-robin 补足负载；confirmation 与自然任务绝不通过复制 prompt 制造并发。

已注册的 controlled profile 使用 greedy。这样 Static 与每种 exact adaptation 方法都会
沿同一条 target-token 轨迹运行，配对计时效应不会来自方法相关的随机数消耗。
调参与 confirmation 仅保存每条生成轨迹的 SHA-256；任一配对方法摘要不同都会 fail
closed，证据表不保留生成文本。Stochastic
coupled-RNG 与分布检查仍是必需的 GPU 测试；stochastic 自然任务只作为鲁棒性副表，不是
因果速度 headline。

每个完整 run 都有绑定规范化 Parquet shard 的 terminal receipt。Resume 只复用身份匹配
的 receipt。中断 shard 不是证据，因此长时间实验可以续跑，不会重复或静默遗漏完整 cell。

## 指标与推断

Headline 指标来自一条覆盖 16K 至上限的专用 `long_region` batch 记录，以及完整轨迹的
配对 batch decode-goodput effect；位置 bucket 只用于解释，不会等权平均进 headline。TTS 与
L0 分别使用 repetition-block BCa 95% 区间对比 Static。Repetition block 才是独立计时
和随机化单元；把共享同一 wall-clock interval 的 32 个请求当作 32 个独立 goodput 样本
会构成伪重复。二者都必须达到注册的平均提升门槛，且置信区间下界大于零。

解释性指标保留 survival-weighted accepted prefix、每次 verification 的 committed 与
verified draft、verification waste、target calls/output token、TTFT、ITL 分位数、各 CUDA
lane 时间、exposed update time、overlap、graph replay、batch fill、queue occupancy、
peak HBM、KV/optimizer 显存、loss 与 trainable parameter count。Target-only estimated
MFU 与 profiler 实测设备利用率使用不同名称。

Optimizer 显存按类别记录，不从 parameter count 反推，从而区分双 moment 的
Adam/AdamW、单 moment 的 SGDm/NAG/Lion，以及 Muon 的 matrix momentum 与非 matrix
辅助 AdamW state。

Adapted run 还为每个 verification round 保留语义记录，包括真实 `prefix_len_before`、
有效的 verified/accepted/committed 计数、proposal source version 与 frozen-KV version
segments。Static 只启用聚合实验计数，绝不分配 adaptation trace buffer。

详细 profiling 使用独立 run；headline decoding 不逐轮 synchronize。Acceptance 单独
改善不能表述为加速。

## OnlineSPEC 对比协议

OnlineSPEC 作为重要对比被注册在独立 manifest 与证据 namespace 下。它使用 controlled
tuning/confirmation window，但不能读取核心 TTS/L0 tuning row 或 confirmation 结果。
三种 learner 在 successive halving 中各自独立缩减，随后每种 learner 选择一个安全配置，
并与配对 Static reference 比较。

Confirmation 在每种方法中将 32 个 held-out prompt 各提交一次并进入同一个 SGLang queue，
同时使用八个随机 block、相同 seed、一个锁定 concurrency，以及相同的 16K 到安全上限
区域。聚合 goodput 只在八个独立 method/block active-interval 并集上推断，prompt 级行仅用于诊断。派生表分别报告 OGD、
optimistic OGD 与 Hedge，绝不把它们折叠成 best-of-baselines 结果。每项比较都包含配对 BCa 区间、安全
计数、update 数、HBM 分类，以及 [OnlineSPEC baseline](onlinespec-baseline.md)定义的
learner-specific 诊断。

该对比使用自己的内容绑定 GPU attestation。缺少 attestation 时结果为 `UNMEASURED`；
即使具备 attestation，它仍然是诊断结果，不能改变核心正式 gate 或其选择配置。

## 正式门槛

两种 adaptation 方法都必须分别比 Static 获得至少百分之三的平均 goodput 提升，配对
BCa 95% 下界大于零。Exactness violation、version mismatch、fallback、non-finite update、
OOM 与 retraction 必须全部为零；adapted run 必须实际 launch 并 publish update。

只有 attestation 同时绑定 manifest、selection、精确 evidence bytes、模型 revision、
patched SGLang tree 与 GPU 硬件报告时，gate 才可能返回 `PASS`。未 attested 的计算始终
是 `UNMEASURED`；实测但未满足任一条件时是 `BLOCKED`。本仓库只包含协议代码，不包含
结果 artifact 或性能宣传。
