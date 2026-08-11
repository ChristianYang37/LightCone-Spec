# 数学方法

[English](../en/mathematical-method.md) · [README](../../README_zh-CN.md)

## 目标与 Semantic Row

下列公式定义已注册目标 method，不表示本 release 可执行每一种 backend 或 topology。
Industrial executor 当前只完成 TP1/DP1 Target-only；Static/TTS/L0 需要尚未实现的
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1` hook。固定 patch 只为
TP1/DP1 DFlash 包含底层 adaptive implementation，并且目前没有任何新 GPU 结果。

令 target distribution 为 (p)，重建 proposal 为 (q_\theta)，semantic mask 为 (m)，
可选 position weight 为 (w_k)。公共 proposal objective 是 target-to-draft cross entropy
（与 KL 只相差 target entropy）：

\[
\mathcal L_{\mathrm{proposal}}(\theta)=
\frac{\sum_{b,k}m_{b,k}w_k
[-\sum_v p_{b,k,v}\log q_{\theta,b,k,v}]}
{\sum_{b,k}m_{b,k}w_k}.
\]

Mask 排除超出 request budget 及 terminal state 之后的 row，但保留 verifier 为完整合法
canvas 生成的 teacher row，包括更早 rejection 后的 sampled counterfactual suffix。Cohort
归一化阻止 batch size 静默缩放 learning rate。空或 non-finite denominator 会产生无效
device candidate 并被丢弃，绝不会转成 zero loss。

已注册 block-16 DFlash recipe 使用

\[
w_k=\exp(-(k-1)/7).
\]

Position weight、teacher-row policy、canvas width 与 source version 都属于 config/evidence
身份，不是从结果反推的选择。

## Backend 证据与重建

令 (E_b) 为 backend-native `ProposalEvidence` envelope。公共字段包括 adapter-free logits
(z^0)、deployed proposal logits、normalized sampling distribution
(q^{\mathrm{sample}})、mask/teacher row、真实 sampled predecessor、cohort/source 身份与
backend payload。注册 backend reconstruction

\[
R_b(E_b,\Delta_\theta)
  \longrightarrow (z_\theta,q_\theta,c_\theta?)
\]

必须保持 row/vocabulary shape 与 confidence availability。Zero functional delta 会与 native
inference 核对。如果 inference 已施加 adapter，再传入非零 delta 会被拒绝，避免更新重复
计入。

结构验证在昂贵工作前由 host 执行。Finiteness、normalization 与 reconstruction/
supervision validity 生成三 byte device receipt；side stream 会在 completion event 前把它
异步复制到 pinned host memory。合法 publication boundary 先执行 nonblocking event query；
仅在完成后、measured update interval 之外读取 receipt。该路径不使用 `.item()` 或 blocking
event synchronization。Invalid receipt 不推进 optimizer step 或 active version，也不发布
任何 tensor。

当前底层 patch 为 TP1/DP1 重建 DFlash differentiable canvas。目标 DSpark contract 要求
native Markov W1/W2 feature 与实际 sampled predecessor；EAGLE/EAGLE3 将绑定 tree state 与
贯穿 proposal chain 的一个 source version；NEXTN 将绑定 native MTP hidden state 与
upstream interface 内容 digest。这些 payload 不能互换，并且 DSpark/EAGLE/EAGLE3/NEXTN
adaptation 保持 `BLOCKED`。

## DSpark Composite Objective

不可执行的目标 DSpark contract 中，hybrid plan 还可训练 native confidence/acceptance
state。对 verified target row (p)
与 sampling-bound proposal (q)，定义 stop-gradient conditional-survival target

\[
s=1-\operatorname{TV}(p,q)
=1-\frac12\sum_v|p_v-q_v|,\qquad s\in[0,1].
\]

令 confidence logit 为 (a_\theta)，tuning-locked 非负系数为 \(\gamma\)，masked hybrid
objective 为

\[
\mathcal L_{\mathrm{DSpark}}
=\mathcal L_{\mathrm{proposal}}
+\gamma\,\operatorname{BCEWithLogits}(a_\theta,s).
\]

Survival target 被 detach，因此 confidence term 不能通过对自身 label 求导来移动 proposal。
Layer-only DSpark 冻结 W1、W2 与 confidence，并拒绝 \(\gamma\)。Hybrid DSpark 把 W1、W2
与 scalar acceptance/confidence parameter 作为 Full replicated state 训练，所选 backbone
layer matrix 则使用选定 Full 或 LoRA plan。

Fixed-budget verification 只用于 tuning control；native-scheduler confirmation 与其分开。
缺失 supervision row 不能通过看到结果后修改 mask 来补救。

## Parameterization 与显存

对所选 native matrix (W)，LoRA 使用

\[
W'=W+BA,\qquad
A\in\mathbb R^{r\times d_{in}},\quad
B\in\mathbb R^{d_{out}\times r}.
\]

初始化确定且 functional delta 为零。注册 rank 为 (1,2,4,8,16,32,64)，并固定
\(\alpha/r=1\)。Full 为所选 `last1`、`last3`、`last5` 或 `all` native layer scope 中每个
合格浮点参数保留 FP32 master。借用的 target embedding、target head 与 target model 冻结。

DSpark 另允许 last-1/3/5 hybrid scope 加 native head。完整 E1a geometry 为
(4\times8+3\times8=56) 个配置。Full/LoRA、scope、rank、alpha、selected/frozen name、
dtype 与 sharded/replicated ownership 全部绑定 digest。

显存按实际 trainable coordinate 推导：active merged value、FP32 master/gradient、真实
optimizer moment、functional candidate、staging、activation/scratch、graph、KV 与 telemetry。
Admission 使用所有 rank 中最小 headroom；任何方法都不能为适应显存而被静默改变。

## Functional Optimizer 与发布

每个 optimizer 实现 functional proposal

\[
(\theta_t,s_t,g_t)\mapsto(\widehat\theta_{t+1},\widehat s_{t+1}).
\]

只有完整 candidate commit 时，active parameter 与 optimizer state 才改变。Adam/AdamW
使用两个 FP32 moment；SGDm/NAG/Lion 使用一个。Muon 使用 matrix momentum 与注册
Newton--Schulz transform，并对 non-matrix 使用辅助 AdamW moment。Schedule 按 published
update 而非 attempted/discarded candidate 前进。

对 source round (r)，TTS 与 L0 计算同一 candidate (u_r)。设 stride 为 (S)，TTS 在
ready 后等待下一个注册 update boundary，L0 则使用 ready 后首个合法 graph boundary。
Loss、trainable plan、optimizer arithmetic、source row 与 candidate byte 必须一致；数值
不同属于实现失败。

在目标 CPU coordinator contract 中，TP2 sharded coordinate 留在 inference owner，
replicated coordinate 在 TP replica 内部 reduce；DP2 cohort sticky 且 replica-local。所有
rank 将对一个 update identity prepare，并导出一个 commit/abort decision。当前 `RunConfig`
会在 model loading 前拒绝全部 TP2/DP2 plan，因此这些公式不是 multi-rank execution claim。

## OnlineSPEC 对比 Learner

独立注册的 OnlineSPEC baseline 使用 projected SGD decision。Projected OGD 为

\[
w_{t+1}=\Pi_K(w_t-\eta g_t).
\]

Optimistic OGD 保存 anchor，并把最近一次 revealed gradient 作为 hint。Hedge 保存独立
expert，在各自 decision 上计算 loss/gradient，并按负累计 loss 的指数权重形成未来 decision。
Full 与 LoRA factor-coordinate decision 是不同 class；factor averaging 不等价于对 dense
product (BA) 求平均。

OnlineSPEC 是 transactional 的，并在独立协议下保持 TP1/DP1。其公式与证据不能改变
TTS/L0 candidate、selection、power plan 或核心 gate。

## Exact Speculative Sampling

记录的 proposal probability (q) 必须正是生成 draft token 的 distribution。给定 target
probability (p)，token (x) 的接受概率为

\[
\alpha(x)=\min(1,p(x)/q(x)).
\]

拒绝时，从归一化 \((p-q)_+\) 采样 replacement。这是 exact speculative sampling 必需的
positive-part rejection residual，不是 adaptation parameterization、trainable module、
schema value 或历史 alias。删除旧 adaptation option 不会改变 sampler exactness rule，二者
必须明确区分。

Greedy controlled trace 保持配对 token trajectory 相同。Stochastic coupled-RNG 与
distributional test 仍是独立 GPU 要求。

## 速度与结论条件

实测 decode time 分解为

\[
T_m=T_{\mathrm{static}}-\Delta T_{\mathrm{target}}
+T_{\mathrm{update}}^{\mathrm{exposed}}
+T_{\mathrm{draft}}^{\mathrm{extra}}+T_{\mathrm{barrier}}.
\]

Acceptance 不是速度结果。结论要求注册 paired goodput inference、target-work consistency、
production SLO/completion accounting、exactness/version/publication safety、HBM/energy
evidence、hardware-envelope validity、durable receipt 与 content-bound GPU attestation。
缺少这些证据时，全部新结果保持 `UNMEASURED`。
