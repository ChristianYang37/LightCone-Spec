# 数学方法

[English](../en/mathematical-method.md) · [README](../../README_zh-CN.md)

## 目标函数与科学身份

下列公式定义已注册目标 method，不表示本 release 可执行每一种 backend 或 topology。当前
唯一可用于 claim 的 execution 仍是 TP1/DP1 Target-only。固定 patch 已包含 native
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1` lifecycle，以及注册的
DFlash、DSpark、NEXTN、compatible EAGLE3、TP2 与 sticky DP2 source implementation。但这些
都不是 GPU 结果：准确 dynamic suite 与 trusted external-control proof 尚未生成，也没有配置
trusted hardware signer。因此 Static/TTS/L0-naive/LightCone 会在 mutation 前 blocked，并且
目前没有任何新 GPU 结果。

Recipe authority 与 publication policy 是不同的数学身份。Target-only 与 Static 在结构上
保持零 adaptation state。TTS 使用 frozen TTS recipe 与 fixed barrier；L0-naive 使用同一个
frozen recipe authority 与 first-ready publication；LC-candidate 使用已注册 E1/E2 recipe 与
相同 first-ready policy。只有准确 E2-sealed winner 才是 LightCone。

令 target distribution 为 $p$，重建 proposal 为 $q_\theta$，semantic mask 为 $m$，
可选 position weight 为 $w_k$。已注册 LightCone-candidate proposal objective 是
target-to-draft cross entropy（与 KL 只相差 target entropy）：

$$
\mathcal L_{\mathrm{proposal}}(\theta)=
\frac{\sum_{b,k}m_{b,k}w_k
[-\sum_v p_{b,k,v}\log q_{\theta,b,k,v}]}
{\sum_{b,k}m_{b,k}w_k}.
$$

Mask 排除超出 request budget 及 terminal state 之后的 row，但保留 verifier 为完整合法
canvas 生成的 teacher row，包括更早 rejection 后的 sampled counterfactual suffix。Cohort
归一化阻止 batch size 静默缩放 learning rate。空或 non-finite denominator 会产生无效
device candidate 并被丢弃，绝不会转成 zero loss。

一个已注册 block-16 **LightCone-candidate** recipe 使用

$$
w_k=\exp(-(k-1)/7).
$$

该 weight 不是 TTS default。Position weight、teacher-row policy、canvas width 与 source
version 都属于 candidate 的 config/evidence 身份，不是从结果反推的选择。

一手 TTS 论文定义从 source proposal $q_t$ 出发、只使用 latest-round row 的一次更新；
objective 包含 target distillation 与 source-point proximal term：

$$
\mathcal L_{\mathrm{TTS}}(q')=
\sum_k w_k\,\operatorname{KL}(p_k\|q'_k)
+\lambda\sum_k w_k\,\operatorname{KL}(q_{t,k}\|q'_k).
$$

经审计的 [arXiv v2 论文](https://arxiv.org/abs/2605.09329v2) 没有披露全部数值字段，因此历史
`TTS-paper-reconstruction` authority 继续是 diagnostic 且不具备 formal eligibility。Formal
TTS-Cal authority 是独立预注册 reconstruction：Adam、准确一步、
`(beta1=.9, beta2=.999, epsilon=1e-8)`、zero decay、无 clipping、全 drafter/
latest-round-only update、digest-bound drafter-native position/proximal loss recipe、逐 request
reset、side stream、learning rate `1e-7, 3e-7, ..., 1e-3` 与 stride
`{1,5,10,15,20,30,40,50}`。四个 excluded pilot 按注册的 safety-first reducer 选出 winner，并
由 signed seal 冻结；在该 seal 存在前 TTS 与 L0-naive 保持 `BLOCKED`。二者不能继承上面的
exponential weight、E1/E2 winner、schema default 或历史 AdamW recipe。与 recipe 正交，TTS
publication 使用论文的 fixed synchronization barrier。

## Backend 证据与重建

令 $E_b$ 为 backend-native `ProposalEvidence` envelope。公共字段包括 adapter-free logits
$z^0$、deployed proposal logits、normalized sampling distribution
$q^{\mathrm{sample}}$、mask/teacher row、真实 sampled predecessor、cohort/source 身份与
backend payload。注册 backend reconstruction

$$
R_b(E_b,\Delta_\theta)
  \longrightarrow (z_\theta,q_\theta,c_\theta?)
$$

必须保持 row/vocabulary shape 与 confidence availability。Zero functional delta 会与 native
inference 核对。如果 inference 已施加 adapter，再传入非零 delta 会被拒绝，避免更新重复
计入。

结构验证在昂贵工作前由 host 执行。Finiteness、normalization 与 reconstruction/
supervision validity 生成三 byte device receipt；side stream 会在 completion event 前把它
异步复制到 pinned host memory。合法 publication boundary 先执行 nonblocking event query；
仅在完成后、measured update interval 之外读取 receipt。该路径不使用 `.item()` 或 blocking
event synchronization。Invalid receipt 不推进 optimizer step 或 active version，也不发布
任何 tensor。

固定 patch 重建 DFlash differentiable canvas、DSpark native Markov W1/W2 feature 与实际
sampled predecessor、NEXTN MTP hidden/interface state，以及受官方 selector 门控的 EAGLE3
路径。这些 payload 不能互换。DSpark、NEXTN 与 compatible EAGLE3 保持
`implemented_pending_dynamic_gpu_proof`；generic EAGLE 不受支持。

## DSpark Composite Objective

DSpark contract 的 hybrid plan 还可训练 native confidence/acceptance state，但必须通过
准确的 dynamic qualification。对 verified target row $p$
与 sampling-bound proposal $q$，定义 stop-gradient conditional-survival target

$$
s=1-\operatorname{TV}(p,q)
=1-\frac12\sum_v|p_v-q_v|,\qquad s\in[0,1].
$$

令 confidence logit 为 $a_\theta$，tuning-locked 非负系数为 $\gamma$，masked hybrid
objective 为

$$
\mathcal L_{\mathrm{DSpark}}
=\mathcal L_{\mathrm{proposal}}
+\gamma\,\operatorname{BCEWithLogits}(a_\theta,s).
$$

Survival target 被 detach，因此 confidence term 不能通过对自身 label 求导来移动 proposal。
Layer-only DSpark 冻结 W1、W2 与 confidence，并拒绝 $\gamma$。Hybrid DSpark 把 W1、W2
与 scalar acceptance/confidence parameter 作为 Full replicated state 训练，所选 backbone
layer matrix 则使用选定 Full 或 LoRA plan。

Fixed-budget verification 只用于 tuning control；native-scheduler confirmation 与其分开。
缺失 supervision row 不能通过看到结果后修改 mask 来补救。

## Parameterization 与显存

对所选 native matrix $W$，LoRA 使用

$$
W'=W+BA,\qquad
A\in\mathbb R^{r\times d_{in}},\quad
B\in\mathbb R^{d_{out}\times r}.
$$

初始化确定且 functional delta 为零。注册 rank 为 (1,2,4,8,16,32,64)，并固定
$\alpha/r=1$。Full 为所选 `last1`、`last3`、`last5` 或 `all` native layer scope 中每个
合格浮点参数保留 FP32 master。借用的 target embedding、target head 与 target model 冻结。

DSpark 另允许 last-1/3/5 hybrid scope 加 native head。完整 E1a geometry 为
$4\times8+3\times8=56$ 个配置。Full/LoRA、scope、rank、alpha、selected/frozen name、
dtype 与 sharded/replicated ownership 全部绑定 digest。

显存按实际 trainable coordinate 推导：active merged value、FP32 master/gradient、真实
optimizer moment、functional candidate、staging、activation/scratch、graph、KV 与 telemetry。
Admission 使用所有 rank 中最小 headroom；任何方法都不能为适应显存而被静默改变。

## Functional Optimizer 与发布

每个 optimizer 实现 functional proposal

$$
(\theta_t,s_t,g_t)\mapsto(\widehat\theta_{t+1},\widehat s_{t+1}).
$$

只有完整 candidate commit 时，active parameter 与 optimizer state 才改变。Adam/AdamW
使用两个 FP32 moment；SGDm/NAG/Lion 使用一个。Muon 使用 matrix momentum 与注册
Newton--Schulz transform，并对 non-matrix 使用辅助 AdamW moment。Schedule 按 published
update 而非 attempted/discarded candidate 前进。

对 source round $r$，TTS publication policy 会在 ready 后等待固定注册 barrier；L0 policy
则在 ready 后首个合法 safe graph boundary 发布。TTS 与 L0-naive 共享 frozen recipe
**authority**，不共享 live optimizer/candidate/evidence identity。LightCone 也使用 L0 policy，
但其 recipe 必须是准确 E2-sealed winner；任何未封存 search recipe 仍只是 LC-candidate。

Candidate equality 只在受控 mechanism replay 中检查。只有两个 replay binding 的
source-state 与 proposal-evidence digest 都完全相同，才比较 candidate byte。Live TTS 与
L0-naive 的 publication decision 分叉后，其未来 candidate 无需继续相等。

在 distributed coordinator contract 中，TP2 sharded coordinate 留在 inference owner，
replicated coordinate 在 TP replica 内部 reduce；DP2 cohort sticky 且 replica-local。所有
rank 将对一个 update identity prepare，并导出一个 commit/abort decision。`RunConfig` 只在
准确 source capability identity 存在时接受这些 topology；formal dispatch 在缺少全新
all-rank GPU qualification 时仍会拒绝。因此公式本身不构成 multi-rank 结果。

## OnlineSPEC 对比 Learner

独立注册的 OnlineSPEC baseline 使用 projected SGD decision。Projected OGD 为

$$
w_{t+1}=\Pi_K(w_t-\eta g_t).
$$

Optimistic OGD 保存 anchor，并把最近一次 revealed gradient 作为 hint。Hedge 保存独立
expert，在各自 decision 上计算 loss/gradient，并按负累计 loss 的指数权重形成未来 decision。
Full 与 LoRA factor-coordinate decision 是不同 class；factor averaging 不等价于对 dense
product $BA$ 求平均。

OnlineSPEC 是 transactional 的，并在独立协议下保持 TP1/DP1。其公式与证据不能改变
frozen TTS authority、E1/E2 LightCone selection、power plan 或核心 gate。

## Exact Speculative Sampling

记录的 proposal probability $q$ 必须正是生成 draft token 的 distribution。给定 target
probability $p$，token $x$ 的接受概率为

$$
\alpha(x)=\min(1,p(x)/q(x)).
$$

拒绝时，从归一化 $(p-q)_+$ 采样 replacement。这是 exact speculative sampling 必需的
positive-part rejection residual，不是 adaptation parameterization、trainable module、
schema value 或历史 alias。删除旧 adaptation option 不会改变 sampler exactness rule，二者
必须明确区分。

Greedy controlled trace 保持配对 token trajectory 相同。Stochastic coupled-RNG 与
distributional test 仍是独立 GPU 要求。

## 速度与结论条件

实测 decode time 分解为

$$
T_m=T_{\mathrm{static}}-\Delta T_{\mathrm{target}}
+T_{\mathrm{update}}^{\mathrm{exposed}}
+T_{\mathrm{draft}}^{\mathrm{extra}}+T_{\mathrm{barrier}}.
$$

Acceptance 不是速度结果。结论要求注册 paired goodput inference、target-work consistency、
production SLO/completion accounting、exactness/version/publication safety、HBM/energy
evidence、hardware-envelope validity、durable receipt 与 content-bound GPU attestation。
缺少这些证据时，全部新结果保持 `UNMEASURED`。
