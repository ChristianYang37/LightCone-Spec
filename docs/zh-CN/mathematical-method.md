# 数学方法

[English](../en/mathematical-method.md) · [README](../../README_zh-CN.md)

## 优化目标

令 \(p_t\) 为 target distribution，\(q_\theta\) 为某个合法 draft 位置的 DFlash
proposal。更新最小化带位置权重的 target-to-draft KL：

\[
\mathcal L(\theta)=
\frac{\sum_{b,k}m_{b,k}\lambda^k
D_{\mathrm{KL}}(p_{b,k}\Vert q_{\theta,b,k})}
{\sum_{b,k}m_{b,k}\lambda^k}.
\]

Semantic mask \(m\) 排除越过请求上限的位置和 terminal token 之后的位置，但刻意保留
首次 rejection 后由 drafter 采样的反事实 suffix：target verification 已经为完整 canvas
给出了有效 teacher distribution，这与 TTS 目标一致。Gradient 在 cohort 内归一化，
因此 batch size 不会暗中改变 learning rate。对于已注册的 block-16 DFlash checkpoint，
位置权重绑定其训练 recipe：

\[
w_k=\exp\!\left(-\frac{k-1}{7}\right),
\qquad \lambda=\exp(-1/7).
\]

DFlash 不使用均匀位置权重，因为靠前 token 的错误会令其后的全部 token 无法通过
prefix verification。因此该 decay 属于 run identity，而不是静默默认值或由结果反推的
超参数。参见 [DFlash 论文](https://arxiv.org/abs/2602.06036)。

## 相同 candidate，不同发布时刻

对于 source round \(r\)，两种 adaptation 方法计算同一个 functional optimizer
proposal \(u_r\)。发布之前，active parameter 与 moment 不会变化。

设 update stride 为 \(S\)，TTS 不早于下一个固定更新边界发布：

\[
a_{\mathrm{TTS}}=
\max\!\left(a_{\mathrm{ready}},
(\lfloor r/S\rfloor+1)S\right).
\]

L0 在 side event ready 后的首个合法 decode 边界发布：

\[
a_{\mathrm{L0}}=a_{\mathrm{ready}}.
\]

Candidate tensor、loss、optimizer、rank、stride、supervision 与 source version 全部
保持一致。任何数值 candidate 差异都是实现错误，而不是方法差异。

## 截断在线梯度

Update round 将历史 paged KV gather 后 detach。当前 canvas K/V 通过可微 DFlash
linear、RMSNorm、NeoX RoPE 与 non-causal SDPA 重算。Device-side reconstruction
predicate 将重算 hidden 与真实 inference hidden 比较；不匹配时 proposal 变为 no-op，
并关闭该 cohort 后续 adaptation。

Target embedding 与 LM head 冻结。最终 target head 将可微 DFlash hidden 映射为 draft
logits。Source-point proximal KL 的一阶导数为零，因此实现不物化无效项，也不把其系数
作为伪 tuning 维度。

## 参数化

对已选择 base matrix \(W\)，LoRA 使用

\[
W'=W+BA,
\qquad A\in\mathbb R^{r\times d_{in}},\quad
B\in\mathbb R^{d_{out}\times r},
\]

其中 \(A\) 按锁定 seed 初始化，\(B\) 为零，两者均可训练。Inference 只看到合并后的
固定地址 matrix。Full 为每个 DFlash 自有浮点参数保留 FP32 master。Tail LoRA/full
计算 \(h'=h+\Delta h\)，再通过冻结 target head 投影一次；residual tail 直接对 logits
施加低秩修正。

对于 DSpark，\(h\) 特指 normalized LM-head input，Markov state 保持为另一份冻结
feature。对于 EAGLE/EAGLE3，同一 proposal chain 中的所有 hidden 都使用钉在同一 source
version 的 tail bank。

## Optimizer 动力学

每个 optimizer 都实现为 functional proposal
\((\theta_t,s_t,g_t)\mapsto(\hat\theta_{t+1},\hat s_{t+1})\)。只有发布提交
candidate 后，active parameter/state 才会改变。因此，即使 TTS 与 L0 的 candidate 在
不同 decode boundary 才能发布，它们仍执行完全相同的 optimizer 算术。

Adam 与 AdamW 使用 bias-corrected FP32 一阶、二阶动量；AdamW 使用 decoupled decay。
对 coupled-decay momentum 方法，令 \(\tilde g_t=g_t+\lambda\theta_t\)，SGDm 与 NAG 为

\[
v_t=\mu v_{t-1}+\tilde g_t,\qquad
d_t^{\mathrm{SGDm}}=v_t,\qquad
d_t^{\mathrm{NAG}}=\tilde g_t+\mu v_t.
\]

Lion 只保留一个 moment：

\[
d_t=\operatorname{sign}(\beta_1m_{t-1}+(1-\beta_1)g_t),\qquad
m_t=\beta_2m_{t-1}+(1-\beta_2)g_t,
\]

并使用 decoupled weight decay。Muon 对每个二维参数构造 Nesterov momentum proposal，
执行已注册的 quintic Newton--Schulz orthogonalization，并按
\(\sqrt{\max(1,d_{out}/d_{in})}\) 缩放 matrix step。Bias、norm 与其他非 matrix 参数
执行显式配置的辅助 AdamW。任何 optimizer 都不会静默分配无用 moment，也不会退回
另一种更新规则。

## OnlineSPEC 对比 learner

独立 OnlineSPEC baseline 使用 projected OGD，而不是核心 optimizer。给定已揭示 gradient
$g_t$，OGD 为

\[
w_{t+1}=\Pi_K(w_t-\eta g_t).
\]

Optimistic OGD 保存 anchor \(\hat w_t\)，并将最近一次揭示的 gradient 作为下一轮 hint。
Hedge 保存相互独立的 OGD 专家，在每个专家自己的 decision 上计算其 loss 与 gradient，
再用负累计 loss 的指数权重形成下一轮 full-parameter decision。实现是 transactional 的，
只在得到反馈后提交，因此收到 gradient 的 decision 正是生成 proposal 的 decision。

完整公式、projection 语义与 clean-room 源码边界见
[OnlineSPEC baseline](onlinespec-baseline.md)。这些是对比方法的数学，不改变
Static/TTS/L0 objective 或速度 gate。

## Exact speculative sampling

Verification 记录的 proposal probability \(q\) 必须正是生成 draft token 的分布。给定
target probability \(p\)，proposal token \(x\) 的接受概率为

\[
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right).
\]

拒绝时从归一化的 \((p-q)_+\) 采样 replacement。即使历史 KV 由旧 drafter version
产生，也保持 target distribution。正式 controlled speed profile 使用 greedy，确保
所有方法遵循相同 token 轨迹；stochastic coupled-RNG 与分布 exactness 单独检查。

## 加速条件

实测 decode time 分解为

\[
T_m=T_{\mathrm{static}}-\Delta T_{\mathrm{target}}
+T_{\mathrm{update}}^{\mathrm{exposed}}
+T_{\mathrm{draft}}^{\mathrm{extra}}+T_{\mathrm{barrier}}.
\]

Acceptance 单独提升不充分。只有配对 long-region decode goodput 提升、target calls 不与
该提升矛盾，且 exactness、version、fallback、non-finite、OOM 与 retraction 计数均为
零时，方法才通过工程门槛。
