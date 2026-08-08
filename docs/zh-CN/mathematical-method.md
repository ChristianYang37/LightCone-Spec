# 数学方法

[English](../en/mathematical-method.md) · [首页](../../README_zh-CN.md)

令 `h` 为冻结 proposal head `W` 前的最终 hidden，`m` 为可选的 DSpark Markov 特征。
LightCone 使用三种 cache-safe tail 参数化之一：

\[
\Delta \ell=B(A_hR_h^\top h+A_m m),
\]

\[
\Delta h=(hA_h+mA_m)B_h,\qquad \Delta\ell=W\Delta h,
\]

\[
\Delta h=hD_h+mD_m,\qquad \Delta\ell=W\Delta h.
\]

三者对应 `residual`、`lora` 和 `full`。`full` 指 full-rank tail matrix；drafter
backbone、target embedding 和 LM head 保持冻结，因此历史 KV 仍然有效。

L0 在首个合法边界发布；L1 使用二元 gate；L2 预测 `[0,1]` 内的 damping；L3 在评估
前 transport 候选。它们处理的是 staleness 下的到达时效用，而不改变 speculative
decoding 的 exactness 规则。生成和拒绝路径共享同一个 corrected proposal distribution。
