# SGLang Patch 工作流

[English](../en/sglang-patches.md) · [首页](../../README_zh-CN.md)

## 源码边界

SGLang 是外部 Apache-2.0 项目。LightCone-Spec 固定 upstream commit
`3312645a307453893a00778592f105581e3d1c3d`，只分发 semantic mail patch。仓库不包含
SGLang source、submodule 或 modified checkout。本地 `sglang/` 目录被忽略，永远不是
integration identity。

只支持完整有序 series。Patch 数量、filename、SHA-256、modified-file inventory、upstream
identity 与 expected final tree 以 `patches/sglang/series` 和
`patches/sglang/manifest.json` 为准；文档不会复制为另一份 authority。

## Schema-v3 目标与已验证 Patch Surface

Schema-v3 envelope 定义一个 coherent **目标** runtime surface。下列 item 是 contract 与
registry vocabulary，不表示当前 patch 已实现每一项：

1. 严格 Target-only/Static/TTS/L0 与 backend-native 配置，disabled path 不分配 adaptation
   state；
2. 公共 proposal evidence、backend payload validation、differentiable reconstruction 与准确
   sampling-distribution preservation；
3. Full/LoRA native layer plan、DSpark W1/W2/acceptance hybrid、functional optimizer
   candidate、source/buffer/optimizer generation 与 fixed-address publication；
4. DFlash、DSpark、EAGLE/EAGLE3、NEXTN native proposal-hook contract，且不重复施加 adapter；
5. 单节点 TP2 与 sticky DP2 identity/ownership、all-rank prepare/decision/receipt publication
   与 fail-closed process-group handling；
6. least-rank HBM admission、固定 cohort slab、有界 device/event telemetry、durable Parquet
   WAL evidence、lifecycle cleanup 与聚焦 regression；
7. OnlineSPEC comparison hook 折叠进相同 version/exactness/evidence infrastructure，同时与
   gate 隔离。

中间 patch state 是 review boundary，不是产品 variant。依赖早期 patch 的新增项必须留在完整
series；verifier 不会跳过 patch 来凑出可应用的 partial combination。

当前固定 patch 已实现并测试严格 schema-v3 parsing、allocation-free Target-only/Static，以及
TP1/DP1 DFlash native-layer Full/LoRA adaptation 与 fixed-address、device-predicated
publication。这是底层 patched-server surface，不是端到端 industrial executor support。
它也实现了纯 CPU DSpark native contract：adapter-free backbone reconstruction 只施加一次
candidate delta，actual sampled predecessor 选择真实 Markov W1 embedding；layer-only scope
冻结 W1/W2/acceptance state，hybrid scope 只以 Full mode 训练这些 head；composite objective
使用准确 proposal/teacher distribution 与 stop-gradient `1-TV` confidence target。fixed-budget
decision 是准确 total-token budget，native-scheduler decision 仍保持 authoritative。该 contract
没有连接 DSpark worker/CUDA publication path，runtime capability gate 仍关闭。
它也让官方 SGLang serving benchmark 对 cumulative/incremental streaming 与 non-streaming
response 暴露 server 原样返回的有序 `output_ids`。这些 ID 不会通过重新 tokenize 生成文本来
重建；缺失、不连续或改写的 trajectory 会无法通过 claim-grade exactness gate。Benchmark
client 现在还接受由 caller 拥有的 async HTTP session，submit 与 abort 共用它，因此一个
server session 可以复用已注册 connection pool，而无需复制 official request parser。

Patch 已实现 content-bound
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1` capability 与
begin/reset/finalize endpoint。当前准确 identity 为 patch SHA-256
`907c8ecc8fdf970a585c359d902777cca3bd08b11ae2342a68cf33016f5272f4` 和 final tree
`c6070bbf97711a01dc9ab01a0e9b3ee3c2d48cb4`；manifest 仍是 authority。Lifecycle 绑定
run/nonce/plan/rank、process/session/reset lineage、expected request ID、准确 ordered token ID、
terminal coverage、Static aggregate safety，以及 TTS/L0 request/round/update/KV/performance
row。它暴露 signer plugin boundary，但不捆绑 trusted hardware key 或 release signer。因此
executor 仍只端到端运行 Target-only，并在 mutation 前阻止 Static/TTS/L0。下列 execution
contract 也尚未实现，因此会在 model loading 前拒绝：

- DSpark、EAGLE、EAGLE3 与 NEXTN adaptation；
- 全部 TP2 或 DP2 run；
- update round 缺少 native supervision 时的 quota-shadow teacher acquisition（固定 ledger
  会记录该需求并由 capability gate 阻止）；
- DSpark worker/CUDA composite-head training 与 NEXTN training interface。

TP1/DP1 DFlash TTS/L0 现在会执行 constant、按 published update 的 inverse-square-root
以及有限 horizon cosine schedule，也会记录 intrinsic readiness 并应用非负 logical delay，
同时保持 TTS fixed-boundary 与 L0 first-ready 的 publication 区别。这些 CPU/native contract
不构成 CUDA 性能证据。

这些 row 保持 `BLOCKED`，不会伪装成 DFlash，也不会作为可运行 `UNMEASURED` work 上报。
底层 DFlash implementation 与有效 terminal envelope 没有绑定固定 tree/challenge 的
allowlist out-of-band signer 时，仍不能升级为可声明 evidence。本 release 不报告 CUDA
graph、multi-GPU、速度、容量或任何其他 GPU 结果。

## 应用

`patches/sglang/apply.sh` 只接受位于准确 upstream HEAD 的 clean checkout。它检查注册 patch
digest，按顺序使用 `git am` 应用，并把结果 Git tree 与 manifest 比较：

```bash
patches/sglang/apply.sh /path/to/clean-sglang
```

Dirty state、错误 commit、被修改 patch byte、mail-application failure 或 final-tree mismatch
都会立即停止。脚本不会 stash、reset、rebase 或编辑输入 checkout 来隐藏 mismatch。

## 验证与编写

使用另一个 clean upstream checkout：

```bash
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream
```

Verifier 必须在 disposable clone 应用完整 series，确认 expected tree 与 modified-file
inventory，编译 changed Python，运行指定 focused test，reverse series，并证明调用者的
upstream checkout 仍然 clean。聚焦 adaptation-protocol test 通过 stub package 只加载已
patch 的 config、runtime、parameter-plan 与 DSpark CPU-contract module；因此 patch-integrity CI 不依赖 SGLang
顶层初始化所需的无关 optional serving package。

新的 integration work 必须 patch-first：从 pin 临时建 branch，完成带测试的 focused semantic
commit，通过 `git format-patch` 导出，并原子更新 series order、patch digest、modified file、
expected final tree、Python pin constant、NOTICE 与 EN/zh 文档。不得编辑或提交 patched
checkout。

## 当前 Evidence Gate

最终 schema-v3 patch 已通过仓库完整 apply/compile/focused-test/reverse verifier。这是
patch-integrity 结果，不是 GPU validation。CPU package test 或旧 patched-tree receipt
不足以证明 GPU 能力。TP1/DP1 DFlash 与严格 native terminal lifecycle 已实现，但 release
trust policy 未配置 signer；因此 Static/TTS/L0 industrial cell 会在任何 mutation 前
`BLOCKED`，而不是可运行 `UNMEASURED` work。Test signer、provider object attribute 或
caller-supplied verifier 都不能解锁该 policy。DSpark CPU contract 不授权 execution。
DSpark/EAGLE/EAGLE3/NEXTN adaptive cell 与
全部 TP2/DP2 cell 保持 `BLOCKED`。Target-only 是唯一 release-executable path。

历史 v2 evidence 仅可用于 regression comparison。它不含新的 Target-only、backend-plan、
topology、registry、trace、statistics 或 telemetry identity，不能通过改标签升级。
