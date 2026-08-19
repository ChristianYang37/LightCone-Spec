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

1. 严格 Target-only/Static/TTS/L0 runtime-method 与 backend-native 配置，disabled path 不分配 adaptation
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
第七个 patch 还会在该 official response 路径保留 server 产生、内容绑定的 native ITL
result pointer。远端 collector 将其作为与 terminal 关联但独立内容寻址的 unsigned bundle
发布；只有后续本地 external-control proof 才能把任一 artifact 提升为正式证据。

Patch 已实现 content-bound
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1` capability 与
begin/reset/finalize endpoint。当前最新 patch 的准确 SHA-256 为
`38b5ec81b9d75950558f8c72c1297bab47badf89d855b3e13dc1ad1c639f7d95`，final tree 为
`c6571336b70cd5f0e0f609d731a65fa98fd7e0b2`；manifest 仍是 authority。Lifecycle 绑定
run/nonce/plan/rank、process/session/reset lineage、expected request ID、准确 ordered token ID、
terminal coverage、Static aggregate safety，以及 adaptive-runtime request/round/update/KV/performance
row。它暴露 signer plugin boundary，但不捆绑 trusted hardware key 或 release signer。因此
所有 source implementation 在对应的本地 external-control proof 验证前仍 fail closed；仅有
CPU contract coverage 不能授权 formal method run。

第六个 patch 让 speculative width 计数识别 Target-only。真正的 Target-only server
（`spec_algorithm=none` 且未设置 draft width）不会再执行 `int(None)`：terminal evidence
仍将 speculative counter 标为不适用，数值型 `/server_info` speed view 则报告零个
verified draft。任何启用 speculative algorithm 却缺少 draft width 的配置仍会 fail closed。

第七个 patch 分别冻结 TP2 与双 TP1/DP2 replica 的 readiness contract。TP2 只在 all-rank
two-phase/sharded update 路径绑定 NCCL；TP1/DP2 使用 sticky replica-local control，不存在
adaptation collective 或跨 replica gradient averaging。同一 patch 还加入不可 skip 的
`session_reset_tp1` 八项 live-server suite。source-owned runner 对 cold process 与复用的
adapted process 使用同一份精确 launch manifest，绑定 token/reset/HBM/fixed-address/HTTP/
fallback/terminal observation，并要求运行前后 assigned GPU 的 compute-process 都为空。
这些路径在产生精确、本地控制的 GPU proof 前保持
`IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF`；remote worker 永远不持有 offline key。

Distributed finalize 路径还会绑定 warmup 与 scored schedule 的完整有序 client-lifecycle
digest。只有 client 证明未提交、且准确包含在该 digest 中的 scored request ID，才允许不
出现在 native rank coverage；每个已提交 request 仍须有匹配的 native completed/aborted
terminal，已经观察到的 native row 不能被降格为 non-submitted，所有 warmup row 则必须
offered、submitted 且 completed。因此 rejected、timed-out、cancelled 与 unfinished 等
client outcome 会留在注册 serving 证据中，而不会让其他方面有效的 scientific run 变成
不完整的 distributed terminal。

第二个 patch 还提供 opt-in native committed-token observation contract。每个 timestamp
是在 stop handling 后，streamer 枚举已经 committed 的最终 token prefix 时由 CPU 采样；它
既不是 decode production time，也不是 CUDA event。因此 producer 明确标为
`CPU_CONTRACT_ONLY`，formal release allowlist 继续为空，这些 event 不能支持 E2 p99 ITL。
Coalesced SSE chunk 也不再被等分成虚构 ITL。Benchmark 会准确输出 expected、observed、
coalesced、missing client interval 与 coverage；coverage 不完整时，包含 p99 在内的全部
aggregate ITL statistic 都明确为 `UNSUPPORTED`/`null`，不会把稀疏子集升级为 distribution。

第七个 patch 已实现 DSpark、NEXTN、单节点 TP2 与 sticky 双 replica DP2 的 source-owned
qualification 和 formal-serving 路径。它还在精确的官方 selector compatibility decision
之后实现 TP1 EAGLE3：兼容 assignment 有真实 qualification bootstrap，formal 路径消费
durable external-control proof；不兼容或尚未验证的 assignment 会签署 N/A 或保持
`BLOCKED`。EAGLE 仍不支持。DSpark/NEXTN TP1、topology-specific TP2/DP2 suite、all-rank
two-phase publication、sticky replica isolation 和 native training interface 都已有 source
path，但在产生精确且零 skip 的 GPU qualification artifact 前仍为
`IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF`。缺少所需 native supervision row 时，
quota-shadow teacher acquisition 仍 fail closed。

第二个 patch 的 CPU committed-token observation 仍只用于诊断。它与第七个 patch 的
native qualification producer 明确分离；只有后者的 first-party token-production/commit
pointer 与 exact suite proof 在 dynamic GPU gate 通过后才有资格进入 native ITL release
policy。

TP1/DP1 DFlash adaptive runtime method 现在会执行 constant、按 published update 的
inverse-square-root 与有限 horizon cosine schedule，也会记录 intrinsic readiness 并应用
非负 logical delay，同时保持 fixed-barrier 与 first-ready publication policy。Recipe
authority 由 host 侧独立绑定且与 policy 正交；该 runtime support 不允许 TTS 继承 E1/E2 或
历史 AdamW recipe。这些 CPU/native contract 不构成 CUDA 性能证据。

同一个 patch 现在还提供 first-party
`sglang.schema_v3.source_owned_all_reset_session.v1` producer，以及 capability、initial-state、
reset admin endpoint。它要求 scheduler 已 drain 且 KV/prefix 已清空，通过 native cache-reset
路径恢复 DFlash adaptation，并绑定 RNG/counter、scheduler/telemetry、weights/master/moments/
candidate/cohort/update state、allocator/HBM 与 completion event。第三个 patch 在支持的
single-tokenizer HTTP/1.1 uvicorn protocol boundary 增加 source-owned HTTP accounting：真实
`connection_made`/`connection_lost` event 产生累计 process/generation/created/closed/current
snapshot，由 HTTP process 注入，并验证守恒与单调连续性。Granian HTTP/2 与 multiple-
tokenizer HTTP-process 路径会在生成该 capability 前 fail closed。Accumulator 在实际 HTTP
serving process 的启动
boundary 显式初始化：同一 PID 的重复初始化会被拒绝，fork child 则以自己的 generation
替换继承的 owner。Request counter、header 与 caller payload 均不能替代。
Initial-state endpoint 只 finalize 一次，并内容绑定它自己的 post-capability snapshot，不能
返回 provisional stale snapshot；generic terminal-evidence route 会在 manager dispatch 前拒绝
全部 reserved session action，故只有专用 HTTP wrapper 能注入 accounting。GPU reset semantics
仍为 `PENDING`；CPU-valid receipt 不授权 GPU reuse，任何
transition 失败都要求 fresh process。第三个 patch 只关闭 reset-state accounting slice。
第四个 patch 注册准确的有序 trace member，把每次 reset 与现有
native terminal begin/reset/finalize lifecycle 绑定，并从 scheduler-owned state 生成排除
warm-up 与独立 scored-clock receipt，最后用 lifecycle-terminal receipt 封存完整 native digest
chain。Close receipt 明确写入 `transport_close_pending=true`：携带该 receipt 的 HTTP response
不可能已经真实观测到自身的 `connection_lost`。Caller 仍须关闭 pool 并终止 process。Durable
live-process reuse、formal GPU reset semantics 与 GPU validation 仍未实现，并保持 `BLOCKED`。

第五个 patch 增加 source-owned
`sglang.schema_v3.source_owned_compile_cache_lifecycle.v1` producer 与专用
begin/finalize endpoint。Scheduler 绑定准确的 plan、cache key、model lock、sampling
profile、prewarm manifest、physical assignment、budget、inventory、patched tree、process、
drained boundary 与有序 request terminal。Host consumer 复用同一个 pinned official bench
pool；submitter 返回值不会被当作 evidence。该链路严格标为 `CPU_CONTRACT_ONLY`。JIT time、
cache hit/miss/write 与 CUDA Graph capture/replay counter 均保持 `null`，并带
`gpu_compile_semantics_unavailable`；formal plan/source allowlist 与独立 GPU-vetted source
registry 继续为空，因此 COMPILE 在 mutation 前仍为 `BLOCKED`。

这些 row 保持 `BLOCKED`，不会伪装成 DFlash，也不会作为可运行 `UNMEASURED` work 上报。
底层 DFlash implementation 与有效 terminal envelope 没有绑定固定 tree/challenge 的
allowlist out-of-band signer 时，仍不能升级为可声明 evidence。本 release 不报告 CUDA
graph、multi-GPU、速度、容量或任何其他 GPU 结果。

## 应用

`patches/sglang/apply.sh` 只接受位于准确 upstream HEAD 的 clean checkout，按注册顺序使用
`git am` 应用 series，并把结果 Git tree 与 manifest 比较。Verifier 与 public-tree gate 会
检查每个注册 patch digest 以及 canonical manifest sidecar：

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
不足以证明 GPU 能力。Patch 已实现注册的 DFlash、DSpark、NEXTN、compatible EAGLE3、TP2
与 sticky DP2 source path，但 release trust policy 未配置 signer，且任何路径都没有 fresh
exact dynamic GPU proof。因此 Static/TTS/L0-naive/LightCone industrial cell 会在任何
mutation 前 `BLOCKED`，而不是可运行 `UNMEASURED` work。Test signer、provider object
attribute 或 caller-supplied verifier 都不能解锁该 policy。Generic EAGLE 不受支持；不兼容
的 EAGLE3 组合保持 N/A 或 `BLOCKED`。Target-only 是当前唯一可用于 claim 的 path。

历史 shared-tuned-AdamW evidence 只是 matched-recipe publication-policy diagnostic，不是
TTS-paper reproduction，仅可用于 regression comparison。它不含新的 Target-only、backend-plan、
topology、registry、trace、statistics 或 telemetry identity，不能通过改标签升级。
