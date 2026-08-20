# CLI

[English](../en/cli.md) · [首页](../../README_zh-CN.md)

## 命令 Surface

准确参数以 `lightcone-spec --help` 与 `lightcone-spec COMMAND --help` 为准。Schema-v3 与
industrial 命令包括：

| 命令 | 用途 |
|---|---|
| `doctor` | 只读 host、Python、CUDA 与 source identity 报告 |
| `validate-config` | 验证一份严格 schema-v3 run config 与 sidecar |
| `bind-formal-workload-authority` | 绑定 release allowlist 中的本地 LiveCodeBench v6 Hard 或 MATH-500 Level 5 raw source |
| `revalidate-formal-workload-authority` | 重开诊断性 workload binding，并重放其 path、revision、raw bytes 与完整筛选结果 |
| `build-industrial-registry` | 绑定一个或更多稳定 logical rank slot 并生成不可变实验 DAG |
| `collect-gpu-inventory` | 收集 nonce-bound physical GPU/topology inventory 与 raw probe receipt |
| `publish-tts-drafter-native-loss-source` | 发布唯一 code-owned pinned DFlash loss descriptor |
| `assemble-gpu-fleet-inventory` | 把重复传入的 single-host inventory/interference pair 组合成 content-bound fleet inventory |
| `build-interference-envelope` | 派生当前 serial interference envelope 及其 inventory-bound raw receipt |
| `materialize-interference-calibration-bootstrap` | 从 raw preflight activation 与准确 inventory 派生仅供校准的 two-way execution envelope |
| `reduce-interference-calibration` | 重开 path-bound execution bundle 与 terminal authority，把 raw isolated/simultaneous evidence 归约为准确 cardinality rule |
| `reduce-e1-activation` | 从 sealed E3a evidence 派生唯一 68-cell E1 slice |
| `reduce-e2-activation` | 从 E1 Pareto artifact 或 prior survivor materialize 一个 E2 round |
| `reduce-e2-successive-halving` | 把准确 E2 stage evidence 归约为 sealed survivor receipt |
| `materialize-confirmation-pilots` | 为一个 confirmation family 激活恰好四个 excluded pilot |
| `reduce-confirmation-family-power` | 把一个 family 的准确四个 pilot block 归约为 sealed power plan |
| `materialize-confirmation-prefix` | 只激活 powered family 已封存的 12--20-block final prefix |
| `validate-evidence-alias` | 用 registry、hardware、inventory 与 terminal authority 重放 raw alias manifest |
| `build-evidence-dependence-map` | 从 reducer 签发的 alias artifact 保留 shared-observation dependence |
| `materialize-stage-activation` | 从 raw registry、lineage、runtime 与 split authority 重放 generic stage dispatchability |
| `materialize-industrial-budgets` | 从 reducer、load、inventory、policy 与 capacity authority 派生 fail-closed `BudgetPlan` |
| `bind-industrial-budget-authority` | 把声明的 `BudgetPlan` 绑定到完整 tagged raw activation/load/capacity 闭包 |
| `estimate-industrial-budget` | 在准确 physical inventory 与 interference envelope 上重放 ready `BudgetPlan` |
| `plan-industrial-dispatch` | 冻结确定性、topology-aware GPU-pool wave 与 physical assignment |
| `materialize-dispatch-execution-bundles` | 绑定一份 path-only raw input graph，并在全 assignment 预检后发布完整 schema-v5 assignment-bundle set |
| `execute-dispatch-wave` | 重开一份已提交的 host-local manifest 来执行 receipt-bounded wave；`--host-request-stdin` 是 noninteractive remote-worker protocol |
| `seal-industrial-stage` | 绑定 activated completion、disposition、budget、runtime、split、dependency 与 locked output |
| `analyze-industrial` | 验证 schema-v3 terminal、budget、family-power 与 hardware evidence |
| `analyze-e3b-long-context` | 验证并归约隔离的 E3b long-context evidence family |
| `build-preliminary-speed-study` | 生成历史 `PRELIMINARY_DIAGNOSTIC_ONLY` 协议；绝不构成 industrial authority |
| `lock-models` | 把 model ID 解析为不可变 revision |
| `prepare-models` | 下载或 offline 验证 locked snapshot |
| `list-preliminary-tuning-candidates` | 写入历史 Full/LoRA 诊断 grid |
| `render-preliminary-target-only-runtime` | 渲染诊断性、关闭 speculation 的 Target-only endpoint |
| `render-preliminary-static-load-runtime` | 渲染诊断性、零 adaptation 分配的 Static endpoint |
| `render-preliminary-tuning-runtime` | 渲染历史 matched-recipe publication-policy diagnostic endpoint；不是 TTS reproduction |
| `run-preliminary-controlled-slice` | 测量一个 preliminary controlled slice |
| `collect-preliminary-static-load-screen` | 验证 preliminary Static load coverage |
| `advance-preliminary-tuning-stage` | 验证 preliminary halving stage |
| `select-preliminary-speed-config` | 应用历史 tuning-only rule |
| `select-preliminary-anchor-config` | 锁定 preliminary anchor，但不形成 claim |
| `render-preliminary-runtime` | 生成历史 matched-recipe diagnostic config 与 launch argv |
| `build-preliminary-confirmation-queue` | 生成 clean-server 诊断 job |
| `run-preliminary-confirmation` | 执行一个 preliminary method/block slice |
| `run-preliminary-target-reference` | 捕获 preliminary Target-only token-ID 轨迹 |
| `collect-preliminary-speed-study` | 派生 preliminary 诊断 table |
| `render-preliminary-replication-runtime` | 渲染 preliminary natural-task 或 profiler slice |
| `run-preliminary-natural-slice` | 运行一个 preliminary natural-task slice |
| `build-preliminary-profiler-plan` | 生成隔离的诊断 profile plan |
| `attest-preliminary-speed-study` | 输出确定性的 non-authority decision（退出 42） |
| `analyze-preliminary-speed-study` | 只生成 preliminary 统计（退出 42） |

隔离的 OnlineSPEC command family 为：

| 命令 | 用途 |
|---|---|
| `build-onlinespec-study` | 生成 provenance-bound comparison protocol |
| `verify-onlinespec-source` | 按 source audit 验证外部 clean checkout |
| `list-onlinespec-candidates` | 写 OGD、optimistic 与 ensemble candidate |
| `render-onlinespec-tuning-runtime` | 渲染 paired Static/candidate tuning endpoint |
| `run-onlinespec-tuning-slice` | 测量一个 learner tuning slice |
| `advance-onlinespec-tuning-stage` | 各 learner 独立 halving |
| `select-onlinespec-config` | 每个 learner 选择一个 safe terminal candidate |
| `select-onlinespec-anchor-config` | 锁定三个 terminal anchor，不声明 grid optimum |
| `render-onlinespec-runtime` | 渲染 sequential Static 与 learner endpoint |
| `build-onlinespec-queue` | 生成 randomized clean-server comparison job |
| `run-onlinespec-confirmation` | 执行一个 learner/method block |
| `collect-onlinespec-study` | 派生隔离 comparison table |
| `attest-onlinespec-study` | 输出确定性的 non-authority decision（退出 42） |
| `analyze-onlinespec-study` | 只生成 learner-versus-Static 诊断区间（退出 42） |

历史 speed-study manifest/API/CLI surface 被严格固定为
`PRELIMINARY_DIAGNOSTIC_ONLY`。它不能消费 industrial registry、activation、budget、
completion authority 或 materialized bundle；其 table 与 receipt 也绝不会被
`analyze-industrial` 接受。历史 schema-v2 manifest 仍可在强制 preliminary scope 下读取，
但不能改写或重新标成 formal。唯一 formal execution 命令是 `execute-dispatch-wave`；它会
重开 materialized bundle 并进入 industrial executor。不存在 generic method override，
也不存在把无 receipt 目录转换为 completed evidence 的命令。

Bootstrap envelope 只授权生成八个已注册 Static calibration observation，不能授权
headline co-tenancy 或更大 cardinality。`reduce-interference-calibration` 会在打开 caller
path 前先检查 source-pinned public root 和 fresh dynamic control；缺少真实签名 evidence 时
只写入 `BLOCKED` decision，不会生成 calibrated envelope。Preflight stage 只有在重开同一份 raw execution authority
后才能封存 `runtime_envelope=PATH`，绝不接受手写 rule list 或 bare digest。

## Compile-cache plan 绑定

所有 preliminary 与 OnlineSPEC runtime render 命令都要求
`--compile-cache-plan /absolute/path/to/plan.json`。它只是 diagnostic launch
authority，不是 attestation，也不授权进入 formal DAG。`CompileCacheKey` 必须由通过的
`doctor` 报告、准确 model lock 与已渲染 RunConfig 以程序方式派生；不得凭记忆手填预期
host 值。当前 diagnostic launcher 只接受 `bfloat16`、`cuda_malloc_async`、graph bucket
`(1,)` 与空 caller build flags。创建 cache attempt 前，它会重新观测 Python、Torch、
Triton、Torch CUDA build、`nvcc`、driver、所选 GPU model 与 SM，并在导入 Torch 前应用
allocator 环境。

通过 source API 派生 key，并签发不可变的 diagnostic build plan：

```python
import json
from pathlib import Path

from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.locking.models import ModelLock
from lightcone_spec.orchestration.runtime import (
    build_target_only_run_config,
    derive_diagnostic_compile_cache_key,
)
from lightcone_spec.runtime.compile_cache import CompileCacheLaunchPlan

doctor = json.loads(Path("/absolute/runtime/doctor.json").read_text())
model_lock = ModelLock.load("/absolute/runtime/model-lock.json")
sampling = SamplingProfile.load("/absolute/runtime/sampling-profile.json")
devices = doctor["gpu"]["parsed_inventory"]["devices"]
if len(devices) != 1:
    raise RuntimeError("Target-only diagnostic requires one visible doctor GPU")
gpu_uuid = devices[0]["uuid"]
config = build_target_only_run_config(
    concurrency=1,
    gpu_uuid=gpu_uuid,
    model_lock=model_lock,
    sampling_profile=sampling,
)
key = derive_diagnostic_compile_cache_key(
    doctor_report=doctor,
    model_lock=model_lock,
    config=config,
)
cache_root = Path("/absolute/runtime/compile-cache")
build = CompileCacheLaunchPlan.issue(
    key=key,
    cache_root=cache_root,
    cache_mode="build",
)
build_path = build.write(Path("/absolute/runtime/build-plan.json"))
```

把 `build_path` 以及同一个 `gpu_uuid`、concurrency、model lock 和 sampling profile 传给
render 命令。Renderer 会重新构造完全相同的 source-owned RunConfig，并把预期 plan、key
与 RunConfig SHA-256 写入 child argv。例如：

```bash
lightcone-spec render-preliminary-target-only-runtime \
  ... \
  --gpu-uuid GPU-01234567-89ab-cdef-0123-456789abcdef \
  --compile-cache-plan /absolute/runtime/build-plan.json
```

`--gpu-uuid` 是必填参数，必须等于 PASS doctor inventory 选中的唯一物理 `GPU-*` UUID。
Renderer 会把它写入现有的 `RuntimeConfig.device_identity`；官方 launcher 会稳定重开这份
RunConfig，并把同一个 UUID 绑定到 `CUDA_VISIBLE_DEVICES`。包含逗号或空白、以及不以
`GPU-*` 表示的 selector，会在创建 runtime 文件或 subprocess 前被拒绝。

Renderer 会把预期 plan、key 与 RunConfig SHA-256 写入 child argv。Child 会稳定重开两份
canonical artifact 及 sidecar，从 RunConfig 派生准确 algorithm 与 draft width，并在 cache、
model 或 GPU mutation 前拒绝 identity/argv 变化。它还要求 Torch 能看到准确一张可用 CUDA
设备，并拒绝在 Torch 已导入后再初始化 allocator。Render 后不得编辑、重新生成、移动或
替换 artifact。

Preliminary model root 只绑定 locked revision path，并不携带 formal
`PreparedModelContentAuthority` replay。因此 launcher 只接受 build mode，且把 cache receipt
封存为 unattributed；它不能授权 reuse、formal execution、attestation 或 `MEASURED`。Source
API 虽可针对同一 key/cache root 的已完成 release-builder receipt 签发 reuse plan，但任何
preliminary receipt 都不满足该条件。在 formal content-authority execution path 提供此 receipt
之前，应签发新的 build plan。

## Formal external workload authority

这些命令绝不会下载 formal benchmark 数据。Bind 命令只接受
`livecodebench_v6_hard` 或 `math500_level5`；准确 repository revision、raw-file
SHA-256、row count、筛选协议与完整 selected-row SHA-256 全部由源码拥有。它按 raw 顺序
选择全部准确匹配项，绝不截取前 32 条。

```bash
lightcone-spec bind-formal-workload-authority \
  --workload math500_level5 \
  --source /absolute/path/to/math500-locked.json \
  --content-verification-receipt /absolute/path/to/content-E3a.json \
  --now-ns 1786900000000000000 \
  --output artifacts/industrial/math500-workload-authority.json

lightcone-spec revalidate-formal-workload-authority \
  --authority artifacts/industrial/math500-workload-authority.json \
  --content-verification-receipt /absolute/path/to/content-E3a.json \
  --now-ns 1786900000000000000
```

两个命令都要求当前有效、经 root 验证的 content-verification receipt，并显式提供验证时间。
缺失或无效的 content authority 会在绑定 raw workload 之前 fail closed。成功绑定会报告
`BOUND_AUTHORIZED_CONTENT`，但生成的 wrapper 仍永久携带
`formal_execution_authorized=false`；它只是必须与同一 verified content authority 一起
replay 的 workload input identity，不是 GPU dispatch、attestation 或 formal result。
旧 source-allowlist API 仍是非授权 diagnostic，不是 production CLI 路径。

## Industrial Registry 工作流

Registry 只包含 scientific identity 与一个或更多 logical rank slot，不包含 host-specific GPU UUID：

```bash
lightcone-spec build-industrial-registry \
  --logical-gpu-slot logical-rank-slot-0 logical-rank-slot-1 \
  --base-port 24000 \
  --cache-root runtime-cache/industrial \
  --evidence-root artifacts/industrial \
  --seed 20260811 \
  --output artifacts/industrial/registry.json
```

输出嵌入 generator identity、输入 parameter、完整 declaration 与 registry SHA-256。加载时
会重新生成 registry 并比较准确内容，因此手工编辑 cell 会被拒绝。独立 content-bound
inventory 提供 physical UUID、model/memory/capability、PCI/NUMA/interconnect topology、
availability 与 topology group。

没有 E1/E2 或 confirmation-family 专用 reducer 的 stage 使用 first-party generic reducer。
它的 raw manifest 只能准确包含 registry、stage name、runtime、split 与完整有序 dependency
receipt path；不接受 caller cell list 或 bare activation plan。Preflight 的 receipt list 为空，
canonical genesis authority 会由准确 registry 派生：

```json
{
  "schema_version": 1,
  "kind": "industrial_registry_stage_activation_manifest",
  "registry_artifact": "artifacts/industrial/registry.json",
  "experiment": "preflight",
  "runtime_artifact": "artifacts/industrial/runtime.json",
  "split_artifact": "artifacts/industrial/preflight-split.json",
  "dependency_receipts": []
}
```

为 manifest 写相邻 `.sha256` 后，可生成 canonical artifact 供检查：

```bash
lightcone-spec materialize-stage-activation \
  --manifest artifacts/industrial/preflight-activation-manifest.json \
  --output artifacts/industrial/preflight-activation.json
```

Consumer 通过 `--activation-plan` 接收 raw manifest path 并现场重放 reducer；单独传序列化
output 会被拒绝。Reducer 从 registry status 与 release dispatch predicate 为每个 declared
stage cell 生成 disposition。当前 release 的全部 preflight cell 都被阻止：Target-only
compile cell 缺少 release-owned 的准确 prewarm manifest、graceful-finalization ACK 与原子
attempt/final-cache result pointer；Static/TTS/L0-naive/LightCone 也没有受支持的 execution contract。
因此 command 写出 canonical `BLOCKED` artifact 并返回 42。E6 download cell 由于没有
first-party download terminal contract，也会被独立阻止。

Budget materialization 会 fail closed。它消费 reducer-generated stage/family activation、完整
`BudgetPolicy`、每个 activated serving cell 各一份 `BudgetLoadBinding`、独立来源的
`CapacityEnvelope`，以及完整 physical inventory。Formal capacity authority 还必须成对传入
`--capacity-manifest` 与 `--capacity-verification-receipt` raw artifact。Verifier 会重新打开
每个 path-bound inventory/source、provider quota、host capacity 与逐 cell sizing input，且
只接受 release-owned trust root。只有 load coverage、provider GPU quota 与 maximum-attempt
disk capacity 全部闭合时才写 `READY`；否则保留真实派生 budget，写入不可变的诊断性
`UNRESOLVED` plan 并返回 42。Bare envelope 或 SHA-256 绝不构成 execution authority。
每个 activated serving cell 都要重复一次 `--budget-load-binding PATH`：

```bash
lightcone-spec materialize-industrial-budgets \
  --registry artifacts/industrial/registry.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --inventory artifacts/industrial/inventory.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --output artifacts/industrial/budget-plan.json
```

不存在 caller-selected cell 或 duration-sequence fallback。Preflight 现在由 generic reducer
闭合；若省略 bound raw manifest，仍会得到明确 unresolved authority error，caller-authored
activation 不能绕过。Serving activation 还要为每个 activated serving cell 重复传入一次
`--budget-load-binding PATH`。

Formal consumer 不接受单独的 `BudgetPlan`。Materialization 后，必须用 tagged raw
activation manifest（不是 serialized activation summary）发布其 path-bound 原始闭包：

```bash
lightcone-spec bind-industrial-budget-authority \
  --activation-manifest artifacts/industrial/preflight-activation-manifest.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --output artifacts/industrial/budget-authority.json
```

每个 activated cell 还要重复 `--budget-load-binding PATH`。该命令会重开所有原始来源、
现场重跑注册 reducer 并准确比较 declared plan，只写出 binding；它绝不会把
`UNRESOLVED` plan 变成 execution permission。

Planning 会重新加载 `BudgetPlan`，用相同 raw input 现场重跑 materialization，并在调度前
要求两者完全相等；同时还强制要求成对的 interference authority：

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference-envelope.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --output artifacts/industrial/dispatch.json
```

Frozen plan 会绑定准确 inventory/interference envelope、每个 cell 的 budget digest、physical
UUID/rank/port/topology assignment、wave membership 与 scheduler identity。Scheduler 接受任意
same-host inventory，并对 1/2/4/8/16 GPU 有 regression test。一个 wave 绝不超过已校准的
co-run class，因此八块空闲 GPU 不表示可以执行八路 headline concurrency。Envelope 通过
structural loading 不能证明 calibration；formal timing 仍要求 raw isolated/co-run
evidence 与 trusted hardware binding。

`materialize-dispatch-execution-bundles` 闭合 frozen planner output 与 formal execution 之间的
结构边界。它的 schema-v1 request 只包含 absolute resolved raw-artifact path、对应 source role，
以及 dispatch 中每个 cell 各一组 assignment-runtime path；刻意不接受 caller-supplied
assignment SHA、execution-plan SHA/summary、output root 或 semantic hash。Reducer 会重开 request
及每一份 sidecar，从 frozen dispatch plan 现场派生 assignment，并要求准确的一一 assignment
覆盖，之后才能构造 schema-v5 bundle。

创建 fresh publication directory 或任何 renderer artifact 之前，命令会预检全部 assignment
的 raw activation/completion、budget、capacity、interference、topology、runtime、sampling、
model、compile、nonce、launch-policy，以及可选 trainable/failure authority。Tagged generic、E1、
E2、confirmation-pilot、confirmation-final 与 stage-aggregate activation authority 都会从完整
path closure 现场重放。Final prefix 只使用 schema-v4/native completion authority；bare
completed-cell ID 会被拒绝。E3b/E5 family aggregate 仍保持独立；E5 还要求为其 264 个非 family
failure-injection cell 提供 deterministic auxiliary activation/completion 准确覆盖。Formal
reconstruction 还必须取得 live `READY` `BudgetPlan` 及其 path-bound materialization authority；
自洽的 planner 或 execution summary 不能自我授权。

Output path 必须为 absolute、resolved、尚不存在，并位于稳定且 owner-private 的 parent 之下。
全 assignment 预检通过后，命令创建 private `0700` directory，以 exclusive write 写入每个
schema-v5 bundle 与相邻 sidecar，最后写入 `dispatch-execution-bundle-manifest.json` 及其
sidecar 作为 commit marker。中断留下的 partial directory 会被保留，但因没有可消费 manifest
而不构成 publication；重试必须使用另一 fresh directory，不能通过删除 evidence 恢复。

```bash
lightcone-spec materialize-dispatch-execution-bundles \
  --request /absolute/path/to/dispatch-bundle-request.json \
  --output-directory /absolute/private/path/dispatch-bundles-attempt-0001
```

`execute-dispatch-wave` 绝不会把 planner JSON、
`IndustrialExecutionPlan.to_dict()` summary 或 raw bundle path 当作 launch authority。它只接受
一份必需的 `--materialization-manifest`。通过 release dispatch trust gate 后，loader 会重开
manifest、对应 request 与 dispatch plan、每个 schema-v5 member，以及完整 path-bound
construction graph；input byte 变化、member 被替换、assignment coverage 不完整或 bundle
位于 manifest directory 外都会被拒绝。之后 execution 才会重放共享 scheduler authority，
并构造所请求 wave 的 physical plan（及 resume receipt 实际引用的历史 plan）。Compile/download
和 unsupported serving assignment 仍明确 `BLOCKED`，每个可能真实 launch 的 plan 也仍必须
通过完整 public validation boundary。

每次命令只执行一个 `--wave-index`。Wave 0 不带 resume input；成功 prefix 的下一波必须传
上一轮 `--resume-receipt`，failed wave 则在同一 index resume。任何 sibling runner 启动前，
命令都会先向 private `RECEIPT.attempt-journal` durable append 一个 wave intent 与所有
per-assignment intent；每次 terminal 或 failure 返回后再 append finish、准确 retry identity
与累计 monotonic cost。因此 partial wave 会保留成功 sibling，只重跑失败 sibling。若 intent
没有 finish，成本无法确认，会以
`dispatch_attempt_intent_without_finish_cost_unresolved` 阻断。

Formal manifest-based dispatch 使用 schema-v2 attempt-journal manifest，并绑定准确的
materialization-manifest SHA-256。因此，即使 dispatch plan 没有变化，使用另一份 bundle
publication 重开同一个 journal 也会产生 identity mismatch。

`--receipt-output` 是 schema-v2 canonical immutable envelope，同时内嵌 receipt、structured
sidecar 与准确 journal manifest/head/event-count prefix。Resume 只把 caller receipt 当作
anchor：授权来自 raw journal 重放及每个成功 `AssignmentTerminalAuthority` 的现场重开。
若所有 finish 已落盘、但 coordinator 在 envelope 前崩溃，用相同 wave/output 重跑命令即可
恢复，不会再次启动 runner。`RECEIPT.sidecar.json` 只是 derived convenience copy，因此在
该 copy 写入前崩溃不会丢失 resume authority；parent directory 必须已经存在。

相对于已发布 envelope prefix，hash chain 能检测删除、替换、symlink 与 event 联合重哈希；
它不是外部签名。能够同时替换整个 unsigned journal 与全部 anchor 的 actor 超出这个纯软件
恢复威胁模型。Formal GPU claim 仍必须使用下述 release-owned trusted attester。

```bash
lightcone-spec execute-dispatch-wave \
  --materialization-manifest \
    /absolute/private/path/dispatch-bundles-attempt-0001/dispatch-execution-bundle-manifest.json \
  --wave-index 0 \
  --receipt-output artifacts/industrial/dispatch-wave-0.json
```

当前 source release 固定 offline Ed25519 public root；真实 inventory 后必须另行提供
root-signed short-lived deployment/hardware policy 和对应 typed controls。Caller/test signer
仍不能解锁 formal execution；bare `CapacityEnvelope` 同样不能授予 execution authority。因此，即使 request 在其他方面完整，
materialization 也会在全 assignment 的 release preflight 处返回 `BLOCKED`/42，且不会创建
publication directory。Formal execution 则会独立地在 entry trust gate 返回 `BLOCKED`/42，
发生在读取 materialization manifest、创建 receipt parent/evidence root、导入 serving client
或启动 process 之前。Bundle publication 只是 structural authority，不是 GPU attestation 或
execution permission。Fresh execution 会拒绝未被 journal 授权的已有 per-plan trace file；
resume 必须同时重放 raw append-only attempt chain 并现场重验 structured terminal binding。
Bare terminal digest 或 caller 联合重哈希的 schedule JSON 都不能跳过 work。若 coordinator
在 durable intent 后、durable finish 前崩溃，则保持 `BLOCKED`，绝不伪造 monotonic cost。

Dispatch 前要对同一个 activated set 估算。每个 budget field 都是显式值；report 会分别列出
optimistic、registered 与 quota-envelope scenario 的 wall、compute、reserved 与
whole-instance billed GPU time：

```bash
lightcone-spec estimate-industrial-budget \
  --registry artifacts/industrial/registry.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference-envelope.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --output artifacts/industrial/budget-report.json
```

Estimator 会执行同样的 raw-input rematerialization 与现场 capacity revalidation。Report
同时绑定 scheduler inventory SHA-256 与 interference-envelope SHA-256；若准确 scheduler
replay 不可执行，它仍保留真实派生的 GPU-hour diagnostics，记录具名 unresolved assumption
并返回 42；该 report 不能用作 launch authority。

Preflight 的实际 GPU hours 只能从已 finalized、sealed 的 preflight chain 物化。该命令不
接受 duration、run count 或 reserve scalar；唯一 compile 与 exactness lifecycle authority
会从 `--source-authority` 传递重开，另强制要求八个不重复、path-bound 的 interference
lifecycle proof，其 cell ID 必须与 finalized 1+1+8 source 精确一致：

```bash
lightcone-spec materialize-preflight-gpu-hour-envelope \
  --dispatch-receipt /absolute/evidence/preflight-dispatch.json \
  --remote-raw-receipt /absolute/evidence/preflight-remote-raw.json \
  --source-authority /absolute/evidence/preflight-source.json \
  --activation /absolute/evidence/preflight-activation.json \
  --coverage /absolute/evidence/preflight-pointer-coverage.json \
  --stage-coverage /absolute/evidence/preflight-stage-coverage.json \
  --interference-lifecycle-proof 9a4d4b4f84399bbd9e33e542d110e237d33ef0d1f738d60406399611cadaf6d6=/absolute/evidence/lifecycle-0.json \
  --interference-lifecycle-proof a88676a576111058c46b44dc2754a4d171e6c8f2226c3ebb65f2db4716c5c253=/absolute/evidence/lifecycle-1.json \
  --interference-lifecycle-proof 11de28835716cb8fa5af2dea84571e2ae930f6f3642a6ccecaaa46529ca76f10=/absolute/evidence/lifecycle-2.json \
  --interference-lifecycle-proof 2761af3f18adceab26d68bc7887b50f21575c8f54062738ff893d402ffa5c32e=/absolute/evidence/lifecycle-3.json \
  --interference-lifecycle-proof 1e7933ad9c1b95af086d1a8626d882beb4f376987786e5799b2e440f4ae536dd=/absolute/evidence/lifecycle-4.json \
  --interference-lifecycle-proof 8902eb962b1ab7703a29446c1f4bb56f183d620496043e02814bfd1d0d94a630=/absolute/evidence/lifecycle-5.json \
  --interference-lifecycle-proof 52f11f9c36ea5016a0cba90b4a1ae843b77a15be00936df846f8a9a6cfe620f8=/absolute/evidence/lifecycle-6.json \
  --interference-lifecycle-proof e93df280d22db443b27ee32e6dc95eed9bc7ecb6a1f4a439dddf57c93c5206e5=/absolute/evidence/lifecycle-7.json \
  --formal-runtime-authority-manifest /absolute/evidence/runtime-authority.json \
  --source-output /absolute/evidence/preflight-gpu-hour-source.json \
  --now-ns 2000000000 \
  --output /absolute/evidence/preflight-gpu-hour-envelope.json
```

两个 output path 必须互异、absolute、normalized 且尚不存在；source manifest 与 schema-2
envelope 都以 atomic no-replace 发布。Envelope 可直接进入 offline scientific signer 的
`stage-gpu-hour-envelope` typed decoder；finalized signed wrapper 与 source manifest 随后进入
`reserve-formal-stage-gpu-hours`。普通 formal stage 的对应命令会在 public、path-bound
stage-source rebuild bundle 出现之前保持不可用；digest、generic JSON 或序列化 private
execution seal 都不能替代它。

### Source-owned 方法 authority

创建 `ProtocolLock` 前，必须从准确 bound source 发布三个 code-owned 方法输入：

```bash
lightcone-spec publish-tts-calibration-tuning-window \
  --tuning-workload-authority /absolute/source/lcb-hard-authority.json \
  --content-verification-receipt /absolute/evidence/content-master.json \
  --output /absolute/source/tts-tuning-window.json

lightcone-spec publish-tts-drafter-native-loss-source \
  --output /absolute/source/drafter-native-loss.json

lightcone-spec publish-tts-calibration-source-authority \
  --paper-pdf /absolute/source/tts-v2.pdf \
  --paper-source /absolute/source/tts-v2-source.tar.gz \
  --tuning-workload-authority /absolute/source/lcb-hard-authority.json \
  --content-verification-receipt /absolute/evidence/content-master.json \
  --tuning-window /absolute/source/tts-tuning-window.json \
  --trainable-plan-authority /absolute/source/trainable-plan.json \
  --drafter-native-loss /absolute/source/drafter-native-loss.json \
  --output /absolute/evidence/tts-calibration-authority.json

lightcone-spec publish-chronobelief-source-authority \
  --paper-pdf /absolute/source/chronobelief.pdf \
  --tex-source /absolute/source/chronobelief.tex \
  --output /absolute/evidence/chronobelief-authority.json

lightcone-spec publish-e1-recipe-anchor-authority \
  --trusted-content-bundle /absolute/source/trusted-content.json \
  --trainable-plan-authority /absolute/source/trainable-plan.json \
  --output /absolute/evidence/e1-recipe-anchor-authority.json
```

必须先完成 schema-2 master content ceremony，且该 ceremony 明确禁止把 TTS tuning
window 放入 master content。随后这些命令会深读 durable master receipt、其准确
replay-reservation record、在 receipt 记录的 verification time 上复核原始 signed
workload authorization，并重开 bound LCB authority。Raw 或 expired workload authorization
不能替代 receipt。Schema-4 window 绑定 receipt digest、verification time 与 reservation
identity，并使用 atomic no-replace 发布。这些命令不接受 caller 提供的
authority digest、tuning partition、time 或 recipe override。Code-owned selector 在 versioned
namespace 下 hash 全部准确 LCB-hard problem ID，将最小四个作为 excluded pilot，并
绑定完整 complement。TTS source publisher 会重开 pinned DFlash loss 的准确描述：float32
target-to-draft forward KL、valid-row mask、`exp(-(k-1)/7)` position weight、
temperature 1 与 masked weighted normalization。source-point value correction 不是独立
proximal penalty，因此无需提供或调节 proximal coefficient。只要 exact source pin、post-master
window、canonical plan 与 loss descriptor 均通过，就能发布 `ProtocolLock` input。该结果分类为
project-calibrated runtime baseline，而非 paper reproduction。ChronoBelief 命令仍是独立的项目
自有 preregistration。
当前 E1 publisher 还要求一个 runtime-`BOUND` trusted content bundle。其 schema-3
artifact 会绑定并重开该准确 bundle，并把 plan 的 target/drafter revision 与 prepared
root 连接到同一 bundle。历史 schema-2 E1 artifact 仅可读取，不能进入 trusted
schema-5 ProtocolLock。E1 anchor publisher 只接受显式 structural selector：registry 中唯一的
Qwen3-8B/DFlash LC-candidate `full`/`last1` AdamW、width-8/concurrency-4 slot。其他即使合法的
plan 也会被拒绝；该 selector 不编码 observed winner。

Stage 的 activated cell 与 disposition durable 后再封存。`--inventory PATH` 是 completed
evidence 所用 physical GPU identity 与 topology 的 authority。每次重复的
`--locked-output` 只能使用 `NAME=PATH`：name 必须唯一，且 `PATH` 必须指向
content-bound artifact；SHA-256 literal 不能替代该 artifact。Runtime/split 同样必须是 bound
file，dependency 必须传 receipt file，而不是复制 hash string：

```bash
lightcone-spec seal-industrial-stage \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --experiment preflight \
  --runtime-artifact artifacts/industrial/runtime.json \
  --split-artifact artifacts/industrial/preflight-split.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --completed-cells artifacts/industrial/completed-cells.json \
  --locked-output runtime_envelope=artifacts/industrial/runtime-envelope.json \
  --output artifacts/industrial/receipts/preflight.json
```

封存 E2 时还强制要求额外 authority：`--e2-final-stage-manifest PATH`。它必须是 raw
`halving_3` manifest；command 会现场重跑 first-party raw reducer，并要求结果为
`FINAL_RECIPE`。随后，`dflash_recipe=PATH` artifact 必须精确绑定该 final candidate、
registry、runtime、split 与 final-stage reduction。用 digest 替代 `PATH`、caller-authored
summary 或绑定错误 candidate 的 recipe 都会被拒绝：

```bash
lightcone-spec seal-industrial-stage \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --experiment E2 \
  --runtime-artifact artifacts/industrial/runtime.json \
  --split-artifact artifacts/industrial/e2-split.json \
  --completed-cells artifacts/industrial/e2-completed-cells.json \
  --e2-final-stage-manifest artifacts/industrial/e2-halving_3-manifest.json \
  --dependency-receipt artifacts/industrial/receipts/E1.json \
  --locked-output dflash_recipe=artifacts/industrial/e2-dflash-recipe.json \
  --output artifacts/industrial/receipts/E2.json
```

这些检查定义的是注册 trusted hardware attester 后的 claim-bearing 路径。当前 release 的
seal 仍会在签发 receipt 之前返回 `BLOCKED`；该 raw-authority contract 不会让 final
execution 或 performance claim 变为可达。Sealed output 只能授权 LightCone role，绝不能
改写单独 frozen 的 TTS 或 L0-naive recipe authority。

下游 stage 对其准确 declared dependency 重复 `--dependency-receipt`，随后把所有 completed
receipt 交给 planner：

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference-envelope.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --budget-load-binding artifacts/industrial/load-cell-000.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --receipt artifacts/industrial/receipts/preflight.json \
  --completed-cells artifacts/industrial/completed-cells.json \
  --activation-plan artifacts/industrial/next-stage-activation-manifest.json \
  --output artifacts/industrial/next-dispatch.json
```

Formal completed-cell row 会绑定 frozen physical assignment、准确 budget、terminal receipt、
强制 `BudgetObservationReceipt` 与三方 budget/terminal digest closure。缺失 phase timing、
compile/download N/A 理由或 GPU accounting 都不能替换为零。Resume 只接受完整 receipt，
绝不会只看目录存在。

Dispatch plan 是目标 protocol 数据，不证明 cell 可执行。Library industrial executor 会在
launch 前验证 provider state。固定 tree 已实现准确 native terminal begin/reset/finalize
hook，但没有配置 trusted hardware signer。Generic release-attested activation 只记录 canonical
blocked preflight disposition；它不会创造 compile runner、execution authority 或 performance
claim。Trusted `formal_single_operator_v1` CLI 是另一条 empirical lane：它不要求 external
signer，但仍要求准确 source/content/runtime、fresh GPU qualification、capacity、terminal 与
coverage artifact。CLI 不会静默 provision hardware，也不会启动 GPU。

## Fleet Inventory 与 Remote Host Wave

Fleet assembly 为每台主机消费一个 `--inventory PATH` 和一个
`--interference-envelope PATH`。两个 option 都要重复传入，其位置构成配对；数量不一致会被
拒绝。每个 input inventory 必须准确描述一台主机，每个 envelope 继续绑定该主机的 hardware
identity。

```bash
lightcone-spec assemble-gpu-fleet-inventory \
  --inventory /external/host-a/inventory.json \
  --interference-envelope /external/host-a/interference.json \
  --inventory /external/host-b/inventory.json \
  --interference-envelope /external/host-b/interference.json \
  --output /external/fleet/inventory.json
```

`GpuFleetInventory`、`GpuFleetScheduler` 与 `GpuFleetDispatchPlan` 是分发 independent
assignment 的 Python control-plane API。`GpuFleetScheduler` 只选择 host 与 serial
partition；每个 child plan 仍由唯一的 same-host `GpuPoolScheduler` 签发，并绑定 `host_id`、
host inventory digest、physical GPU UUID，以及在该 host 内无冲突的 port/cache/evidence/
contention resource；不同 host 可以重复 literal resource value。独立 remote execution
binding 再固定 host-local execution manifest。完整 gang、matched-geometry baseline anchor set 和
confirmation block 留在一台主机。若 gang 需要跨主机，会以
`cross_host_collectives_unvalidated` 拒绝；fleet composition 绝不创建 cross-host rendezvous。

Remote orchestration 当前同样只提供 Python-library coordinator API。它使用 coordinator-local
`SshHostRoute` 与 `execute_fleet_wave`；不存在 public fleet-coordinator CLI。Route 强制使用
SSH agent 与固定 `known_hosts` file。不含 path 的 route-authority digest 把 destination、port
与准确 known-host bytes 绑定到 attempt。Address、user、agent socket、known-hosts path/key
bytes 与 raw stdout/stderr 都留在 serialized request/receipt 之外。Password、token、private
key 与 provider credential 不得放入 argv、stdin、artifact 或日志。

Remote node 只暴露一个 worker entry point：

```bash
lightcone-spec execute-dispatch-wave --host-request-stdin
```

该 mode 供 coordinator 使用，不是 interactive shell；它与 `--materialization-manifest`、
`--wave-index`、`--resume-receipt`、`--receipt-output` 互斥，并从 stdin 读取一份有 size bound 的
canonical request。Worker 重开声明的 absolute host-local manifest，再向 stdout 写一份有界
canonical response。本地 pre-dispatch 拒绝形成 failed transport outcome；一旦 dispatch 可能
已经发生，timeout、connection loss、截断或 invalid output 以及缺失 authority 都必须形成
`REMOTE_OUTCOME_UNKNOWN`，既不能解释为 completion receipt，也不能直接重试。

一台主机失败不会使其他主机已完成的 receipt 失效。Unknown outcome 只能通过准确原始
destination、port 与 known-host-key authority 下的独立 fetch reconcile；取回的准确
receipt/evidence bytes 会在本地重算 content identity，缺失、伪造、不完整或超限 input 都继续
保持 unknown。只有 terminal-negative result 才能为同一 host
创建新的 receipt-bound attempt。Download、compile、profiler 与 shared-I/O contention 继续
保持 host-exclusive；headline concurrency 绝不超过该主机自己的 calibrated envelope，fleet
SSH concurrency 另有独立上限。

没有任何 command 接受 caller-selected trusted-attester bundle 作为 release authority。Package
固定 offline Ed25519 public root/fingerprint；真实 inventory 后由该 root 签署短期动态 policy，
且 formal dispatch 原子消费 deployment/control challenge。私钥只通过本地
`python -m lightcone_spec.runtime.offline_signer` 的 TTY prompt 或数字 `--key-fd` 输入；CLI
没有 key path/bytes 参数，GPU host 也不得接收私钥。缺失、过期、replay、wrong-key 或硬件不
匹配时继续 fail closed。

本地 signer 的完整 help surface 为：

```bash
python -m lightcone_spec.runtime.offline_signer sign-deployment --help
python -m lightcone_spec.runtime.offline_signer sign-control --help
python -m lightcone_spec.runtime.offline_signer sign-scientific --help
python -m lightcone_spec.runtime.offline_signer finalize-scientific --help
```

`sign-scientific` 只接受 closed allowlist 内的 typed scientific wrapper；生成的 candidate 必须
再由 `finalize-scientific` 重验 policy，并将 challenge 原子写入 private single-use ledger 后才
能成为 signed artifact。不存在 generic JSON signer，也没有 key path/bytes 参数。

## 身份与 Topology 链

最小 industrial identity chain 为：

```text
source + patched tree + model/data/trace locks
                         |
                         v
          registry logical slots + activation reducers
                         |
                         v
 inventory + interference + budgets -> frozen physical waves
                         |
                         v
       terminal + reset + budget-observation receipts
                         |
                         v
 family power + dependence-aware statistics + trusted attestation
```

TP2 与 sticky DP2 是已注册的源码 vocabulary。`RunConfig` 只在准确 source-owned
`patched_two_gpu_v1` capability identity 与 runtime receipt claim 存在时接受 distributed
row；formal dispatch 随后深验 matching dynamic GPU proof。CPU `gloo` contract 或 caller
自己填写 receipt 都不能启用。Identity 必须准确绑定 rank、UUID、rendezvous、router、
clock、process-group、ownership 与 receipt。

当 command contract 要求时，generated JSON 使用相邻 `.sha256` sidecar。Loader 验证
canonical content，不只检查 filename。

## 核心 Tuning 与 Confirmation

Target-only 与 Static 渲染时不含 adaptation state 或 reserve。TTS 使用准确 TTS-Cal seal
冻结的 recipe authority 与 fixed-barrier publication；L0-naive 使用同一个 frozen recipe
authority 与 first-ready publication。Release-attested lane 会签署该 seal；trusted
single-operator lane 则直接消费 reducer-owned content identity，无需 external signer。历史
`TTS-paper-reconstruction` authority 只用于
diagnostic。E1/E2 中采用 `l0` search recipe 的是 LC-candidate，
只有准确 E2-sealed winner 才是 LightCone。共享 runtime code 不会合并 live candidate、
optimizer 或 evidence identity。Rendering 只是 pure planning；当前 release-attested lane 只有
Target-only 可以用于 claim。Trusted single-operator lane 可在准确 source/content/runtime、fresh
dynamic GPU qualification、capacity、terminal 与 coverage gate 通过后运行完整 empirical DAG；
TTS 与 L0-naive 还必须等待 disjoint TTS-Cal 的准确 content-sealed winner。External signer 只
决定能否提升为 release-level `MEASURED`。

Industrial E1/E2 command 由 reducer 控制。E1 消费 sealed E3a 与 TTS-Cal receipt，恰好
materialize 68 个 cell：四个 fixed role，加上 32 个 geometry、每个两个 LightCone
candidate。L0-naive 保持 mechanism anchor，不参与 candidate ranking。对 `g` 个 survivor
geometry，E2 首轮 materialize `n0 = 105g` 个 recipe（七种 optimizer × 三种 schedule ×
五个 learning rate），后三轮都要求 prior sealed survivor receipt，并满足
`n(k+1) = max(ceil(nk/4), 21)`；每轮另加四个 fixed anchor。它只把 LightCone candidate
与 fixed Static、frozen-TTS reference 比较，不会展开 eager sentinel matrix。
Confirmation planning 为
每个准确 family 激活四个 excluded pilot，在 confirmation data 可见前封存 `POWERED` 的
12--20 final block 或 `UNDERPOWERED`，并只 materialize sealed prefix。

Family power 不接受 bare score 或 hand-authored count：

```bash
lightcone-spec reduce-confirmation-family-power \
  --manifest artifacts/industrial/family-power-manifest.json \
  --output artifacts/industrial/family-power-plan.json
```

准确 `industrial_family_power_manifest` 绑定 registry artifact、pilot activation、hardware
envelope 与四个 pilot block。每个 pilot cell 都提供 terminal receipt、hardware receipt 与
budget observation。Reducer 保持 confirmation hidden，并拒绝 missing、extra、cross-family 或
unsafe evidence。

`validate-evidence-alias` 不会从 label 创造 scientific equivalence，也不会 round-trip
caller-authored receipt。它会将 bound raw manifest 对照当前 registry、hardware envelope、
GPU inventory 及其 source receipt、execution/load/config/split/sampling/model/budget artifact、
source terminal evidence 与 native terminal artifact 完整重放：

```bash
lightcone-spec validate-evidence-alias \
  --manifest artifacts/industrial/raw-alias-manifest.json \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/gpu-inventory.json \
  --hardware-envelope artifacts/industrial/hardware-envelope.json \
  --output artifacts/industrial/alias-reduction.json

lightcone-spec build-evidence-dependence-map \
  --direct-map artifacts/industrial/direct-dependence-map.json \
  --alias-reduction artifacts/industrial/alias-reduction.json \
  --output artifacts/industrial/evidence-dependence-map.json
```

Dependence map 只接受 first-party reduction artifact，让 shared control 在 covariance/
bootstrap 中保持一个 observation。Static 是 backend-specific；adaptive scientific role 永远不能 alias。
schema-v3 industrial analysis manifest 会列出 raw alias manifests；formal reducer 会逐一重放，
并拒绝与重放结果不同的 serialized map。旧 alias receipt、caller-authored reduction summary
以及旧 `--alias` flags 都会被拒绝。

每个 completed slice 都以 content-bound evidence 结束。只有 manifest/config/method/block/
data/trace 与全部 shard digest 验证通过，resume 才跳过它。中断 Parquet WAL segment 没有
terminal receipt，会被排除。Profiler output 带 `headline_evidence_forbidden=true`，不能合并
进 measured timing table。

## OnlineSPEC 工作流

OnlineSPEC 复用 model lock、controlled data 与 paired Static reference，但绝不复用核心
tuning/confirmation row。`verify-onlinespec-source` 检查外部 checkout 的准确 commit/tree、
clean status、audited key-file hash 与 license inventory；checkout 保持在仓库外。

Successive halving 在 OGD、optimistic OGD 与 ensemble 内独立运行。Selection 还绑定核心
reference load，但不能改变该 load 或影响核心 gate。Anchor selection 标注
`optimized_grid_claim=false`。OnlineSPEC 保持 TP1/DP1 且只能用于诊断；其 attestation
命令只能输出 non-authority decision。

## 状态、Resume 与 Exit Code

`analyze-preliminary-speed-study` 始终写入 `PRELIMINARY_DIAGNOSTIC_ONLY` 并退出
42。即使未来配置 trusted attester，它也会拒绝全部 attestation。
`attest-preliminary-speed-study` 与 `attest-onlinespec-study` 同样只输出确定性的
non-authority decision 并退出 42。Identity、schema、I/O、receipt 或 runtime 错误是普通
nonzero failure，不是科学结果。
Supplied legacy/static attestation 会被拒绝；只有 fresh root-authorized dynamic policy、typed
control、真实 terminal/qualification evidence 和 complete coverage 才可能生成 `MEASURED`。
缺其中任何一项时 `analyze-industrial` 继续退出 42。

Industrial registry 可能把目标 cell 声明为 `UNMEASURED`，但 declaration status 不等于
executable readiness。Native hook 与 source-owned runner 已存在；在尚未取得 fresh dynamic
control 或对应 GPU qualification proof 时，executor preflight 仍将相关角色解析为 `BLOCKED`。
TP2/DP2、DSpark、NEXTN 与 native ITL 必须分别消费 typed suite proof；EAGLE3 还要求 official
model/selector compatibility。历史 v2
artifact 仅用于 regression，不能作为 schema-v3 stage receipt。

### Target-output Reference 诊断

针对一个单独锁定的 Target-only server，`run-preliminary-target-reference` 记录每个 prompt 的 token
数量，以及完整有序 output-token-ID 数组带格式标识的 SHA-256；它绝不会用 decoded-text
digest 替代。Legacy collector 要求 `--target-reference`，任何 method/block 的轨迹不匹配
reference 都会被拒绝；仅 speculative method 彼此一致不能证明 exactness。无论 hardware
是否可用，该结果始终是 `PRELIMINARY_DIAGNOSTIC_ONLY`。

## Credential 与 Output Root

Artifact、model root、cache、trace、provider state、profile、selection、attestation 与
handoff file 必须位于 ignored external root。通过临时 `HF_TOKEN` 环境变量或其他安全渠道
传递模型权限。不要把 token、password、provider API key、private prompt、instance address
或 machine-specific path 放入 argument、manifest、log、文档或 Git。

## 单可信操作者正式实验链

### 结论边界

`formal_single_operator_v1` 是单一可信操作者从受控 clean checkout 采集并归约完整 v03
empirical campaign 的路径。它记录 canonical SHA-256 provenance，且只向新的不可覆盖目录
发布。该模式既不是对抗式 attestation，也不允许把无签名结果称为 formal `MEASURED`
evidence。缺少当前 release-root attestation 时，finalization 必须报告
`trusted_single_operator_empirical_no_signature`、`formal_measured=false` 与
`UNMEASURED`。

公开 supervisor 命令为：

| 命令 | 用途 |
|---|---|
| `formal-single-operator status` | 不读取 run evidence、不分配 GPU，只报告 21 个节点的源码 capability |
| `publish-v03-model-lock` | 不经网络解析，发布由代码所有的准确 19 模型/revision lock |
| `write-v03-e0-raw-source-path-inputs` | Canonical 发布 registry-accurate 七项 raw E0 path |
| `publish-v03-e0-source-authorities` | 扫描七项 source 并发布 typed authority |
| `write-v03-content-path-inputs` | Canonical 发布准确 model/data/E0/inventory 与 future-doctor path |
| `publish-v03-content-path-spec` | 派生 digest-free pre-doctor content path spec |
| `publish-stage-capacity` | 绑定 pre-doctor spec、run-root filesystem 与所需 free-byte threshold |
| `publish-trusted-content` | 深度绑定 typed content path spec 与 runtime observation，发布 BOUND bundle |
| `publish-preflight-workload` | 从该准确 BOUND bundle 派生 preflight workload authority |
| `publish-tts-cal-trainable-plan` | 从 BOUND content 派生固定 trusted TTS-Cal plan |
| `publish-e1-anchor-trainable-plan` | 从 BOUND content 派生固定 trusted E1 anchor plan |
| `publish-onlinespec-source-authority` | 只为 E0 绑定已审计的外部 OnlineSPEC checkout |
| `build-trusted-protocol-lock` | 从 path-bound source 构建 schema-v5 trusted ProtocolLock |
| `write-dag-driver-config` | 发布一份不可变、只含路径的 21 节点 driver config |
| `write-bootstrap-config` | 发布非 LLM supervisor config |
| `bootstrap-once` | 最多推进一个 durable controller/scheduler cycle |
| `bootstrap-run` | 阻塞至 DAG 完成或遇到真实 unresolved block |

较底层的 `materialize-node`、`prepare-run`、`execute-run`、`finalize-run` 与 reducer 命令
仍可用于 focused replay/debugging。不要把它们与正在运行的 bootstrap supervisor 混用，也
不要启动第二个 scheduler。

### 发布不可变输入

先在 Git 外创建 private source 目录、空的 E0-authority 目录和 private run root，然后严格按
下列顺序执行。该顺序先发布 pre-doctor path spec，从该 spec 派生 capacity，再发布 fresh
doctor report，最后才封存 runtime-`BOUND` content bundle；任何一步都不依赖后文才生成的
artifact。

若准确的 19 个 snapshot 尚未全部存在，先发布 code-owned lock，再让现有 model preparer
只获取这些不可变 revision。lock publisher 不接受 model、revision、digest、token 或网络输入；
审计已填充 cache 时给 `prepare-models` 加 `--offline`：

```bash
lightcone-spec formal-single-operator publish-v03-model-lock \
  --output /absolute/sources/formal-v03-model-lock.json

lightcone-spec prepare-models \
  --lockfile /absolute/sources/formal-v03-model-lock.json \
  --model-cache /absolute/models \
  --output /absolute/sources/formal-v03-prepared-model-roots.json
```

先收集 nonce-bound physical inventory，写入准确七项 E0 raw-source handoff，再派生七份
source authority：

```bash
lightcone-spec collect-gpu-inventory \
  --challenge-nonce-sha256 FRESH_64_LOWERCASE_HEX_NONCE_SHA256 \
  --receipt-output /absolute/sources/gpu-inventory-receipt.json \
  --output /absolute/sources/gpu-inventory.json

lightcone-spec formal-single-operator write-v03-e0-raw-source-path-inputs \
  --source AIME-2025=/absolute/cache/e0/aime25-test.jsonl \
  --source Alpaca=/absolute/cache/e0/alpaca-eval.json \
  --source Arena-Hard=/absolute/cache/e0/arena-hard-v2.0-question.jsonl \
  --source GSM8K=/absolute/cache/e0/gsm8k-test.jsonl \
  --source HumanEval=/absolute/cache/e0/humaneval.jsonl.gz \
  --source MBPP=/absolute/cache/e0/mbpp-sanitized.json \
  --source MT-Bench=/absolute/cache/e0/mt-bench-question.jsonl \
  --output /absolute/sources/e0-raw-source-path-inputs.json

lightcone-spec formal-single-operator publish-v03-e0-source-authorities \
  --inputs /absolute/sources/e0-raw-source-path-inputs.json \
  --output-directory /absolute/sources/e0-authorities
```

然后写入 registry-complete content handoff。以下 19 个 model key/path 由 source code 所有；
CLI 不接受 revision 或 digest override。六个 BurstGPT name 与七个 E0 name 也必须各出现准确
一次。此时 doctor output 必须尚不存在，但其 resolved parent 必须已存在。

```bash
lightcone-spec formal-single-operator write-v03-content-path-inputs \
  --repository-root /absolute/clean/lightcone-checkout \
  --model-snapshot gemma4_12b_dflash_e0=/absolute/models/models--deepseek-ai--dflash_gemma4_12b_block7/snapshots/7490ce60c7630107917fe558e2bbe3dcec6195cb \
  --model-snapshot gemma4_12b_dspark_e0=/absolute/models/models--deepseek-ai--dspark_gemma4_12b_block7/snapshots/2fa72e765eec2965fc4d86a8663ce6769eba6218 \
  --model-snapshot gemma4_12b_eagle3_e0=/absolute/models/models--deepseek-ai--eagle3_gemma4_12b_ttt7/snapshots/0bc24c312350910419cf371e54082f040d65cc82 \
  --model-snapshot gemma4_12b_target=/absolute/models/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --model-snapshot qwen35_122b_a10b_fp8_nextn=/absolute/models/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9 \
  --model-snapshot qwen36_35b_a3b_nextn=/absolute/models/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --model-snapshot qwen3_14b_dflash_e0=/absolute/models/models--deepseek-ai--dflash_qwen3_14b_block7/snapshots/ab0a8b28236654620bb41d64b336d00a14cb467f \
  --model-snapshot qwen3_14b_dspark_e0=/absolute/models/models--deepseek-ai--dspark_qwen3_14b_block7/snapshots/83207b416acf99f41c2184648923632fccea6dd0 \
  --model-snapshot qwen3_14b_eagle3_e0=/absolute/models/models--deepseek-ai--eagle3_qwen3_14b_ttt7/snapshots/d7ea05d0b0009badfff0df2dcaedf82cce0f74f8 \
  --model-snapshot qwen3_14b_target=/absolute/models/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18 \
  --model-snapshot qwen3_4b_dflash_e0=/absolute/models/models--deepseek-ai--dflash_qwen3_4b_block7/snapshots/02d530b7962ea1412beaf41a05c0b8e36d5f9b1d \
  --model-snapshot qwen3_4b_dspark_e0=/absolute/models/models--deepseek-ai--dspark_qwen3_4b_block7/snapshots/3457dff1417cb84927f6098a5fcb7cee85c934b7 \
  --model-snapshot qwen3_4b_eagle3_e0=/absolute/models/models--deepseek-ai--eagle3_qwen3_4b_ttt7/snapshots/b0b90fd15d052217c226be5e46d468d8d129e0cd \
  --model-snapshot qwen3_4b_target=/absolute/models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --model-snapshot qwen3_8b_dflash_core=/absolute/models/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4 \
  --model-snapshot qwen3_8b_dflash_e0=/absolute/models/models--deepseek-ai--dflash_qwen3_8b_block7/snapshots/9e44dbbb6cb68b0c943abf9c5fc3c17c00897cdf \
  --model-snapshot qwen3_8b_dspark_e0_core=/absolute/models/models--deepseek-ai--dspark_qwen3_8b_block7/snapshots/03326e5043815da1f81b109078b2889737c26017 \
  --model-snapshot qwen3_8b_eagle3_e0=/absolute/models/models--deepseek-ai--eagle3_qwen3_8b_ttt7/snapshots/f6485ba8d21e11942958617dbe7e71b467f38f38 \
  --model-snapshot qwen3_8b_target=/absolute/models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --livecodebench-raw /absolute/cache/livecodebench/test6.jsonl \
  --math500-raw /absolute/cache/math-500/test.jsonl \
  --burstgpt-asset BurstGPT_1.csv=/absolute/cache/burstgpt/BurstGPT_1.csv \
  --burstgpt-asset BurstGPT_2.csv=/absolute/cache/burstgpt/BurstGPT_2.csv \
  --burstgpt-asset BurstGPT_3.csv=/absolute/cache/burstgpt/BurstGPT_3.csv \
  --burstgpt-asset BurstGPT_without_fails_1.csv=/absolute/cache/burstgpt/BurstGPT_without_fails_1.csv \
  --burstgpt-asset BurstGPT_without_fails_2.csv=/absolute/cache/burstgpt/BurstGPT_without_fails_2.csv \
  --burstgpt-asset BurstGPT_without_fails_3.csv=/absolute/cache/burstgpt/BurstGPT_without_fails_3.csv \
  --e0-source-authority AIME-2025=/absolute/sources/e0-authorities/formal-v03-e0-aime_2025-source-authority.json \
  --e0-source-authority Alpaca=/absolute/sources/e0-authorities/formal-v03-e0-alpaca-source-authority.json \
  --e0-source-authority Arena-Hard=/absolute/sources/e0-authorities/formal-v03-e0-arena_hard-source-authority.json \
  --e0-source-authority GSM8K=/absolute/sources/e0-authorities/formal-v03-e0-gsm8k-source-authority.json \
  --e0-source-authority HumanEval=/absolute/sources/e0-authorities/formal-v03-e0-humaneval-source-authority.json \
  --e0-source-authority MBPP=/absolute/sources/e0-authorities/formal-v03-e0-mbpp-source-authority.json \
  --e0-source-authority MT-Bench=/absolute/sources/e0-authorities/formal-v03-e0-mt_bench-source-authority.json \
  --inventory /absolute/sources/gpu-inventory.json \
  --doctor-output /absolute/sources/doctor.json \
  --output /absolute/sources/v03-content-path-inputs.json

lightcone-spec formal-single-operator publish-v03-content-path-spec \
  --inputs /absolute/sources/v03-content-path-inputs.json \
  --output /absolute/sources/v03-content-path-spec.json
```

从该 pre-doctor spec 派生 capacity，再直接发布 fresh canonical doctor report。非 `PASS`
report 仍会保留并返回 42，但不能作为 `BOUND` content 使用。

```bash
lightcone-spec formal-single-operator publish-stage-capacity \
  --content-path-spec /absolute/sources/v03-content-path-spec.json \
  --run-root /absolute/run/formal-v03-study \
  --output /absolute/sources/v03-stage-capacity.json

lightcone-spec doctor \
  --project-root /absolute/clean/lightcone-checkout \
  --sglang-root /absolute/runtime/patched-sglang \
  --trusted-single-operator-capacity /absolute/sources/v03-stage-capacity.json \
  --output /absolute/sources/doctor.json
```

只有 doctor `PASS` 后，才能绑定 content，并派生所有 downstream workload、trainable plan 与
method source。Trusted TTS 和 E1 authority 命令直接消费同一个 BOUND bundle；
release-signed workload/receipt 参数不属于该 lane。

```bash
lightcone-spec formal-single-operator publish-trusted-content \
  --spec /absolute/sources/v03-content-path-spec.json \
  --output /absolute/sources/trusted-content.json

lightcone-spec formal-single-operator publish-preflight-workload \
  --content-source /absolute/sources/trusted-content.json \
  --output /absolute/sources/preflight-workload.json

lightcone-spec formal-single-operator publish-tts-cal-trainable-plan \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --output /absolute/sources/tts-trainable-plan.json

lightcone-spec formal-single-operator publish-e1-anchor-trainable-plan \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --output /absolute/sources/e1-trainable-plan.json

lightcone-spec publish-tts-calibration-tuning-window \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --output /absolute/sources/tts-tuning-window.json

lightcone-spec publish-tts-drafter-native-loss-source \
  --output /absolute/sources/tts-native-loss.json

lightcone-spec publish-tts-calibration-source-authority \
  --paper-pdf /absolute/sources/tts-paper.pdf \
  --paper-source /absolute/sources/tts-v2-source.tar.gz \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --tuning-window /absolute/sources/tts-tuning-window.json \
  --trainable-plan-authority /absolute/sources/tts-trainable-plan.json \
  --drafter-native-loss /absolute/sources/tts-native-loss.json \
  --output /absolute/sources/tts-calibration-authority.json

lightcone-spec publish-chronobelief-source-authority \
  --paper-pdf /absolute/sources/chronobelief-paper.pdf \
  --tex-source /absolute/sources/chronobelief-paper.tex \
  --output /absolute/sources/chronobelief-authority.json

lightcone-spec publish-e1-recipe-anchor-authority \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --trainable-plan-authority /absolute/sources/e1-trainable-plan.json \
  --output /absolute/sources/e1-recipe-anchor-authority.json

lightcone-spec formal-single-operator publish-onlinespec-source-authority \
  --checkout /absolute/sources/onlinespec-checkout \
  --audit /absolute/sources/onlinespec-source-audit.json \
  --output /absolute/sources/onlinespec-source-authority.json

lightcone-spec publish-formal-runtime-authority-manifest \
  --repository-root /absolute/clean/lightcone-checkout \
  --output /absolute/sources/runtime-authority.json

lightcone-spec formal-single-operator build-trusted-protocol-lock \
  --protocol-id lightcone-v03-study \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --runtime-authority-manifest /absolute/sources/runtime-authority.json \
  --tts-calibration-authority /absolute/sources/tts-calibration-authority.json \
  --chronobelief-authority /absolute/sources/chronobelief-authority.json \
  --e1-recipe-anchor-authority /absolute/sources/e1-recipe-anchor-authority.json \
  --output /absolute/sources/protocol-lock.json
```

所有 source/output path 必须 absolute 且 normalized。Canonical output 使用 atomic
no-replace publication，并在下一步前 deep-reopen。Model/data payload、provider state、
credential 与全部 run artifact 都留在 checkout 外。

### 配置并到达第一个 GPU 边界

先在 Git checkout 外创建新的 private run 与 prerequisite-catalog 目录，再发布两份只含路径
的 config：

```bash
lightcone-spec formal-single-operator write-dag-driver-config \
  --repository-root /absolute/clean/lightcone-checkout \
  --run-root /absolute/run/formal-v03-study \
  --protocol-lock /absolute/sources/protocol-lock.json \
  --content-source /absolute/sources/trusted-content.json \
  --runtime-authority-manifest /absolute/sources/runtime-authority.json \
  --inventory /absolute/sources/gpu-inventory.json \
  --doctor-report /absolute/sources/doctor.json \
  --preflight-workload-authority /absolute/sources/preflight-workload.json \
  --profiler-tool /absolute/tools/nsys \
  --profiler-tool /absolute/tools/ncu \
  --prerequisite-catalog /absolute/run/formal-v03-prerequisites \
  --output /absolute/run/formal-v03-study/driver-config.json

lightcone-spec formal-single-operator write-bootstrap-config \
  --driver-config /absolute/run/formal-v03-study/driver-config.json \
  --onlinespec-source-authority /absolute/sources/onlinespec-source-authority.json \
  --output /absolute/run/formal-v03-study/bootstrap-config.json
```

`status` 的 `readiness_scope=code_capability_only`。21/21 只证明每个节点的 materializer/
producer/mapper/executor/finalizer code 均可调用；不证明本次 run 的文件、模型、tool、GPU 或
upstream receipt 已就绪：

```bash
lightcone-spec formal-single-operator status
```

前文 capacity/doctor 步骤会把 live GPU 的 UUID/model/compute capability 与 path spec 中
准确 inventory 做 fresh join。该 empirical 模式只允许一个 16 GiB wave 加 15 GiB safety
margin，禁用 physical 与 E6/E0 auxiliary 的 automatic retry，且绝不授权 formal
`MEASURED` claim。

Cold start 刻意保留一个可检查边界。第一次 cycle materialize preflight；第二次规划准确
一条 compile、一条 exactness 与八条 interference cell，并把十个 attempt 及其双卡 physical
group 全部提交为 `PENDING`，此时没有 PID/PGID。第三次 cycle 才可能 atomic 标记该 group
为 `RUNNING` 并 spawn `setsid` child：

```bash
lightcone-spec formal-single-operator bootstrap-once \
  --config /absolute/run/formal-v03-study/bootstrap-config.json
lightcone-spec formal-single-operator bootstrap-once \
  --config /absolute/run/formal-v03-study/bootstrap-config.json
```

应先检查导出的 progress 与 host boundary，再明确调用第三次 cycle 或 `bootstrap-run`。

### 自动推进 21 节点

GPU host、滚动归档与 finalizer input 就绪后，非 LLM supervisor 会从同一 durable state 恢复：

```bash
lightcone-spec formal-single-operator bootstrap-run \
  --config /absolute/run/formal-v03-study/bootstrap-config.json
```

唯一的 `/absolute/run/formal-v03-study/operator.sqlite3` WAL 是 lifecycle authority；唯一的
`formal-dag-driver.lock` 阻止竞争 scheduler。Watchdog 按固定 30 秒 cadence reconcile
PID/PGID、heartbeat、GPU state、log growth、timeout/OOM、terminal publication 与 capacity；
progress table 在 run root 下 atomic 导出。退出码 `42` 表示保留了真实 block，`43` 表示整个
DAG 达到 controller completion，`0` 表示中间 cycle 发生状态变化。

21 个节点是以下顺序的准确展开：
`preflight -> E3a -> TTS-Cal -> E1 -> E2(r0..r3) ->
E4(screen,local,profiler) -> E3b(pilot,final) -> E1a ->
E5(pilot,final) -> E6(pilot,final) -> E0(tuning,pilot,final)`。任何时刻只 materialize 当前
node。Prerequisite/auxiliary catalog append-only 且 source-owned；未来结果不能解锁早期节点。

Preflight 仍准确包含十个 stage cell，同时 exactness execution 会运行六个 source-defined
native qualification suite。八个 fresh interference observation 使用 paired BCa 95% interval
归约；只有 goodput 与 native-p99-ITL 均满足 <=1% 的 pass gate，scheduler 才可切换为两个
独立单卡 headline worker，其他结果一律 isolated。

E6 只从冻结 Qwen target checkpoint 内置的 `mtp.*` component 发布准确两条 TP2 launch
descriptor，不下载也不接受 external drafter。E0 先发布 12 个 model/backend pre-probe
interface，再以 12 个 launch group 分别运行九个 task-native probe，形成准确 108 个
compatibility terminal。EAGLE3 task authority 只能 post-probe 产生。控制 E0 tuning/serving
materialization 的是最终 VALID/N/A bundle，而不是 caller 输入的 `V`。

E5 pilot reducer 会在 final unblinding 前封存每个选中的 backend/topology p99 family。该
family 中五个 paired method 的既有 cell 都接收同一份准确 11,000-offer extension pool，且
必须至少完成 10,000 个 request；不会因此新增 diagnostic 或 headline row。

### 滚动归档与恢复

远端 spool 较小时，应让本机非 LLM companion 与 DAG 同时运行。Endpoint file 只包含 routing
和固定 SSH/rsync tool path，不含 credential。Companion 每 30 秒轮询，只归档 sealed safe
boundary；逐文件验证 SHA-256、完成本地 full rehydrate 后，才请求远端 operator 驱逐准确的
inode-bound v03 文件。它不会 recursive delete directory、model cache 或旧 run：

```bash
/absolute/runtime/python -m lightcone_spec.orchestration.formal_rolling_archive_companion \
  --endpoint /absolute/archive-host/control/archive-endpoint.json \
  --remote-run-root /absolute/gpu-host/formal-v03-study \
  --local-results-root /absolute/archive-host/formal-v03-study/results \
  --state-root /absolute/archive-host/formal-v03-study/archive-state \
  --lock /absolute/archive-host/formal-v03-study/archive.lock \
  run
```

后续 reducer 若需要已驱逐 member，companion 只把该 member stream 回原路径，并执行远端
free-space gate、no-replace publication 与 per-file restart record。
`restore-all --order reverse` 可在手工 audit 前按 node 逆序恢复所有保留 archive；远端
scientific closure 也会消费其准确 rehydration catalog。

### 跨主机归档、关机与计费

Finalization 有三个不可逆阶段。该边界的 config 必须用 typed
`publish_remote_closure_config`、`publish_cross_host_ssh_endpoint` 与
`publish_cross_host_finalizer_config` source API 创建；不得手工填写 digest，也不得把 provider
token 写入 config。

A 阶段只在 21 个节点全部 `REDUCED` 后于 GPU host 运行。它停止 dispatch、恢复 catalog 中
的 rolling member、证明零 running attempt/writer、导出 progress，并 seal 一份按 digest
寻址的 whole-run payload。Receipt 发布后，远端 SQLite 永久只读：

```bash
/absolute/runtime/python \
  /absolute/checkout/scripts/formal_experiment_production_finalizer.py \
  remote-seal \
  --config /absolute/gpu-host/formal-v03-study/remote-closure-config.json
```

B/C 阶段只在 archive host 运行一次。`run` 拉取 closure/payload，校验本地 SHA manifest，
完整 rehydrate archive，取得 fresh 远端 zero-writer probe，并发布 pre-power composite。只有
随后才 journal 最多一次 AutoDL `power_off` mutation，要求 `code=="Success"`，再通过 status
与 list 两条 response 确认同一实例为 `shutdown`，关闭 provider boot interval，并发布
compute、reserved、allocated-billed、whole-instance-billed、archive、idle 与 wall-time
accounting：

```bash
/absolute/runtime/python \
  /absolute/checkout/scripts/formal_experiment_production_finalizer.py \
  run \
  --config /absolute/archive-host/formal-v03-study/local-finalizer-config.json
```

Provider token 只从本机 process environment 读取。Power intent 之后 crash 会保留为
indeterminate，不能再次发送 mutation；restart 会重开 journal 并继续 status/list confirmation。
即使 cross-host completion 成功，若缺少独立 release-root attestation，仍保持
`formal_measured=false`。

Pilot 前，`gpu-hours-pre` 只报告固定 cell 数、`duration_unmeasured` 与最小 pilot 集。
Pilot 后，`gpu-hours-post` 消费真实 single-operator run manifest 与 lifecycle timing，分别
报告 actual pilot、same-stratum projection 和 one-shot diagnostic；它不会把 registry
prefix 冒充 whole-study completion。
