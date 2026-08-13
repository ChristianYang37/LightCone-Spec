# 工程就绪与验收矩阵

[English](../en/engineering-readiness.md) · [首页](../../README_zh-CN.md)

本文把两份 industrial engineering spec 映射到当前 source boundary。它是 implementation/
readiness ledger，不是实验结果。GPU measurement、host address、credential、model/data
payload、provider state 与 result-derived selection 始终留在仓库外。

## 状态模型

三个 release 标签刻意保持严格：

| 标签 | 含义 | 不表示 |
|---|---|---|
| `CPU_READY` | Source、schema、deterministic reducer/scheduler、failure semantic 与 CPU/mock receipt contract 已存在，并通过适用的非 GPU gate。 | 不证明 CUDA/NCCL behavior、速度、显存容量或 formal execution。 |
| `GPU_SMOKE_READY` | 具名外部输入存在后，source path 可进行有界 device check。 | Smoke 不一定已运行；它绝不是 benchmark 或 `MEASURED` evidence。 |
| `MEASURED` | 已注册 run 取得 content-bound terminal evidence、immutable input、hardware/interference coverage 与 release-owned trusted-attester chain。 | Local mock、diagnostic signer、历史 v2 artifact 或正向算术都不能产生该标签。 |

`BLOCKED`、`UNRESOLVED`、`UNDERPOWERED`、`INVALIDATED` 与 `N/A` 仍是合法 terminal
disposition。Component 可以在 CPU contract 上 `CPU_READY`，而 formal GPU path 同时
`BLOCKED`；下表分别记录这两个事实。

## 当前 Release 边界

| Surface | Source/control 状态 | 下一项可执行 gate |
|---|---|---|
| 单主机任意 N GPU inventory 与 independent work | `CPU_READY`；same-host scheduler 对 1/2/4/8/16 GPU 有明确 contract coverage。 | 收集 content-bound inventory 并进行有界 host smoke。 |
| 多主机 independent cell | `CPU_READY`；fleet composition、host-local namespace、确定性均衡、有界 transport concurrency、failure isolation 与 unknown-outcome reconcile 已实现。 | 通过 pinned SSH route 运行 Python coordinator；目前没有 public coordinator CLI。 |
| 同主机 gang placement | Scheduler/state-machine contract 为 `CPU_READY`。 | 真实 TP2/DP2 launch 仍受下述 runtime/source capability 阻止。 |
| 跨主机 TP/DP collective | 固定以 `cross_host_collectives_unvalidated` 为 `BLOCKED`。 | 需要独立 source release 与真实 multi-host NCCL/rendezvous validation。 |
| TP1/DP1 Target-only | 准确 model/runtime/data lock 与 host preparation 完成后为 `GPU_SMOKE_READY`。 | Formal evidence 仍需要 release attester 与 registered DAG closure。 |
| TP1/DP1 DFlash Static/TTS/L0 | 底层 source lifecycle 为 `CPU_READY`；release execution 因 trust/device validation 为 `BLOCKED`。 | Pinned-tree GPU smoke、准确 terminal evidence 与 reviewed source-release attester anchor。 |
| DSpark adaptation | Proposal/reconstruction/selector/loss/scheduler-mode contract 为 `CPU_READY`；worker/CUDA candidate path 为 `BLOCKED`。 | 实现并 patch native worker update/publication path，再做 GPU validation。 |
| EAGLE/EAGLE3/NEXTN adaptation | 只有 target schema 与严格 compatibility guard；`BLOCKED`。 | 需要准确 pinned upstream training interface 与 semantic SGLang patch。 |
| TP2/DP2 adaptation/publication | 只有 CPU/gloo coordinator contract；formal source path 在 model load 前 `BLOCKED`。 | 需要 vocab-parallel loss、真实 all-rank producer/publication、native routing 与 same-host GPU/NCCL validation。 |
| Server-session reuse | Source-owned lifecycle 与 incremental evidence 为 `CPU_READY`，但本 release 不授权任何 live reuse。 | 使用 clean-process fallback；未来 release 还必须通过 device reset、durable close 与 whole-inventory continuity。 |
| Trusted attestation | Anchored external public-bundle loader 为 `CPU_READY`；禁止 private material。Source-release anchor 未设置。 | Provision/review source-owned anchor、live signer、nonce replay store 与 allowed hardware envelope。 |
| Formal industrial result | `UNMEASURED`。 | 必须先闭合全部 code、workload、hardware、evidence、power 与 attestation gate。 |

## Native-System Spec 验收

该矩阵按第一份 engineering spec 的主要编号要求逐项映射。“Accepted”只表示最小且真实的
source/CPU contract 已存在，绝不会把缺失 device path 升级为完成。

| Spec 项 | 当前 source 的验收 | Evidence 边界 / 剩余工作 |
|---|---|---|
| §0 audit、preserve、reconcile | 按用户后续指示，直接在既有 industrial worktree/branch 继续；unrelated work 保留，result artifact 仍在仓库外。 | Local handoff 记录 Git/test identity；public source 不含 host/result material。 |
| §0.1 隔离 branch/worktree | 原始“再创建隔离 branch/worktree”的要求已被用户后续“继续使用既有隔离的 `codex/lightcone-industrial-20260811` worktree”指示取代，因此未再创建 branch。 | G0 在该 branch 停止；`main` 保持不变，本文不暗示 push 或自动 merge。 |
| §0.2 科学与服务 invariant | Source/CPU contract 保留 Target-only、零 adaptation Static、仅 publication timing 不同的单一 TTS/L0 candidate、冻结历史 KV、fail-closed publication、显式 memory accounting 与 missing-as-missing evidence。 | CUDA exactness、服务行为与性能仍为 `UNMEASURED`；CPU acceptance 不能授权正向 GPU claim。 |
| §1 common backend contract | Common proposal evidence、trainable plan、candidate identity、reconstruction 与 strict backend payload validation 已接受。 | DFlash 有底层 adaptive path；DSpark 为 CPU-only；其他 backend 是 guarded target contract。 |
| §2 GPU-resident update/publication | Pinned DFlash source-contract 已接受，含唯一 candidate lifecycle、fixed-address publication、event 与准确 termination semantic。 | CUDA graph、stream priority、pointer 与 timing 需要 GPU smoke；不声明 device result。 |
| §3 Full/LoRA 与 native selector | Rank grid、zero functional initialization、native scope、准确 memory-plan accounting 与 56-cell DSpark selector 已接受。 | DSpark composite-head runtime execution 仍 blocked。 |
| §4 TP 与 replica-local DP | Topology、sharded/replicated ownership、sticky cohort、atomic publication、failure 与 CPU/gloo contract 已接受。 | TP2/DP2 runtime blocked；跨主机 collective 明确不支持。 |
| §5 HBM/cohort governance | 单一 ledger、least-feasible-rank admission、eviction order、bounded slab、quota、generation 与显式 cold offload 已接受。 | 在目标 GPU 验证真实 allocator/HBM receipt 与 pressure behavior。 |
| §6 production request/failure path | Bounded admission、terminal state、timeout/cancellation、clean failure evidence 与 fail-closed release preflight 已接受。 | 当前只有 Target-only 可进行 end-to-end GPU smoke；speculative/multi-rank gate 见上。 |
| §7 telemetry/profiling/useful work | Bounded writer/WAL durability、准确 coverage、budget observation、safety counter 与 isolated profiler contract 已接受。 | Device timing、power、HBM 与 collective telemetry 仍未测量。 |
| §8 CLI/identity/CI/release | Strict CLI input、content identity、semantic patch verification、package/public-tree gate 与 categorical blocked outcome 已接受。 | 最终 local gate transcript 写入不提交的 handoff，不写入 public doc。 |
| §9 declarative registry | 单一 registry/DAG、immutable cell、reducer-owned activation、准确 dependency receipt 与单一 executor/evidence path 已接受。 | Declaration 不授权 blocked method。 |
| §10 load/corpus | Immutable open/closed-loop trace contract、request accounting、synthetic label 与 external data lock 已接受。 | 准确 formal dataset revision 与 payload 是外部 gate。 |
| §11 registered E3a–E0 cell | Truthful registry、activation、budget、power 与 completion contract 已接受。 | Formal timing 未开始；DSpark、NEXTN、TP2/DP2、compile、download cell 保留具名 blocker。 |
| §12 analysis/power/evidence | 四个 excluded pilot、family-local 12–20-block power、Holm/BH、hierarchical/time-block bootstrap、evidence dependence 与 no missing-as-zero 已接受。 | Formal analysis 需要 attested GPU terminal input。 |
| §13 CPU-ready stage | Source implementation 为 `CPU_READY`；GPU-marked test 在本阶段只 collect、不执行。 | 准确 gate transcript 与 package identity 见最终 handoff。 |
| §14 no-card SSH preparation | Coordinator/worker protocol 为 `CPU_READY`。 | 实际 SSH routing、patched checkout、cache、data lock 与 public trust bundle 是外部且不提交。 |
| §15 provider 与 GPU smoke | 只有上表具名 bounded path 为 `GPU_SMOKE_READY`；本阶段未运行 smoke。 | 需要用户提供 host 与 immutable external input。 |
| §16 formal two-GPU schedule | `BLOCKED`；formal experiment 尚未开始。 | TP2/DP2 source capability、signer、smoke、interference 与全部 registered authority 必须先通过。 |
| D1 single-GPU baseline/tuning | Scheduling、activation、halving、Pareto 与 budget contract 为 `CPU_READY`；execution 为 `UNMEASURED`。 | 任何 E3a/E1/E2 run 前都需要 Target-only/Static device evidence、逐 host interference calibration、immutable input 与 formal authority。 |
| D2 mechanism/confirmation | Profiler isolation 与 family-level 四 pilot/12–20 block power contract 为 `CPU_READY`；execution 为 `UNMEASURED`。 | E4/E3b 与 DSpark confirmation 仍受 selected-recipe device evidence 阻止；DSpark 还缺 native worker path。 |
| D3 production single-GPU load | Registered trace、request accounting、lambda/p99、soak 与 paired-block contract 为 `CPU_READY`；未运行 production trace。 | 需要准确 replay input、同一 preregistered p99 anchor 至少 10,000 completion、terminal evidence 与 release attestation。 |
| D4 two-GPU topology/scale | Placement/state-machine vocabulary 为 `CPU_READY`，真实 TP2/DP2 与 E6 two-rank execution 为 `BLOCKED`。 | 需要后续 patched-runtime capability 与同主机 GPU/NCCL validation；跨主机 collective 继续 fail-closed。 |
| D5 breadth last | Frozen-recipe ordering、sequential staging、alias 与 OnlineSPEC isolation contract 已存在；execution 按 backend 为 `UNMEASURED`/`N/A`。 | E0 仍最后运行，并在 recipe/interface freeze 前 blocked；没有准确 model/backend interface 时保持 `N/A`。 |
| §17 return package | Code/docs/contract 是交付物；数值结果保持 `UNMEASURED`。 | Final SHA 与 local gate output 写入 local handoff；immutable evidence 留在仓库外。 |

## Optimization Spec 验收

该矩阵按第二份 optimization/pre-experiment spec 逐项映射。

| Spec 项 | 当前 source 的验收 | Evidence 边界 / 剩余工作 |
|---|---|---|
| §0 starting point 与 isolation | 已按后续指示协调为直接在既有 industrial branch 工作，并在 G0 稳定化后停止。 | 本文不暗示新建 branch、destructive cleanup、result import 或自动 merge。 |
| §0.1 当前 capability boundary | 已按当前 source 重新审计，而不是沿用 prompt 的历史 test count/identity；当前 release boundary 见上表。只有 TP1/DP1 Target-only 可进行 end-to-end bounded GPU smoke。 | Static/TTS/L0 release execution、DSpark、其他 adaptive backend 与 TP2/DP2 明确为 `BLOCKED`；不存在 trusted source-release anchor 或 `MEASURED` result。 |
| §0.2 科学与优化 invariant | Immutable cell、paired TTS/L0 semantic、独立 clean start、excluded pilot、fixed-duration trace、p99 minimum、显式 GPU-hour 与 artifact 外置已作为 source/CPU contract 强制。 | Session reuse 未获授权；准确 device 与 attestation evidence 存在前，全部 empirical value 保持 `UNMEASURED`。 |
| §1 first-class budget | 已接受：immutable per-cell budget 区分 startup、compile、warm-up、scored time、deadline、drain、reset、evidence、retry、special job、request/token/p99 及 compute/reserved/billed GPU time。 | Estimate 只是 planning artifact；actual delta 需要 terminal observation。 |
| §2 E1 activation/E2 halving | 已接受：E1 materialize 一个 130-cell slice；E2 为 sealed 25% retention，逐 round、保留 pair，并有 adversarial validation。 | 真实 survivor evidence 不在仓库中。 |
| §3 family power | 已接受：准确四个 excluded pilot 按 family 选择 sealed 12–20 final prefix 或 `UNDERPOWERED`。 | Pilot terminal evidence 存在前 confirmation 不可用。 |
| §4 evidence alias | 已接受：alias content-bound、验证 equivalence、保留 dependence；formal non-singleton use 在未重新计算时 fail closed。 | Alias 本身不创造新 observation 或 GPU saving claim。 |
| §5 clean server session | 已接受为 source lifecycle/audit contract 与 deterministic fresh-process fallback；其 receipt 在本 release 不能授权 reuse。 | 未来 live-reuse release 还必须取得完整 device reset/close/whole-inventory evidence。 |
| §6 native terminal/session evidence | Source-owned begin/reset/finalize identity、incremental durable step、terminal coverage 与 incomplete-session retention 已接受。 | Source-release signer 缺失；device reset semantic 仍需 smoke。 |
| §7 HTTP/writer/compile cache | Supported HTTP/1.1 lifecycle counter、每次 execution 一个 caller-owned pool、bounded durable writer、immutable cache base/private overlay 与 source-owned compile lifecycle 已接受。 | Unsupported HTTP path fail closed；compile GPU prewarm/finalization 必须在目标 host 验证。 |
| §8 arbitrary GPU-pool scheduler | 任意 same-host inventory 与明确 1/2/4/8/16 test 已接受；fleet composition 在不改变 placement semantic 的前提下增加 independent multi-host scaling。 | Cross-host collective blocked。 |
| §9 code/release gate | G0 要求全部适用 non-GPU check，并在未来任何 merge 决策前记录到 local handoff。 | GPU/integration/system test 留给 SSH smoke。 |
| §10 exact dry-run plan | Reducer、budget、assignment 与 bundle materialization 已实现。 | 真实 dry run 前必须在仓库外 freeze 准确 model/data/provider/inventory input。 |
| §11 no-card/provider preparation | Software path 已准备；credential、route、address、provider state 与 model payload 刻意缺席。 | SSH handoff 后由 external operator 执行。 |
| §12 GPU smoke/session equivalence | 未运行；顺序与 fail-closed expectation 如下。 | 不得重新标为 formal evidence。 |
| §13 optimized DAG execution | 尚未开始；formal status 为 `UNMEASURED`。 | 全部 source、smoke、budget、interference 与 trust gate 闭合后才执行。 |
| §13.1 E3a baseline/capacity | Activation、balancing、width-selection 与 capacity-envelope contract 为 `CPU_READY`；未运行 E3a cell。 | Materialize 130-cell E1 slice 前必须通过 Target-only/Static native evidence、准确 external input 与逐 host calibration。 |
| §13.2 E1/E2 tuning | Reducer-owned E1 activation 与保留 pair 的 quarter-retention E2 halving 为 `CPU_READY`；execution 为 `UNMEASURED`。 | 需要 device-capable DFlash、准确 receipt 与 formal authority；later-stage loser 保持不 materialize。 |
| §13.3 E4 mechanism/profiler | Isolation、compile-cache identity、private overlay 与 profiler accounting contract 为 `CPU_READY`；未运行 profiler 或 headline timing。 | 需要 frozen selected recipe 与 device validation；profiler timing 永远不能成为 headline evidence。 |
| §13.4 E3b pilot/power/confirmation | 四个 excluded pilot、family-local power、whole-block scheduling、legal alias 与逐 host interference limit 为 `CPU_READY`；confirmation 尚不可用。 | Pilot terminal evidence 必须在 unblinding 前 seal 12–20 block；不得因有利结果提前停止。 |
| §13.5 E1a/E5 production | Registry、trace、lambda/p99、paired arrival、soak 与 gang-scheduling contract 为 `CPU_READY`；DSpark 与 TP2/DP2 execution 为 `BLOCKED`。 | 需要 native DSpark path、frozen anchor、准确 replay evidence、同主机 capability 与 release attestation。 |
| §13.6 E6/E0 | Compatibility-first gate、frozen-recipe transfer/breadth order、准确 alias 与 OnlineSPEC isolation 已表达；execution 为 `UNMEASURED`/`N/A`。 | 缺少准确 model/backend interface 时保持 `N/A`；这些 gate 通过前不得开始 download 或 transfer run。 |
| §14 continuous budget/evidence control | Observation receipt、receipt-only resume、immutable attempt、cost accounting 与 completion sealing 已接受。 | Populated receipt 只能来自真实 external run。 |
| §15 completion/return | Public source 不含 result/secret；local handoff 记录 operational status。 | 未来 result package 必须继续 content-bound、evidence-first。 |

## SSH 到实验的 Runbook

Operator 提供 SSH 后按以下顺序执行。遇到首个 `BLOCKED`、identity mismatch、缺少 external
authority 或 smoke failure 时立即停止。

1. 在每台主机运行 `doctor`，绑定 repository commit、patched SGLang tree、driver/runtime/
   toolchain、storage、clock/power 与 background state。
2. 在每台主机分别创建 nonce-bound `GpuInventory` 与该主机自己的 serial/calibrated
   `InterferenceEnvelope`；即使机器型号相同也不能复制 envelope。
3. 用成对重复的 `--inventory`/`--interference-envelope` 组装 fleet，确认 host ID/GPU UUID
   唯一，并检查 port、cache、evidence 与 manifest namespace 在每台 host 内无冲突；不同 host
   可以重复 literal value。
4. 在 disposable checkout 重跑 patch apply/tree/file-list/compile/focused-test/reverse
   verification 及 runtime-manifest check。
5. 运行有界 device、terminal、session、reset、compile-cache、HTTP、writer 与 evidence
   smoke。本 release 仍必须选择 fresh-process fallback；完整 CPU audit receipt 也不能视为
   reuse authority。
6. 先运行 single-GPU Target-only smoke。只有准确 capability 存在时才把 Static/DFlash TTS/L0
   作为 diagnostic 运行；没有 reviewed source-release signer 时，它们对 formal DAG 仍为
   non-formal 且 `BLOCKED`。
7. 检查 same-host TP2/DP2 preflight 与 diagnostic contract。当前 source 必须在 model load 前
   拒绝真实 launch；后续 source capability 与 same-host NCCL smoke 存在前不得进行 formal
   two-GPU run。
8. 通过 SSH 运行 Python fleet coordinator。Remote worker 是
   `execute-dispatch-wave --host-request-stdin`；保留 completed host receipt，通过独立的同主机
   receipt/evidence fetch reconcile 每个 unknown outcome，只对 terminal-negative work 在原
   host 上创建新 attempt。
9. 每台 host 分别做 interference calibration。只有准确通过的 cardinality 才能启用并发
   headline work；profiler、download、compile 与 shared-I/O domain 继续 exclusive。
10. 加载 externally anchored public attester bundle、nonce replay store、immutable
    model/data/trace lock 与 registered hardware envelope。Private key 绝不进入 repository、
    argv、artifact 或日志。
11. 只有前述 gate 全部满足后才 materialize/execute formal DAG；否则封存准确
    `BLOCKED`/`UNMEASURED` disposition，并保留 interrupted evidence，禁止其进入 analysis。

## 完成规则

目标 branch 在 handoff SHA 上 clean、`main` 按指示保持不变、全部适用
non-GPU/package/public-tree gate 通过，且每个未运行 device/formal surface 都有明确 contract 与
blocker 时，G0 才完成。Empirical phase 只有在 immutable external evidence 闭合 registered
DAG 后才完成。
在此之前，无论 code coverage 或 smoke readiness 如何，真实 project-level result 都是
`UNMEASURED`。
