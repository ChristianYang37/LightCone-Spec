# LightCone-Spec

[English](README.md) · [中文文档](docs/zh-CN/architecture.md) · [许可证](LICENSE)

LightCone-Spec 是一个以证据为先的 speculative decoding 在线 drafter adaptation
研究框架。它严格区分 runtime 工程与实证结论：CPU 合同可以验证身份、持久性和
fail-closed 行为，但只有已注册且 attested 的 GPU 证据才能证明速度或容量。

> 当前为 Alpha 软件。全部 formal industrial GPU 结果在新的 clean source/patch/protocol/
> registry identity 通过注册 GPU gate 前均为 `UNMEASURED`。Release-level `MEASURED`
> 结论只能由 fresh root-signed control 授权，并绑定准确 external input、inventory、hardware
> 与 native terminal evidence；trusted single-operator 路径可以收集 unsigned empirical
> evidence，但不能越过该结论边界。历史 v2 artifact 仅可用于
> regression/debugging；它们不能证明 schema-v3 协议或任何新性能结论。

<!-- RESULT_TRUTH_GATE_BEGIN -->
## 结果真实性 gate

以下字段明确分隔证据状态、当前 release 可执行状态、主机运营状态与源码身份。GPU 主机关机
不代表实验完成，历史 measurement 也不会因此升级为 formal result。

| Truth field | 值 | 范围 |
|---|---|---|
| `formal_industrial_gpu_evidence` | `UNMEASURED` | 当前 schema-v3 formal 证据；没有发布任何 industrial 性能结论。 |
| `formal_industrial_execution` | `BLOCKED_PENDING_QUALIFICATION` | release-attested lane 已配置 public trust root，但新的 root-authorized source、model、hardware、control 与 mandatory-preflight receipt 尚未生成，因此该 lane 在 mutation 前 fail closed。 |
| `historical_snapshot_evidence` | `PRELIMINARY_NON_FORMAL` | 下方数字 snapshot 仅为历史工程证据。 |
| `historical_snapshot_host_at_archive` | `POWERED_OFF_NOT_RELEASED` | 归档时的运营状态；实例已经关机，但没有释放或删除。 |
| `current_sglang_upstream_commit` | `3312645a307453893a00778592f105581e3d1c3d` | 当前 patch manifest 锁定的完整 Git commit。 |
| `current_patched_sglang_tree` | `bb6371242e82592d1b8a2f5f4ba6d0630d8365cb` | 应用当前 patch series 后预期的完整 Git tree。 |
| `current_patch_payload_sha256` | `0c4db4f8798645c0ba65e97031030fb5e891d15f63cd75105fc1e1656c1a2874` | 最新 semantic mail-patch 原始字节的 SHA-256。 |
| `current_patch_manifest_sha256` | `ff2fbc43c89e9f476a3fcf5690ba4287c00d434ba4a0f8c1a5d10d77bf79e716` | 当前 canonical patch-manifest JSON 的 SHA-256。 |
| `historical_main_code_prefix` | `0db2ff4` | 仅绑定 preliminary snapshot 的短 code prefix。 |
| `historical_patched_tree_prefix` | `e795ecc` | 仅绑定 preliminary snapshot 的短 tree prefix。 |

当前 commit、tree、patch payload 与 manifest digest 是四个不同的身份域。两个七字符历史
prefix 不是当前 release identity，不能满足任何 formal identity gate。
<!-- RESULT_TRUTH_GATE_END -->

执行权限与结论权限属于两条独立 lane。Trusted `formal_single_operator_v1` 在准确
source/content/runtime、fresh GPU qualification、capacity、terminal 与 coverage gate 全部通过后，
无需 external signer 即可运行完整 empirical DAG。其 unsigned closure 始终为
`trusted_single_operator_empirical_no_signature`、`formal_measured=false` 与 `UNMEASURED`。
Release-root signature 只用于把合格 evidence 提升为 `MEASURED`，不是这条 trusted lane 的
empirical execution 前置条件。

## 研究范围

正式对比包含五种科学角色。Recipe authority 与 publication policy 是正交身份；共享
runtime 实现不代表配置、live candidate、optimizer state 或 evidence 相同。

| 科学角色 | Runtime method | Recipe authority | 发布 |
|---|---|---|---|
| Target-only | `target_only` | 结构性零 adaptation | 原生 target decoding |
| Static | `static` | 结构性零 adaptation | 原生 speculative decoding |
| TTS | `tts` | 冻结且绑定一手来源的 TTS recipe | 固定 TTS barrier |
| L0-naive | `l0` | 同一个冻结 TTS recipe authority | first-ready safe boundary |
| LightCone | `l0` | 一个 tuning-sealed E2 winner | first-ready safe boundary |

使用已注册 E1/E2 search recipe 的 `l0` run 是 **LC-candidate**，不是 LightCone。只有
准确 sealed E2 final-recipe receipt 才能物化或命名 LightCone。Target-only 与 Static 在
结构上不分配 adaptation、optimizer、gradient、candidate 或 adaptation-telemetry state。

历史 `TTS-paper-reconstruction` baseline 继续是 diagnostic 且 `BLOCKED`：截至 2026-08-15
的 primary-source 审计未找到作者官方代码/config，论文也没有固定全部数值字段。因此 formal
路径使用独立的预注册 TTS-Cal authority：固定 Adam、一次 optimization step、
`(beta1=.9, beta2=.999, epsilon=1e-8)`、zero weight decay、无 clipping、全 drafter/
latest-update-round-only update、digest-bound DFlash position-weighted target-to-draft KL、逐 request
reset、side stream、learning-rate grid `1e-7, 3e-7, ..., 1e-3` 与 stride
`{1,5,10,15,20,30,40,50}`。Code-owned post-master split 覆盖完整 tuning domain 并排除准确
四个 pilot；trusted single-operator reducer 以 content identity 封存 winner，无需 external
signer。Release-attested lane 会额外签署同一 seal。TTS 与 L0-naive 绝不能继承 E1/E2 winner、
schema default 或历史 AdamW configuration；TTS publication 独立固定为论文的
synchronization barrier。
[Provenance authority](manifests/provenance/tts_recipe_authority_v1.json) 绑定 arXiv
`2605.09329v2`、PDF SHA-256
`7688b05bab7696f4a47a5987f2fcad13d46f1d84cec9f90caf661fb397f3ee20` 与 source SHA-256
`22c549c0297fc0a2a71af002c3721f71ddfd06d86bc46b2f41592bd6748afe59`。

目标 industrial registry 先覆盖 DFlash 与 DSpark，再进入 production load、多 GPU
topology、原生 NEXTN preflight 与 breadth template。EAGLE/EAGLE3 仍是带严格兼容性
guard 的目标 backend contract；注册并不表示当前 release 可以执行。OnlineSPEC 是重要
对比，但使用独立 tuning、证据、attestation 与分析，不能选择或改变核心 gate。

当前固定 SGLang patch 已包含 DFlash、DSpark、NEXTN 与 official-selector-compatible EAGLE3
的 source implementation，并在正式矩阵注册的范围内实现 TP1、双 rank TP2 publication 和
sticky 双 replica DP2 isolation。DFlash 执行注册的 constant、inverse-square-root、有限
horizon cosine schedule 与非负 logical publication delay；DSpark 绑定真实 predecessor、
W1/W2 与 confidence head；NEXTN 绑定 MTP teacher/interface 与 TP2 shard authority。Patch
也提供准确 begin/reset/finalize terminal evidence，host adapter 会重验内容。但这些只是实现
状态，不是 GPU 结果。Release-attested lane 尚未配置 trusted hardware signer，也没有 fresh
dynamic GPU qualification receipt，因此 Static/TTS/L0-naive/LightCone 以及所有 adaptive
backend/topology 仍不可用于 release 结论，并在该 lane 的 preflight、任何 mutation 之前 fail
closed。这个 release-claim blocker 不会阻止 trusted single-operator empirical lane；后者使用
source-owned unsigned authority，但仍必须通过准确 dynamic GPU、capacity、terminal 与 coverage
proof。Generic EAGLE 不受支持；没有适用 compatibility decision 的 EAGLE3 组合保持 N/A 或
`BLOCKED`。
请求 quota-shadow row 时会记录准确 identity 与 quota；不支持的 acquisition 会被拒绝，不会
伪造 teacher row。

## Runtime 合同

- Schema v3 在模型分配前拒绝未知或已退役字段；
- 公共 backend envelope 绑定 adapter-free logits、采样时使用的准确 proposal
  distribution、semantic mask、target teacher row、真实 sampled predecessor、cohort
  身份、source version 与 backend-native payload；
- 目标 backend contract 要求 validator 在不重复施加 adapter 的前提下重建可微 proposal。
  固定 patch 已实现 DFlash、DSpark、NEXTN 与 compatibility-authorized EAGLE3 surface，但每个
  准确 backend/topology pair 仍须通过其 named dynamic GPU suite 与 durable external-control
  proof 才能解锁；generic EAGLE 未实现；
- Adaptation 只有 `full` 与 `lora`。Layer scope 为 `last1`、`last3`、`last5`、`all`；
  LoRA rank 为 1、2、4、8、16、32、64，且 `alpha/r=1`；
- DSpark 另注册三种 hybrid scope：最后 1、3 或 5 层 backbone，加原生 W1、W2 与
  acceptance/confidence 参数。E1a 恰有 56 个 adaptive configuration：32 个 layer-only，
  24 个 hybrid；
- 历史 drafter KV 冻结并带版本。发布 candidate 只影响未来 KV；重建旧 KV 或对其求导
  会定义另一种方法。
- Candidate equality 只用于受控 mechanism replay，且要求 source-state 与 proposal-evidence
  digest 完全相同；TTS 与 L0-naive 的 publication history 分叉后，live run 可以继续分叉。

Rejection sampler 在 proposal token 被拒绝后仍从归一化 $(p-q)_+$ 采样。这个正部
rejection distribution 属于采样数学，不是 adaptation mode 或配置 alias。

## 工业级执行

声明式依赖顺序为：

```text
preflight -> E3a -> TTS-Cal -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0
```

每个 cell 都绑定科学轴、seed、逻辑 rank slot、port、cache/evidence root、资源隔离与真实
初始状态。Stage receipt 在下游运行前封存 runtime/split digest、准确 dependency receipt、
activation/disposition artifact 与 locked output。唯一的确定性同主机 GPU-pool scheduler
会把逻辑声明映射到内容绑定的物理 UUID，并支持 1、2、4、8、16 或更多 GPU。它会串行
独占 headline/profile/download/compile 工作；只有准确注册的 interference-envelope rule
允许时，互相独立的工作才可并行。

资源隔离不等于 terminal authority。COMPILE 与 DOWNLOAD 使用 typed first-party assignment、
immutable terminal 与 atomic no-replace result pointer；缺少 dynamic control 或 raw coverage
不完整会阻止 activation。

TTS-Cal 有 288 个不相交 tuning-only row 并封存一个 numeric TTS recipe。E1 随后恰好
materialize 68 行：四个 fixed role 加 32 个 LightCone geometry 各自的两个 optimizer anchor。
E2 为每个 survivor geometry 注册 105 个 recipe，并运行四轮、family floor 为 21、每轮加四个
fixed anchor 的 successive halving。E4 为 48 screen + 96 local-factorial + 3 profiler 行；
E3b 为 `480B`；E1a 为 116 行；E5 为 `450B+264`；E6 为只覆盖两个 target model 的
`2+60B`。E0 先签署 108 个 compatibility decision，只为其中 `V` 个 valid combination
materialize `16VB` 行；不再展开 blocked sentinel matrix。每个 family 的四个 pilot 会在
confirmation 前锁定 12--20 final prefix 或 `UNDERPOWERED`。Per-cell budget、物理 assignment、
terminal evidence 与 observed-versus-registered GPU-time receipt 全部内容绑定。固定 patch
现已包含 first-party all-reset producer，可生成 source-owned capability、initial-state 与 reset
receipt，覆盖 drain、KV/prefix、RNG/counter、scheduler/telemetry、adaptation state、allocator/HBM
和 completion event。只有准确 runtime identity 通过注册的 device reset qualification 后才
允许 GPU reuse。在支持的 single-tokenizer HTTP/1.1 uvicorn 路径上，
HTTP process 会在真实 protocol `connection_made`/`connection_lost` lifecycle 上累计 process、
generation、created、closed、current；scheduler request counter 与 client field 都不能替代。
Granian HTTP/2 与 multiple-tokenizer HTTP-process 路径会在生成该 capability 前 fail closed。
若 reset/trace/close receipt 或 dynamic proof 缺失，live GPU reuse 会 fail closed 并回退到
每条 trace 一个 clean process 与 HTTP pool。

`RunConfig` 只支持单机最多两 rank 的 TP1/DP1、TP2/DP1 与 sticky-replica TP1/DP2。TP2
要求 all-rank prepare/decision/application/receipt 与 zero-partial abort；DP2 保持 replica-local
state 且禁止跨 replica gradient averaging。CPU `gloo` harness 仍只验证 state machine，不能
attest NCCL、CUDA stream、graph boundary、device copy、throughput 或双卡正确性；只有 fresh
root-authorized mode-specific GPU proof 才能启用对应 topology。

## 部署规模与就绪状态

资源池 control plane 有两个相互独立的扩展维度。单台主机可暴露任意数量 GPU。
`GpuFleetScheduler` 只负责 host affinity 与 serial partition；它选定 host 后，每个 child
placement 都继续由唯一的 `GpuPoolScheduler` 按该主机 topology 与 interference envelope
完成。`GpuFleetInventory` 还可组合多台这样的主机，在主机间分发 independent cell 或完整
confirmation block；这不会把 fleet 变成一个 model-parallel rank group。所有 TP/DP gang
必须完整留在一台主机内；若 placement 需要跨主机 collective，会固定以
`cross_host_collectives_unvalidated` 拒绝。

每台主机都保留自己的 content-bound inventory digest、physical GPU UUID、interference
envelope、port、cache namespace、evidence namespace 与 host-local materialization manifest。
Remote wave 使用 SSH agent、固定 known-hosts file 和 stdin 上的 canonical JSON；routing 数据与
credential 只存在于 coordinator 本地，绝不复制到 artifact 或日志。主机失败时保留已完成
receipt，in-flight work 仍属于原主机。一旦 request 可能已到达 worker，权威 response 丢失必须
标记为 `REMOTE_OUTCOME_UNKNOWN`；只有通过准确原始 destination、port 与 known-host-key
authority 独立、内容寻址地取回并验证该 attempt 的 receipt/evidence 后才能 reconcile，不能
直接重试。Endpoint value、host-key bytes 与 credential 都不会持久化到 request、receipt、
evidence 或 log。只有 terminal-negative result 才能创建新的 receipt-bound attempt，fleet
transport concurrency 也必须显式有界。

就绪标签刻意不表示性能结论：

- `CPU_READY` 表示非 GPU schema、scheduling、identity、failure 与 receipt contract 已通过
  CPU/mock gate；
- `GPU_SMOKE_READY` 表示准确 device path 与外部输入已组装好，可进行有界 smoke；它不表示
  smoke 已通过，也不表示 formal cell 已获授权；
- `MEASURED` 必须取得完整注册 GPU evidence 与 release-owned attestation chain。前两个标签
  都不得报告为 `MEASURED`。

## 显存、Trace 与证据

HBM admission 由最不可行 rank 决定。账本分别计费 model/KV、FP32 master、gradient、
实际 optimizer moment、candidate/staging、activation、graph buffer 与 telemetry。显存压力
处理首先保留不可变与活跃 correctness 状态，只允许驱逐原生 inactive prefix，随后取消
pending adaptation，并 queue 或拒绝新工作；绝不静默把 Full 改成 LoRA。

Cohort state 使用固定大小 slab、tenant quota、replica 身份、generation counter 与确定性
回收。可选 cold offload 必须显式启用、计时，并且只适用于 inactive cohort；它不是自动
OOM 逃生路径。

Open-loop production trace 不可变且与 method 无关。Closed-loop run 绑定同一个不可变最大
request pool 与每个 client 的顺序；由于更快的方法会更早 replenishment，实际 offer time 与
连续 client prefix 必须逐项记录。固定 arrival window 结束前耗尽 pool 的 run 不可用于结论。
Synthetic Poisson 与 immediate burst 必须标为 synthetic；BurstGPT 命名需要绑定外部身份。

Evidence 先进入有界内存 queue，再写入 durable、process-unique 的 Parquet WAL segment。
Flush/checkpoint 状态、重复身份、backpressure 与任何显式 drop 都有计数。只有 coverage 与
文件系统 durability 检查通过后才发布最终 Parquet shard，随后以不可覆盖方式发布内容绑定
completion receipt。中断 WAL 保持可审计，但没有 receipt 就不能进入分析。

## 安装与快速开始

仅开发框架：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

在准确 pin 上创建一次性 SGLang checkout：

```bash
git clone https://github.com/sgl-project/sglang.git /path/to/sglang
git -C /path/to/sglang checkout --detach \
  3312645a307453893a00778592f105581e3d1c3d
patches/sglang/apply.sh /path/to/sglang
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream --compile-only
```

当前 schema-v3 SGLang series 在任何 GPU run 成为合格证据前，必须通过 clean-pin、
patch-digest、expected-tree、compile/test 与 reverse-removal gate。文档不能替代该 gate。

先用逻辑 rank slot 生成 registry；物理 device 身份只来自另行锁定的 inventory 与 dispatch
assignment：

```bash
lightcone-spec build-industrial-registry \
  --logical-gpu-slot logical-rank-slot-0 logical-rank-slot-1 \
  --cache-root runtime-cache/industrial \
  --evidence-root artifacts/industrial \
  --output artifacts/industrial/registry.json

lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --budget-load-binding artifacts/industrial/load-cell-000.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --activation-plan artifacts/industrial/stage-activation-manifest.json \
  --output artifacts/industrial/dispatch.json
```

Fleet 使用时，必须在每台主机分别收集 inventory 与 calibration，再按相同的 repeated-argument
顺序组装 content-bound host pair：

```bash
lightcone-spec assemble-gpu-fleet-inventory \
  --inventory artifacts/host-a/inventory.json \
  --interference-envelope artifacts/host-a/interference.json \
  --inventory artifacts/host-b/inventory.json \
  --interference-envelope artifacts/host-b/interference.json \
  --output artifacts/fleet/inventory.json
```

Remote coordinator 当前只提供 Python library API。唯一 remote worker CLI 入口是
`lightcone-spec execute-dispatch-wave --host-request-stdin`；它从 stdin 接收 canonical
coordination request，不是 interactive operator command。

该命令只生成目标 declaration，不会绕过 executor 的 native-evidence preflight，也不会让
speculative 或 multi-rank cell 变得可运行。

完成一个 stage 后，封存其内容绑定 output，再把 receipt 传回 planner。准确参数以
`lightcone-spec COMMAND --help` 为准；CLI 不会把 registry 隐式当作 shell script 执行。
Model lock、证据、trace、credential、provider state 与 selection 必须放在忽略的外部 root，
不得提交。

## Trusted v03 复现

已实现的 `formal-single-operator` 路径用一个 SQLite WAL、一个受 `flock` 保护的 scheduler
和 source-owned physical producer 运行不可变的 21 节点 DAG。它是可信操作者 empirical
路径，不能替代 release attestation chain。缺少当前 root-authorized attestation 时，即使全部
完成，结果也只能标记为 `trusted_single_operator_empirical_no_signature` 并保持
`UNMEASURED`；绝不能报告成 `MEASURED`。

发布 capacity 前，path-bound content replay authority 会对每个唯一 physical snapshot
逐字节读取一次全部模型内容。后续 ceremony 不再重读 model payload，而是 replay 完整
namespace、VFS object metadata、symlink target/blob identity 与 filesystem/mount identity。
它通过 inode、ctime、metadata 或 namespace drift 捕获正常 VFS mutation；但不声称抵抗能够
在不产生可观察 metadata 变化时静默改写字节的特权 kernel 或底层 storage。Model
prepare/load 会绑定 receipt 的准确 byte hash，并在有界 safetensors-header 检查或实际 load
前立即重新验证 metadata/namespace identity；不接受 caller-supplied digest。

发布 source authority、BOUND trusted-content bundle、preflight workload 与 schema-v5
ProtocolLock 后，写入只含路径的 DAG/bootstrap config。先在不分配 GPU 的情况下检查 code
capability，再运行非 LLM supervisor：

```bash
lightcone-spec formal-single-operator status
lightcone-spec formal-single-operator bootstrap-run \
  --config /absolute/results/formal-v03-run/bootstrap-config.json
```

`status` 报告 21 个节点的 code capability，而不是 run-specific GPU readiness。Cold start
准确执行两次 `bootstrap-once` 后，会得到十个 preflight attempt 均为 `PENDING` 的 queue；
第三次才是第一个 GPU launch boundary。后续 headline 使用两条独立单卡 worker 还是
isolated scheduling，只能由
fresh paired interference evidence 决定。

E6 只从两个冻结 target checkpoint 内置的 `mtp.*` component 派生准确两条 TP2 launch，并
禁止 external draft model。E0 先发布准确 12 个 model/backend pre-probe interface，再运行
108 个 model/backend/task compatibility probe；task-specific EAGLE3 authority 只能在真实
post-probe evidence 成功后发布。滚动归档与三段式跨主机 closure 在限制 GPU host spool 的
同时保留完整 rehydrate 的本地 archive。完整绝对路径命令及关机边界见
[可信工作流 CLI](docs/zh-CN/cli.md#%E5%8D%95%E5%8F%AF%E4%BF%A1%E6%93%8D%E4%BD%9C%E8%80%85%E6%AD%A3%E5%BC%8F%E5%AE%9E%E9%AA%8C%E9%93%BE)。

## 统计与结论

四个被排除的 pilot block 仅用于估计方差。Power plan 会在下游 unblinding 前把 final
block 数固定在 12--20；若两个 contrast 都无法对 3% 最小效应达到 80% power，则记录
`UNDERPOWERED`。Primary LightCone--TTS 与 LightCone--Static hypothesis 使用 Holm
family-wise correction；预注册 secondary decomposition 是 L0-naive--TTS 与
LightCone--L0-naive。已注册的报告目标包含 method-by-model、method-by-context 与
method-by-load interaction，且不预设其正负。当前 `CrossFamilyInteractionReducerArtifact`
只是内容绑定、structural、non-formal 的 `UNRESOLVED` contract；它不能证明 GPU coverage
已完成、interval 有效或任何 formal interaction。只有后续对 attested final evidence 执行
已注册 statistical reduction 才可能形成这类结论。其他 secondary breadth family 使用
Benjamini--Hochberg FDR。长上下文/request 数据
使用 block 后 request 的 hierarchical bootstrap；production trace 使用 time-block
bootstrap。P99 latency 只有达到注册的最小完成数才合格。

每个 system point 都同时报告 throughput/goodput、TTFT/ITL、completion/error accounting、
target work、update/publication timing、HBM、每个 output token 的能耗，以及锁定的
power/clock/thermal envelope。缺失值不能替换为零。

只有 content-bound GPU attestation 同时覆盖 registry/manifest、selection、模型与数据
revision、runtime capability 与 patched tree、hardware/power report、trace 身份及精确
Parquet 输入时，gate 才可能返回 `PASS`。本地 mock、历史 v2 evidence 或正向算术仍然是
`UNMEASURED`。本 release 只携带 offline Ed25519 public trust root，不携带 private key 或
预授权硬件身份。必须由新的 challenge-bound deployment policy 与 source-owned control
attestation 绑定实测 inventory 和新的 immutable source identity。Caller-authored
doctor/attestation JSON 会被拒绝；在这条 qualification chain 与 mandatory preflight
通过前，任何 analyzer 都不能产生新的 `MEASURED` GPU outcome。

对于 preliminary exactness 诊断，`run-preliminary-target-reference` 可以捕获一份锁定的
Target-only greedy reference；每个 prompt 的 output 是完整有序 token-ID 轨迹带格式标识的
hash。Legacy collector 要求每种 method 的每个 block 都匹配该 reference；解码文本一致或
仅 speculative method 彼此一致都不充分。该 reference 始终为
`PRELIMINARY_DIAGNOSTIC_ONLY`：它不能让 Static/TTS/L0-naive/LightCone 获得 formal execution authority，
也不能替代 industrial authority。

## 历史 preliminary 机制 snapshot

以下 snapshot 来自较早的 main-line 实现，仅保留为 preliminary 工程证据。它不会改变
上文当前 industrial 协议的 formal truth：schema-v3 Stage B 仍为 `UNMEASURED` 且
`BLOCKED`，这些数字不能激活 cell、选择配置或通过 release gate。

单张 RTX PRO 6000 Blackwell（96 GB）运行 Qwen3-8B + DFlash-b16，使用 16 条重复型
受控 prompt、并发 8 和 40,928-token 安全 context 上限。每条 diagnostic path 均在单个 timing block
中生成 654,042 tokens：

| 方法 | Decode goodput | 相对 Static | p99 ITL | Peak HBM |
|---|---:|---:|---:|---:|
| Static | 1,342.0 tok/s | 1.00x | 45.04 ms | 90.36 GiB |
| Matched-recipe fixed-barrier diagnostic | 2,497.9 tok/s | +86.13% | 45.94 ms | 90.52 GiB |
| Matched-recipe first-ready diagnostic | 2,519.5 tok/s | +87.74% | 46.21 ms | 90.53 GiB |

本次 snapshot 中 first-ready diagnostic 比 fixed-barrier diagnostic 快 0.86%。三种路径的
完整有序 token-ID 轨迹一致，且
exactness violation、version mismatch、fallback、non-finite update、OOM 与 retraction
计数均为零。绑定的历史身份是 main code `0db2ff4`、旧 patched SGLang tree `e795ecc`、
execution-policy SHA `231ca579` 与 tuning-window SHA `132019ee`；该策略关闭 CUDA Graph
与 Radix Cache。

归档的 selection 与 runtime config 将两条 adaptive path 绑定到同一套 shared tuned
Adam recipe（learning rate 0.001、weight decay 0、LoRA rank 8、stride 80）。它们只是
matched-recipe publication-policy diagnostic，**不是** TTS paper reproduction，也不能调优
新的 LightCone recipe 或满足 formal gate。机制诊断表明在线工作已基本隐藏：adaptive
main-side overlap 约为 96.7%；累计约
8.4--8.7 秒的 training、optimizer、merge 与 publication 工作，只在 decode critical path
暴露约 0.27 秒。First-ready diagnostic 的优势主要对应更少的 target calls：52,083，
而 fixed-barrier diagnostic 为 52,879
（每次 call 分别 commit 12.558 与 12.369 tokens）。Adaptation 显存账本主要由 candidate
scratch（7.88 GiB）与 resident buffer（5.91 GiB）占用，而不是约 31.3 MiB 的 optimizer
state。因此后续优化顺序应为 candidate scratch、staging、training activation 生命周期、
动态 reserve calibration，最后才是 optimizer moment。固定 16 GiB reserve 使报告中的
KV capacity 从 461,703 降到 359,227 tokens，约减少 22.2%。

这只是 `n=1` 机制检查，没有正式 BCa 置信区间。循环型 prompt 具有很强的在线可学习性，
因此它既不是自然任务结果，也不是论文 LiveCodeBench、MATH-500 或 OnlineSPEC 实验复现。
下一项科学优先级应是锁定的 LiveCodeBench v6 Hard 与 MATH-500 Level 5 协议，而不是临时的
前 32 条 loader。此前 CUDA-Graph-on 诊断只有 14/16 条 token 轨迹一致，并且慢约 2.85%；
在修复 graph replay exactness 前继续关闭 CUDA Graph。

Operator 已被有意停止，主机关机。它正确封存 `failed_resumable` 与退出码 143，没有伪装成
complete；未完成的 Target-only reference 没有最终 JSON，必须重新运行。60 文件 raw archive
保持 ignored，不提交到 public tree。

## 文档

- [架构](docs/zh-CN/architecture.md)
- [数学方法](docs/zh-CN/mathematical-method.md)
- [安装](docs/zh-CN/installation.md)
- [配置](docs/zh-CN/configuration.md)
- [CLI](docs/zh-CN/cli.md)
- [SGLang patch 工作流](docs/zh-CN/sglang-patches.md)
- [实验协议](docs/zh-CN/experiment-protocol.md)
- [OnlineSPEC baseline](docs/zh-CN/onlinespec-baseline.md)
- [故障排查](docs/zh-CN/troubleshooting.md)

## 限制

- 全部 formal industrial GPU 结果在新的 clean code、patch、protocol、registry、external-source
  与 hardware identity 产生 fresh signed GPU receipt 前均为 `UNMEASURED`；
- TP2/DP2、native ITL、DSpark、NEXTN 与 compatible EAGLE3 路径都要求准确的 mode-specific
  device qualification。CPU contract 与 caller-provided key 不能授权它们。Multi-host 只执行
  independent host-local work；跨主机 collective、world size 大于二、Kubernetes、elastic
  membership 与自动 failover 仍不支持；
- TTS 与 L0-naive 要求准确 content-sealed TTS-Cal winner；release-attested lane 会额外签署该
  seal，trusted single-operator lane 则直接消费 reducer-owned content identity，不要求 external
  signer。LightCone 还要求 sealed E2 final recipe。任何 method 都不能继承 result-derived 或
  legacy diagnostic recipe；
- ChronoBelief 已注册准确 equation 与 state semantics，但仍与其他 adaptive recipe 一样需要
  runtime/GPU qualification；
- 历史 KV 按设计冻结。重新计算它需要新的算法、显存边界、协议与结论。

## 贡献与许可证

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)、
[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md) 与 [SECURITY.md](SECURITY.md)。
LightCone-Spec 使用 [Apache-2.0](LICENSE)；外部模型、数据集与 SGLang 仍遵循各自许可。
