# LightCone-Spec

[English](README.md) · [中文文档](docs/zh-CN/architecture.md) · [许可证](LICENSE)

LightCone-Spec 是一个以证据为先的 speculative decoding 在线 drafter adaptation
研究框架。它严格区分 runtime 工程与实证结论：CPU 合同可以验证身份、持久性和
fail-closed 行为，但只有已注册且 attested 的 GPU 证据才能证明速度或容量。

> 当前为 Alpha 软件。全部 formal industrial GPU 结果均为 `UNMEASURED`。工业级 executor 目前只有
> Target-only 可以端到端运行。Static、TTS、L0 及其他所有 speculative method 会在任何
> process 或 network mutation 前 `BLOCKED`，因为固定
> `sglang.schema_v3.content_bound_terminal_speculative_evidence.v1` hook 尚未配置可信
> hardware signer。实证 Stage B 还因缺少 provider credential、已解析 model/data/trace lock
> 与已注册硬件而 `BLOCKED`。历史 v2 artifact 仅可用于
> regression/debugging；它们不能证明 schema-v3 协议或任何新性能结论。

<!-- RESULT_TRUTH_GATE_BEGIN -->
## 结果真实性 gate

以下字段明确分隔证据状态、当前 release 可执行状态、主机运营状态与源码身份。GPU 主机关机
不代表实验完成，历史 measurement 也不会因此升级为 formal result。

| Truth field | 值 | 范围 |
|---|---|---|
| `formal_industrial_gpu_evidence` | `UNMEASURED` | 当前 schema-v3 formal 证据；没有发布任何 industrial 性能结论。 |
| `formal_industrial_execution` | `BLOCKED` | 当前 release 授权；缺少 trusted signer 且 Stage-B 输入未解析，因此在 mutation 前 fail closed。 |
| `historical_snapshot_evidence` | `PRELIMINARY_NON_FORMAL` | 下方数字 snapshot 仅为历史工程证据。 |
| `historical_snapshot_host_at_archive` | `POWERED_OFF_NOT_RELEASED` | 归档时的运营状态；实例已经关机，但没有释放或删除。 |
| `current_sglang_upstream_commit` | `3312645a307453893a00778592f105581e3d1c3d` | 当前 patch manifest 锁定的完整 Git commit。 |
| `current_patched_sglang_tree` | `81a86871d01dcf19853612da3d8eb63faa812013` | 应用当前 patch series 后预期的完整 Git tree。 |
| `current_patch_payload_sha256` | `8b9cf1f19277c765482eb0afe6b31a76f4aa8bd93e6cf29fd0e019a01011ac31` | 最新 semantic mail-patch 原始字节的 SHA-256。 |
| `current_patch_manifest_sha256` | `046bde6c6671ce490939515d06d5a38bb92dd3058e010f782e7d22269b2ace85` | 当前 canonical patch-manifest JSON 的 SHA-256。 |
| `historical_main_code_prefix` | `0db2ff4` | 仅绑定 preliminary snapshot 的短 code prefix。 |
| `historical_patched_tree_prefix` | `e795ecc` | 仅绑定 preliminary snapshot 的短 tree prefix。 |

当前 commit、tree、patch payload 与 manifest digest 是四个不同的身份域。两个七字符历史
prefix 不是当前 release identity，不能满足任何 formal identity gate。
<!-- RESULT_TRUTH_GATE_END -->

## 研究范围

核心对比包含四种方法：

| 方法 | Speculation | 在线 candidate | 发布 |
|---|---:|---:|---|
| Target-only（`target_only`） | 否 | 无 | 原生 target decoding |
| Static（`static`） | 是 | 无 | 原生 speculative decoding |
| TTS（`tts`） | 是 | side-stream update | 下一个固定更新边界 |
| L0（`l0`） | 是 | 与 TTS 逐字节相同 | ready 后首个合法边界 |

TTS 与 L0 共用完全相同的 evidence、trainable plan、loss、optimizer candidate 与
reconstruction 路径；唯一实验差异是发布时间。Target-only 与 Static 不分配任何
adaptation、optimizer、candidate 或 adaptation telemetry 状态。

目标 industrial registry 先覆盖 DFlash 与 DSpark，再进入 production load、多 GPU
topology、原生 NEXTN preflight 与 breadth template。EAGLE/EAGLE3 仍是带严格兼容性
guard 的目标 backend contract；注册并不表示当前 release 可以执行。OnlineSPEC 是重要
对比，但使用独立 tuning、证据、attestation 与分析，不能选择或改变核心 gate。

当前固定 SGLang patch 只包含 TP1/DP1 DFlash 的底层可执行 adaptive path。它会执行注册的
constant、inverse-square-root 与有限 horizon cosine schedule，也会执行非负 logical
publication delay。它已实现准确 begin/reset/finalize terminal-evidence
hook，host adapter 也会验证其内容，但仓库不携带可信 hardware signer。Static/TTS/L0 因此
仍不可用于结论，并在 release preflight、任何 mutation 之前失败。DSpark 现有纯 CPU native
contract，覆盖 adapter-free reconstruction、actual-predecessor W1 feature、native-head
selection、composite loss 与 verification-budget semantic；它没有连接 CUDA worker，也没有
授权 runtime adaptation。DSpark/EAGLE/EAGLE3/
NEXTN adaptation 与所有 TP2/DP2 execution 同样 fail closed。请求 quota-shadow row 时会记录
准确 identity 与 quota，但当前 DFlash backend capability 会阻止 acquisition，不会伪造 teacher row。

## Runtime 合同

- Schema v3 在模型分配前拒绝未知或已退役字段；
- 公共 backend envelope 绑定 adapter-free logits、采样时使用的准确 proposal
  distribution、semantic mask、target teacher row、真实 sampled predecessor、cohort
  身份、source version 与 backend-native payload；
- 目标 backend contract 要求 validator 在不重复施加 adapter 的前提下重建可微 proposal。
  只有固定 patch 的底层 TP1/DP1 DFlash path 实现该 adaptive surface；DSpark、
  EAGLE/EAGLE3 与 NEXTN 仍是不可执行 contract；
- Adaptation 只有 `full` 与 `lora`。Layer scope 为 `last1`、`last3`、`last5`、`all`；
  LoRA rank 为 1、2、4、8、16、32、64，且 `alpha/r=1`；
- DSpark 另注册三种 hybrid scope：最后 1、3 或 5 层 backbone，加原生 W1、W2 与
  acceptance/confidence 参数。E1a 恰有 56 个 adaptive configuration：32 个 layer-only，
  24 个 hybrid；
- 历史 drafter KV 冻结并带版本。发布 candidate 只影响未来 KV；重建旧 KV 或对其求导
  会定义另一种方法。

Rejection sampler 在 proposal token 被拒绝后仍从归一化 \((p-q)_+\) 采样。这个正部
rejection distribution 属于采样数学，不是 adaptation mode 或配置 alias。

## 工业级执行

声明式依赖顺序为：

```text
preflight -> E3a -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0
```

每个 cell 都绑定科学轴、seed、逻辑 rank slot、port、cache/evidence root、资源隔离与真实
初始状态。Stage receipt 在下游运行前封存 runtime/split digest、准确 dependency receipt、
activation/disposition artifact 与 locked output。唯一的确定性同主机 GPU-pool scheduler
会把逻辑声明映射到内容绑定的物理 UUID，并支持 1、2、4、8、16 或更多 GPU。它会串行
独占 headline/profile/download/compile 工作；只有准确注册的 interference-envelope rule
允许时，互相独立的工作才可并行。

资源隔离不等于 terminal authority。当前 release 会在预算与 dispatch 前阻止全部 COMPILE
与 DOWNLOAD cell，因为它既没有 release-owned compile prewarm/finalization result-pointer
contract，也没有 first-party download terminal receipt contract。

Reducer-owned E1 activation 会从 2,730-cell envelope 只物化一个 130-cell width/load slice；
E2 只物化当前 quarter-retention successive-halving round。每个 family 的四个 pilot 会在
confirmation 前锁定 12--20 final prefix 或 `UNDERPOWERED`。Per-cell budget、物理 assignment、
terminal evidence 与 observed-versus-registered GPU-time receipt 全部内容绑定。固定 patch
现已包含 first-party all-reset producer，可生成 source-owned capability、initial-state 与 reset
receipt，覆盖 drain、KV/prefix、RNG/counter、scheduler/telemetry、adaptation state、allocator/HBM
和 completion event。其 GPU reset semantics 仍明确为 `PENDING`，receipt 尚未 durable 接入
terminal 与 whole-inventory accounting。由于 scheduler 不拥有 HTTP connector，
`continuous_connection_accounting` 明确为 `false`，相应 state field 为 `null`，绝不用本地
counter 冒充。因此 live reuse 继续阻止并回退到每条 trace 一个 clean process 与 HTTP pool。

目标 schema 与 CPU coordinator vocabulary 定义单节点 TP2、sticky-replica DP2 identity、
inference-sharded TP state、replica-local DP cohort，以及 all-rank prepare/decision/
application/receipt transition；这些定义不会让 topology 变得可执行。当前 `RunConfig` 与
固定 SGLang patch 会在 model loading 前拒绝全部 TP2/DP2 run，也不能签发
`patched_two_gpu_v1` capability receipt。

真实 CPU `gloo` harness 只验证 collective state-machine 行为。它不能 attest NCCL、CUDA
stream、graph boundary、device copy、throughput 或双 GPU 正确性。

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

该命令只生成目标 declaration，不会绕过 executor 的 native-evidence preflight，也不会让
speculative 或 multi-rank cell 变得可运行。

完成一个 stage 后，封存其内容绑定 output，再把 receipt 传回 planner。准确参数以
`lightcone-spec COMMAND --help` 为准；CLI 不会把 registry 隐式当作 shell script 执行。
Model lock、证据、trace、credential、provider state 与 selection 必须放在忽略的外部 root，
不得提交。

## 统计与结论

四个被排除的 pilot block 仅用于估计方差。Power plan 会在下游 unblinding 前把 final
block 数固定在 12--20；若两个 contrast 都无法对 3% 最小效应达到 80% power，则记录
`UNDERPOWERED`。Primary L0--Static 与 L0--TTS hypothesis 使用 Holm family-wise
correction；secondary breadth family 使用 Benjamini--Hochberg FDR。长上下文/request 数据
使用 block 后 request 的 hierarchical bootstrap；production trace 使用 time-block
bootstrap。P99 latency 只有达到注册的最小完成数才合格。

每个 system point 都同时报告 throughput/goodput、TTFT/ITL、completion/error accounting、
target work、update/publication timing、HBM、每个 output token 的能耗，以及锁定的
power/clock/thermal envelope。缺失值不能替换为零。

只有 content-bound GPU attestation 同时覆盖 registry/manifest、selection、模型与数据
revision、runtime capability 与 patched tree、hardware/power report、trace 身份及精确
Parquet 输入时，gate 才可能返回 `PASS`。本地 mock、历史 v2 evidence 或正向算术仍然是
`UNMEASURED`。本 release 故意未配置 trusted hardware attester；caller-authored
doctor/attestation JSON 会被拒绝，任何 analyzer 都不能产生新的 `MEASURED` GPU outcome。

对于 exactness 诊断，`run-target-reference` 可以捕获一份锁定的 Target-only greedy
reference；每个 prompt 的 output 是完整有序 token-ID 轨迹带格式标识的 hash。Legacy
collector 要求每种 method 的每个 block 都匹配该 reference；解码文本一致或仅 speculative
method 彼此一致都不充分。该 reference 只能增强 `UNMEASURED` 诊断：它不会让
Static/TTS/L0 变得可执行，也不能替代缺失的 trusted attester。

## 历史 preliminary 机制 snapshot

以下 snapshot 来自较早的 main-line 实现，仅保留为 preliminary 工程证据。它不会改变
上文当前 industrial 协议的 formal truth：schema-v3 Stage B 仍为 `UNMEASURED` 且
`BLOCKED`，这些数字不能激活 cell、选择配置或通过 release gate。

单张 RTX PRO 6000 Blackwell（96 GB）运行 Qwen3-8B + DFlash-b16，使用 16 条重复型
受控 prompt、并发 8 和 40,928-token 安全 context 上限。每种方法均在单个 timing block
中生成 654,042 tokens：

| 方法 | Decode goodput | 相对 Static | p99 ITL | Peak HBM |
|---|---:|---:|---:|---:|
| Static | 1,342.0 tok/s | 1.00x | 45.04 ms | 90.36 GiB |
| TTS | 2,497.9 tok/s | +86.13% | 45.94 ms | 90.52 GiB |
| L0 | 2,519.5 tok/s | +87.74% | 46.21 ms | 90.53 GiB |

本次 snapshot 中 L0 比 TTS 快 0.86%。三种方法的完整有序 token-ID 轨迹一致，且
exactness violation、version mismatch、fallback、non-finite update、OOM 与 retraction
计数均为零。绑定的历史身份是 main code `0db2ff4`、旧 patched SGLang tree `e795ecc`、
execution-policy SHA `231ca579` 与 tuning-window SHA `132019ee`；该策略关闭 CUDA Graph
与 Radix Cache。

机制诊断表明在线工作已基本隐藏：TTS/L0 的 main-side overlap 约为 96.7%；累计约
8.4--8.7 秒的 training、optimizer、merge 与 publication 工作，只在 decode critical path
暴露约 0.27 秒。L0 的优势主要对应更少的 target calls：52,083，而 TTS 为 52,879
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

- 全部 formal industrial GPU 结果均为 `UNMEASURED`；native terminal hook 已存在，但 Stage B 因缺少可信
  hardware signer、provider credential、已解析 model/data/trace lock 与已注册硬件而保持
  `BLOCKED`；
- 当前端到端 industrial execution surface 只有 TP1/DP1 Target-only。TP2/DP2 仅存在于
  目标 registry/coordinator contract，本 release 会拒绝它们。不声明 multi-node、
  Kubernetes、elastic cluster、remote object store 或自动 failover；
- CPU `gloo` 合同不是 GPU/NCCL 证据。Topology config 只是目标 vocabulary；无论 caller
  是否提供 patched-runtime capability receipt，本
  release 都会拒绝 TP2/DP2；
- Static/TTS/L0 在 out-of-band 可信 signer 绑定准确 native terminal hook 与固定 tree 前
  保持 `BLOCKED`。DSpark/EAGLE/EAGLE3/NEXTN adaptive cell 与所有 TP2/DP2 cell 同样
  `BLOCKED`，绝不能暗示成功；
- ChronoBelief tuning cell 在 authoritative update equation 与 source identity 注册前保持
  `BLOCKED`，不会使用替代 optimizer；
- 历史 KV 按设计冻结。重新计算它需要新的算法、显存边界、协议与结论。

## 贡献与许可证

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)、
[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md) 与 [SECURITY.md](SECURITY.md)。
LightCone-Spec 使用 [Apache-2.0](LICENSE)；外部模型、数据集与 SGLang 仍遵循各自许可。
