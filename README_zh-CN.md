# LightCone-Spec

[English](README.md) · [中文文档](docs/zh-CN/architecture.md) · [许可证](LICENSE)

LightCone-Spec 是一个以证据为先的 speculative decoding 在线 drafter adaptation
研究框架。它严格区分 runtime 工程与实证结论：CPU 合同可以验证身份、持久性和
fail-closed 行为，但只有已注册且 attested 的 GPU 证据才能证明速度或容量。

> 当前为 Alpha 软件。全部新 GPU 结果均为 `UNMEASURED`。工业级 executor 目前只有
> Target-only 可以端到端运行。Static、TTS、L0 及其他所有 speculative method 会在任何
> process 或 network mutation 前 `BLOCKED`，因为固定 SGLang integration 尚未提供
> `sglang.schema_v3.content_bound_terminal_speculative_evidence.v1`。实证 Stage B 还因缺少
> provider credential 与已注册硬件而 `BLOCKED`。历史 v2 artifact 仅可用于
> regression/debugging；它们不能证明 schema-v3 协议或任何新性能结论。

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

目标 industrial registry 先覆盖 DFlash 与 DSpark，再进入 production load、双 GPU
topology、原生 NEXTN preflight 与 breadth template。EAGLE/EAGLE3 仍是带严格兼容性
guard 的目标 backend contract；注册并不表示当前 release 可以执行。OnlineSPEC 是重要
对比，但使用独立 tuning、证据、attestation 与分析，不能选择或改变核心 gate。

当前固定 SGLang patch 只包含 TP1/DP1 DFlash 的底层 adaptive path，且 optimizer schedule
为 constant、extra logical delay 为零。它没有暴露 industrial executor 所需的内容绑定
terminal evidence provider，因此该路径还不能端到端执行或支持结论。Static/TTS/L0 会在
executor preflight、任何 mutation 之前失败。DSpark/EAGLE/EAGLE3/NEXTN adaptation、所有
TP2/DP2 execution、非 constant schedule 与正 extra delay 同样 fail closed；在新的 patch
与 provider identity 实现这些能力前，对应 registry cell 保持 `BLOCKED`。

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

每个 cell 都绑定科学轴、seed、GPU UUID、port、cache/evidence root、资源隔离与真实初始
状态。Stage receipt 在下游运行前封存 runtime/split digest、准确 dependency receipt 与
locked output。双 GPU scheduler 会串行执行独占的 headline/profile/download/compile
工作；只有 interference gate 通过后，互相独立的单 GPU 工作才可并行。

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

用真实不可变 device 身份生成双 GPU industrial registry：

```bash
lightcone-spec build-industrial-registry \
  --gpu-uuid GPU-UUID-0 GPU-UUID-1 \
  --cache-root runtime-cache/industrial \
  --evidence-root artifacts/industrial \
  --output artifacts/industrial/registry.json

lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
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

- 全部新 GPU 结果均为 `UNMEASURED`；Stage B 因缺少 native terminal speculative-
  evidence provider、provider credential 与已注册硬件而保持 `BLOCKED`；
- 当前端到端 industrial execution surface 只有 TP1/DP1 Target-only。TP2/DP2 仅存在于
  目标 registry/coordinator contract，本 release 会拒绝它们。不声明 multi-node、
  Kubernetes、elastic cluster、remote object store 或自动 failover；
- CPU `gloo` 合同不是 GPU/NCCL 证据。Topology config 只是目标 vocabulary；无论 caller
  是否提供 patched-runtime capability receipt，本
  release 都会拒绝 TP2/DP2；
- Static/TTS/L0 在准确 native terminal evidence hook 实现并绑定固定 tree 前保持
  `BLOCKED`。DSpark/EAGLE/EAGLE3/NEXTN adaptive cell 与所有 TP2/DP2 cell 同样
  `BLOCKED`，绝不能暗示成功；
- ChronoBelief tuning cell 在 authoritative update equation 与 source identity 注册前保持
  `BLOCKED`，不会使用替代 optimizer；
- 历史 KV 按设计冻结。重新计算它需要新的算法、显存边界、协议与结论。

## 贡献与许可证

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)、
[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md) 与 [SECURITY.md](SECURITY.md)。
LightCone-Spec 使用 [Apache-2.0](LICENSE)；外部模型、数据集与 SGLang 仍遵循各自许可。
