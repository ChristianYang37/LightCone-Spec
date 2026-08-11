# CLI

[English](../en/cli.md) · [首页](../../README_zh-CN.md)

## 命令 Surface

准确参数以 `lightcone-spec --help` 与 `lightcone-spec COMMAND --help` 为准。Schema-v3 与
industrial 命令包括：

| 命令 | 用途 |
|---|---|
| `doctor` | 只读 host、Python、CUDA 与 source identity 报告 |
| `validate-config` | 验证一份严格 schema-v3 run config 与 sidecar |
| `build-industrial-registry` | 绑定两个 GPU UUID 并生成不可变实验 DAG |
| `plan-industrial-dispatch` | 验证 receipt 并生成确定性单/双 GPU wave |
| `seal-industrial-stage` | 绑定 completed stage 的 runtime、split、dependency 与 locked output |
| `analyze-industrial` | 验证 schema-v3 terminal evidence，并写入绑定的 E3b/E5 analysis manifest |
| `build-speed-study` | 生成较小的核心源协议 |
| `lock-models` | 把 model ID 解析为不可变 revision |
| `prepare-models` | 下载或 offline 验证 locked snapshot |
| `list-tuning-candidates` | 写入已注册 Full/LoRA tuning grid |
| `render-target-only-runtime` | 渲染关闭 speculation 的 Target-only endpoint |
| `render-static-load-runtime` | 渲染零 adaptation 分配的 Static endpoint |
| `render-tuning-runtime` | 渲染 matched TTS/L0 tuning endpoint |
| `run-controlled-slice` | 测量一个已注册 controlled slice |
| `collect-static-load-screen` | 验证 Static load coverage 并选择 reference load |
| `advance-tuning-stage` | 验证 halving stage 并封存 survivor |
| `select-speed-config` | 应用 tuning-only 注册 selection rule |
| `select-anchor-config` | 锁定 terminal registered anchor，但不声明 optimum |
| `render-runtime` | 生成 matched sequential 核心 config 与 launch argv |
| `build-confirmation-queue` | 生成 clean-server confirmation job |
| `run-confirmation` | 执行一个 method/block confirmation slice |
| `collect-speed-study` | 从 completed receipt-bound evidence 派生 table |
| `render-replication-runtime` | 渲染 natural-task 或 profiler-only slice |
| `run-natural-slice` | 运行一个 locked natural-task slice |
| `build-profiler-plan` | 生成隔离 profile plan，并禁止 headline evidence |
| `attest-speed-study` | 绑定 GPU、runtime、model、selection、trace 与 evidence identity |
| `analyze-speed-study` | 评估已注册 paired gate |

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
| `attest-onlinespec-study` | 绑定 comparison evidence 与 source identity |
| `analyze-onlinespec-study` | 生成 learner-versus-Static 诊断区间 |

不存在 generic method override，也不存在把无 receipt 目录转换为 completed evidence 的命令。

## Industrial Registry 工作流

使用不可变 device UUID，不要使用 CUDA ordinal：

```bash
lightcone-spec build-industrial-registry \
  --gpu-uuid GPU-UUID-0 GPU-UUID-1 \
  --base-port 24000 \
  --cache-root runtime-cache/industrial \
  --evidence-root artifacts/industrial \
  --seed 20260811 \
  --output artifacts/industrial/registry.json
```

输出嵌入 generator identity、输入 parameter、完整 declaration 与 registry SHA-256。加载时
会重新生成 registry 并比较准确内容，因此手工编辑 cell 会被拒绝。

没有 receipt 时，planner 只生成 `preflight` wave；由于 interference gate 尚未通过，单 GPU
工作保持串行：

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --output artifacts/industrial/preflight-dispatch.json
```

Stage 的注册 output durable 后再封存。每个 `--locked-output` 格式为
`NAME=LOWERCASE_SHA256`；dependency 必须传 receipt file，而不是复制 hash string：

```bash
lightcone-spec seal-industrial-stage \
  --registry artifacts/industrial/registry.json \
  --experiment preflight \
  --runtime-sha256 RUNTIME_SHA256 \
  --split-sha256 SPLIT_SHA256 \
  --locked-output runtime_envelope=OUTPUT_SHA256 \
  --output artifacts/industrial/receipts/preflight.json
```

下游 stage 对其准确 declared dependency 重复 `--dependency-receipt`，随后把所有 completed
receipt 交给 planner：

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --receipt artifacts/industrial/receipts/preflight.json \
  --completed-cells artifacts/industrial/completed-cells.json \
  --interference-receipt artifacts/industrial/interference.json \
  --output artifacts/industrial/next-dispatch.json
```

可选 completed-cell artifact 必须把每个 cell 绑定到 measured evidence 与 terminal receipt。
可选 interference artifact 必须针对相同 registry/GPU UUID 且为 `PASS`；缺少时 cell 保持
串行。即使通过，exclusive work 也绝不配对。

Dispatch plan 是目标 protocol 数据，不证明 cell 可执行。Library industrial executor 会在
launch 前验证 provider state，目前只有 TP1/DP1 Target-only 可端到端运行。除非 injected
provider 针对准确固定 tree 实现
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1`，否则全部 speculative cell
会在 process/network mutation 前被阻止；本 release 不包含该 provider。CLI 不会静默启动
server 或 provision hardware。

## 身份与 Topology 链

最小 industrial identity chain 为：

```text
source + patch tree + model/data locks + two-GPU capability
                         |
                         v
              immutable registry + traces
                         |
                         v
           dependency receipts + dispatch waves
                         |
                         v
              terminal evidence receipts
                         |
                         v
        derived statistics + GPU attestation + gate
```

TP2 与 sticky DP2 字段是目标 registry/CPU-coordinator vocabulary。当前 `RunConfig` 会在
model loading 前拒绝全部 TP2/DP2 value；CPU `gloo` contract 或 caller 自己填写 receipt 都
不能启用。未来实现必须准确绑定 rank、UUID、rendezvous、router、clock、process-group、
ownership 与 receipt identity。

当 command contract 要求时，generated JSON 使用相邻 `.sha256` sidecar。Loader 验证
canonical content，不只检查 filename。

## 核心 Tuning 与 Confirmation

Target-only 与 Static 渲染时不含 adaptation object 或 reserve。TTS 与 L0 的目标 declaration
从已注册 native layer scope 共享一个 Full/LoRA candidate。Rendering 只是 pure planning；
当前 release 只有 Target-only 可以进入 industrial execution。Static/TTS/L0 会在 endpoint
启动前因 native terminal-evidence preflight 失败。

Load screening 与 tuning 只能使用各自注册 data window。后续 halving stage 必须绑定前一
survivor artifact。`select-speed-config` 要求完整 safe coverage；`select-anchor-config` 是较窄
reproduction 路径，并记录未优化全网格。Confirmation data 不能进入任一 selection。

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
`optimized_grid_claim=false`。OnlineSPEC 保持 TP1/DP1，并使用自己的 GPU attestation。

## 状态、Resume 与 Exit Code

缺少 attestation 时，`analyze-speed-study` 可输出诊断，但状态为 `UNMEASURED` 且退出 42。
提供有效 content-bound GPU evidence 后，只有完整注册 pass 才退出零；有效 evidence 未满足
任一标准时为 `BLOCKED`，同样退出 42。Identity、schema、I/O、receipt 或 runtime 错误是
普通 nonzero failure，不是科学结果。
当前 release 没有 trusted hardware-attester identity，因此 supplied legacy attestation
会被拒绝；即使 JSON 内部一致，`analyze-industrial` 仍保持
`UNRESOLVED`/`UNMEASURED` 并退出 42。

Industrial registry 可能把目标 cell 声明为 `UNMEASURED`，但 declaration status 不等于
executable readiness。准确 native terminal hook 缺失时，executor preflight 会把
Static/TTS/L0 解析为 `BLOCKED`；当前 schema/patch 同样阻止全部 TP2/DP2 与
DSpark/EAGLE/EAGLE3/NEXTN adaptive cell。历史 v2 artifact 仅用于 regression，不能作为
schema-v3 stage receipt。

## Credential 与 Output Root

Artifact、model root、cache、trace、provider state、profile、selection、attestation 与
handoff file 必须位于 ignored external root。通过临时 `HF_TOKEN` 环境变量或其他安全渠道
传递模型权限。不要把 token、password、provider API key、private prompt、instance address
或 machine-specific path 放入 argument、manifest、log、文档或 Git。
