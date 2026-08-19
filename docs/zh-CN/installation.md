# 安装

[English](../en/installation.md) · [首页](../../README_zh-CN.md)

## 框架环境

在隔离的 user-space 环境中使用 CPython 3.12。项目把 PyTorch 固定为 2.11.0，与固定
SGLang checkout 的准确要求一致；允许 package resolver 升级 PyTorch 会产生不受支持的
runtime。核心包不导入 vendored SGLang tree：

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
lightcone-spec doctor
pytest -q
```

可选 `gpu` extra 安装外部 dataset 支持。Controlled trace、registry generation、statistics、
evidence durability 与 CPU/gloo 测试不需要它。CPU 成功不表示 GPU measurement。

在受限中国网络中，只在执行下载的 shell 临时设置组织批准的 package index 或 Hugging
Face endpoint，把 endpoint 写入脱敏 environment receipt，并在完成后 unset。不得提交
mirror credential；mirror 缺少 locked artifact 时也不得静默替换。

## SGLang Patch Gate

SGLang 必须保持在仓库外。从准确 pin clone，并把完整 mail series 应用到 clean disposable
checkout：

```bash
git clone https://github.com/sgl-project/sglang.git /path/to/sglang
git -C /path/to/sglang checkout --detach \
  3312645a307453893a00778592f105581e3d1c3d
patches/sglang/apply.sh /path/to/sglang
```

从另一个 clean upstream checkout 审计：

```bash
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream --compile-only
```

Schema-v3 patch 已通过 clean-HEAD、patch digest、expected tree、changed-source
compile/focused-test 与 reverse-removal verification。`--compile-only` 只检查 patch
integrity，不是 release gate；CI 会安装固定 dependency 并运行 patched-tree focused test。
GPU 工作前必须记录新的 verifier output 与 final-tree receipt。该结果不构成 GPU validation。

## GPU Inventory 与 Rank 合同

加载模型前运行 `lightcone-spec doctor --project-root /path/to/lightcone-spec
--sglang-root /path/to/patched-sglang`。记录 driver、toolkit、PyTorch/CUDA runtime、compiler、
GPU UUID、clock、temperature、power state、storage、background process 与 patched tree。
Planning 前先 materialize content-bound `GpuInventory`，包含 PCI/NUMA/interconnect 与 allowed
topology group。不要替换 system Python/CUDA、使用 `sudo` 或复用身份不明的环境。

Pool scheduler 接受任意 same-host inventory size，并对 1/2/4/8/16 GPU 有 regression
coverage。Scientific registry 仍携带两个 logical rank slot；frozen assignment 为每个 cell
把它们解析为 physical UUID。这种 inventory scaling 支持更多 independent job 与
topology-aware gang，不表示较大 rank method 已可执行。Tracked compatibility manifest 仍会
独立描述其准确 reference host。

源码 runtime 实现三种已注册单机模式：`tp1_dp1`、`tp2_dp1` 与 sticky-replica
`tp1_dp2`。Distributed `RunConfig` 只在携带准确的 source-owned
`patched_two_gpu_v1` capability identity 与 runtime-envelope receipt 时被接受。Schema 支持
不等于 GPU authority；TP2/DP2 在全新 dynamic qualification、all-rank publication 与
terminal evidence 被验证前仍保持 fail closed。真实 CPU `gloo` harness 只测试 collective
state transition，不能提供 GPU/NCCL evidence。

HBM preflight 必须测量每个 rank，并由最不可行 rank 决定。Adaptation reserve、KV pool、
safety margin、固定 cohort-slab capacity 与 telemetry queue bound 都来自已注册 memory ledger。
绝不能通过静默改变 Full/LoRA、precision、scope、optimizer 或 context 来让 run 勉强可用。

## Multi-host 准备

每台主机都必须作为独立 security 与 contention domain 准备。Fleet assembly 前，在每台主机
分别运行 `doctor`、收集 nonce-bound single-host `GpuInventory`，并派生或校准该主机自己的
`InterferenceEnvelope`。即使 GPU 型号相同，也不能复用其他主机的 envelope。Port、compile-
cache overlay、evidence root 与 materialization manifest 必须在各自主机的 resource domain
内无冲突；不同主机可以重复使用相同 literal value。

Repeated inventory 与 envelope argument 必须按同一顺序成对出现：

```bash
lightcone-spec assemble-gpu-fleet-inventory \
  --inventory /external/host-a/inventory.json \
  --interference-envelope /external/host-a/interference.json \
  --inventory /external/host-b/inventory.json \
  --interference-envelope /external/host-b/interference.json \
  --output /external/fleet/inventory.json
```

该 artifact 只启用 independent host-local cell 的 scheduling，不授权跨主机 gang。TP/DP gang
必须能在一台主机内完整放置；否则 planning 返回
`cross_host_collectives_unvalidated`。异构 hardware envelope 在 analysis 中保持分离。

Remote wave 需要 coordinator-local SSH agent socket 以及 absolute、固定、不可写的
`known_hosts` file。Transport 禁用 password/interactive authentication、agent forwarding、
port forwarding 与 user SSH configuration。Host address、user、agent socket 与 host-key path
属于 routing state，不得进入 manifest 或 evidence。Coordinator 当前只提供 Python library
API，不得虚构 fleet execution CLI。Remote worker command 准确为
`lightcone-spec execute-dispatch-wave --host-request-stdin`，只从 stdin 接收 canonical JSON。
Fleet transport concurrency 必须显式有界。一旦 request 可能已到达 worker，任何 timeout、
connection loss、截断 response 或缺失 authority 都是不可直接重试的
`REMOTE_OUTCOME_UNKNOWN`。只有绑定准确原始 destination、port 与 known-host key 的独立
fetch 严格重开并验证准确 receipt 与 evidence bytes 后才能 reconcile；endpoint value、
host-key bytes 与 credential 不会持久化。否则 attempt 必须保持 unknown。

## Provider Staging

仓库刻意不提供一键 cloud installer。Provider image、driver、mount 与 firewall behavior 属于
外部状态。创建 instance 前，先通过 provider secure channel 获得 credential，确认所需 GPU
inventory 和 storage 可用，并记录脱敏 provisioning receipt。不得把 provider secret、
temporary URL、instance address 或 access token 写入 command、manifest、evidence、handoff
document 或 Git。

Source tree 本身不包含可直接宣称实测完成的 Stage B。Formal session 在取得 fresh provider
state、root-authorized deployment/hardware policy、不可变 prepared-model/workload content
receipt、精确 compile/exactness/interference terminal 以及 stage capacity control 前保持
`BLOCKED`。固定 integration 已提供 first-party compile 与 non-serving terminal contract；
checkout 中存在 source capability 并不等于已有 execution evidence，硬件或 test signer 也
不能单独解除 speculative blocker。Formal TTS/L0-naive 还要求 sealed TTS-Cal winner，
LightCone 要求准确 sealed E2 winner，二者都不能从 default 推断。TP2/DP2、DSpark、NEXTN、
native ITL 与 session reuse 已实现，但必须等待各自精确的 dynamic GPU proof。EAGLE3 还要求
独立签署的官方 model/selector compatibility decision；不支持或不兼容的组合保持 N/A 或
`BLOCKED`。

## Trusted Attester Bundle

Package 在 `manifests/runtime/release_ed25519_root_v1.json` 固定一个离线 Ed25519 **公钥**及
fingerprint，并用相邻 raw-file SHA-256 sidecar 绑定；wheel 与 sdist 携带相同 public
resource。Repository 不保存 private key、signing seed、credential、host route 或 hardware
digest。

取得真实 inventory 后，离线 root 对短期、challenge-bound、包含准确 homogeneous hardware
allowlist 与 typed control-attester keys 的 deployment policy 签名。该动态层不需要猜测未来
GPU topology，也不要求更改 source HEAD。Loader 会验证 pinned root、policy signature 与
validity、精确 policy digest、hardware membership、challenge replay reservation 以及
path/content identity；wrong key、expiry、replay、TOCTOU、symlink、hard link、非 canonical
bytes 或 caller-selected trust root 均被拒绝。External private key 不会复制到 remote instance，
也不会进入 argv、environment variable 或 log。

Public root 的存在不会把 run 变成 `MEASURED`；每个 formal session 仍需要 fresh root-signed
deployment policy 与相应的 locally controlled terminal/aggregate attestation。

### 本地离线签名仪式

只在可信本地签名端运行签名。Source-owned signer 通过继承的 file descriptor 接收 private
key，或通过不回显的 TTY 提示读取绝对 key path；它没有 private-key path、private bytes 或
passphrase 参数，也不会从 environment 读取这些值。Key file 必须位于 private directory，
由当前用户拥有、权限为 `0600`、只有一个 hardlink；公钥必须匹配 pinned root 或
root-authorized signing policy。

```bash
python -m lightcone_spec.runtime.offline_signer sign-deployment \
  --bundle /safe/public/deployment-bundle.json \
  --inventory-sha256 "$INVENTORY_SHA256" \
  --challenge-id deployment-2026-08-17-001 \
  --output /safe/evidence/deployment-authorization.json

python -m lightcone_spec.runtime.offline_signer sign-control \
  --subject /safe/public/compile-control-subject.json \
  --deployment-authorization /safe/evidence/deployment-authorization.json \
  --hardware-envelope-sha256 "$HARDWARE_ENVELOPE_SHA256" \
  --attester-id release-signer \
  --key-id release-signer-key \
  --challenge-id compile-control-2026-08-17-001 \
  --output /safe/evidence/compile-control.json

python -m lightcone_spec.runtime.offline_signer sign-scientific \
  --artifact-type stage-materialization \
  --payload /safe/public/stage-materialization.json \
  --deployment-authorization /safe/evidence/deployment-authorization.json \
  --attester-id release-signer \
  --key-id release-signer-key \
  --challenge-id stage-materialization-2026-08-17-001 \
  --output /safe/evidence/stage-materialization.candidate.json

python -m lightcone_spec.runtime.offline_signer finalize-scientific \
  --artifact-type stage-materialization \
  --candidate /safe/evidence/stage-materialization.candidate.json \
  --deployment-authorization /safe/evidence/deployment-authorization.json \
  --challenge-ledger /safe/private/single-use-challenge-ledger \
  --output /safe/evidence/signed-stage-materialization.json
```

所有签名命令都会创建 fresh challenge，并以 no-replace 语义发布 canonical file。Scientific
authority 使用 closed、typed 的两阶段仪式：第一步只接受注册的 payload type；finalizer 会重新
核验 deployment policy，并先在 private single-use ledger 中保留 challenge，再发布 signed
wrapper；不存在 generic JSON signing mode。自动化只能
通过 `--key-fd` 传入数字型 inherited descriptor；不得把 key 或 key path 放进 argv。只将
public authorization/control artifact 复制给执行流程；绝不能把 private key 或 signer input
FD 复制到 GPU host。

本地签名 CLI 还提供帮助：

```bash
python -m lightcone_spec.runtime.offline_signer sign-deployment --help
python -m lightcone_spec.runtime.offline_signer sign-control --help
python -m lightcone_spec.runtime.offline_signer sign-scientific --help
python -m lightcone_spec.runtime.offline_signer finalize-scientific --help
```

帮助命令同样不会读取或输出 private material。

## Model 与数据准备

下载前解析不可变 revision：

```bash
lightcone-spec lock-models --output artifacts/locks/models.json \
  Qwen/Qwen3-8B z-lab/Qwen3-8B-DFlash-b16
lightcone-spec prepare-models \
  --lockfile artifacts/locks/models.json \
  --model-cache /path/to/model-cache \
  --output artifacts/locks/model-roots.json
```

使用临时 `HF_TOKEN` 环境变量或其他安全 credential channel。Tokenizer、dataset revision、
split、prompt compiler 与 trace identity 分别锁定。除非提供并 digest 不可变外部 corpus，
BurstGPT-shaped synthetic trace 始终标为 synthetic。

## Evidence Root

Model snapshot、runtime cache、provider state、trace、WAL segment、Parquet shard、receipt、
profile、selection、attestation 与 handoff file 必须放在 ignored external root。每个 rank/
process 使用唯一 evidence prefix。中断 WAL 保留用于审计，但没有 exclusive terminal receipt
就不能进入分析。

Compile cache 必须构建为 verified content-addressed immutable base，并为每个 process 提供
private writable overlay。不要共享 writable cache，也不要在 process/device 之间移动 captured
CUDA Graph。Immutable session key 与 boundary receipt schema 现已有 first-party source
producer；在支持的 single-tokenizer HTTP/1.1 uvicorn 路径上，连续 HTTP accounting 来自真实
protocol lifecycle，并绑定一个 process generation；Granian HTTP/2 与 multiple-tokenizer
HTTP-process 路径会在生成该 capability 前 fail closed。它只覆盖 reset-state accounting，尚无
native warm-up/trace/close receipt，且 GPU semantics 仍为 `PENDING`。在 GPU reset validation、
durable session
receipt binding 和连续 whole-inventory 计费可用之前，shared-session execution 会在 mutation
前被阻止。高层 block executor 会先
验证完整 block，再对每个 logical trace 使用独立 clean process、official HTTP pool 与 native
provider；HTTP pool 只会在该 single-trace server execution 内复用。

不得提交 model/data payload、实验结果、从结果选择的 hyperparameter、machine path、credential
或 provider metadata。Source checkout 除刻意 code/documentation change 外应保持 clean。
