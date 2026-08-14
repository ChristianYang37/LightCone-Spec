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

目标 registry 与 CPU coordinator 描述单节点 TP2 与 sticky DP2 identity，但当前 release
只接受 TP1/DP1，并会在 model loading 前拒绝全部 TP2/DP2 `RunConfig`。未来 multi-rank
release 将要求内容绑定的 `patched_two_gpu_v1` capability receipt 与每个 rank 的 matching
receipt。真实 CPU `gloo` harness 只测试 collective state transition，不能启用该 release
surface，也不能提供 GPU/NCCL evidence。

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

当前实证 Stage B 因 trusted hardware signer、provider credential、immutable model/data/trace
lock 以及已注册 hardware/interference envelope 均不可用而 `BLOCKED`。固定 integration 已
实现准确 native begin/reset/finalize hook，但 industrial executor 的 release run 仍只有
TP1/DP1 Target-only；Static/TTS/L0 会在任何 mutation 前 fail preflight。仅获得硬件或 test
signer 也不能解除 speculative blocker。DSpark/EAGLE/EAGLE3/NEXTN adaptation 与全部
TP2/DP2 工作还需要其他实现，保持 blocked。

## Trusted Attester Bundle

可信 verification material 以 canonical `TrustedAttesterPolicyBundle` 及相邻 semantic SHA-256
sidecar 的形式，在仓库外 provision。Public bundle 包含 Ed25519 public key 与 attester
allowlist、challenge-nonce lifetime 与 external single-use replay policy、hardware-envelope
digest allowlist 及 validity interval；绝不能包含 private key、signing seed、test identity、
credential 或 host routing detail。

信任来自另行 provision 的 `TrustedAttesterAnchorDescriptor`；它固定 bundle 的 absolute
normalized path、semantic digest、validity 与 authority。Loader 会拒绝 symlink、hard link、
重复 JSON key、非 canonical byte、读取中变化的文件、expired policy、未 allowlist hardware，
以及 replayed 或未绑定 challenge。Caller 自选 path 加 digest 不构成 trust root。

本 source release 的 `SOURCE_RELEASE_TRUSTED_ATTESTER_ANCHOR` 刻意未配置。因此，即使 operator
能为 diagnostic deployment 加载 public verification material，formal release 仍会 fail
closed。未来经过 review 的 release 必须先在 out-of-band source authority 中 anchor public
bundle，formal DAG 才可能输出 `MEASURED`；signing key 始终留在外部 attester，绝不提交或
放入 command argument。

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
