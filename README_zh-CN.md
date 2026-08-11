# LightCone-Spec

[English](README.md) · [中文文档](docs/zh-CN/architecture.md) · [许可证](LICENSE)

LightCone-Spec 是一个以证据为先的 speculative decoding 测试时 drafter 更新研究框架。
0.2.0 只回答一个刻意收窄的问题：严格按论文实现的 TTS，以及在更新就绪后立即发布的
策略，能否都比不更新的 Static baseline 获得更高的 decode goodput？

> 当前为 Alpha 软件。下方初步 GPU snapshot 只是受控机制检查，不是正式论文复现结论。
> 在不可变实验通过统计、target exactness 与安全门槛前，正式 GPU 状态仍为
> `UNMEASURED`。

## 研究范围

正式实验只包含三种方法：

| 方法 | Candidate 计算 | 发布策略 |
|---|---|---|
| Static | 无 | 原生 SGLang speculative decoding |
| TTS | 每个 stride 在 side CUDA stream 更新 | 在下一个固定更新边界前同步并发布 |
| L0（`naive_async`） | 与 TTS 逐项相同的更新 | ready event 完成后的首个合法 graph 边界发布 |

TTS 遵循 [Test-Time Speculation 论文](https://arxiv.org/abs/2605.09329)中的调度定义。
L0 只改变发布时间，不改变 loss 或 optimizer。

正式速度实验仍固定 Qwen3-8B + DFlash，且要求 TP=DP=1。Patchset 也通过 cache-safe
tail update，为 DSpark 及 single-layer、top-k-one EAGLE/EAGLE3 实现相同的
Static/TTS/L0 发布合同。这些兼容路径的 GPU 状态为 `UNMEASURED`，不会被静默替换进
正式 DFlash gate。三个 clean-room OnlineSPEC learner 隔离在 `baselines` 包中。
OnlineSPEC 是拥有独立 tuning 与配对证据链的重要已注册对比，但不能改变核心配置选择或
Static/TTS/L0 速度 gate。固定且机器可读的
[源码审计](manifests/provenance/onlinespec_source_audit_v2.json)会区分论文公式、公开
recipe 与本项目实现。
OnlineSPEC 原论文在 Qwen3-8B 上使用 Qwen3-0.6B Lookahead Reasoning drafter；把
clean-room learner 应用于 DFlash 是独立的跨架构对比，不能写成对该系统结果的复现。

## 性能模型

实验不会预设更新一定有益：

\[
T_m=T_{\mathrm{static}}-\Delta T_{\mathrm{target}}
+T_{\mathrm{update}}^{\mathrm{exposed}}
+T_{\mathrm{draft}}^{\mathrm{extra}}+T_{\mathrm{barrier}}.
\]

只有减少 target calls 所节省的时间超过训练、资源竞争、发布与 barrier 开销，方法才会
更快。因此正式结论必须同时包含配对 decode goodput、exactness 计数、target-call
计数、CUDA 时间、HBM 账本和置信区间。

## 初步受控 GPU snapshot

下表是在一张 RTX PRO 6000 Blackwell（96 GB）上进行的机制检查：Qwen3-8B +
DFlash-b16、并发 8、greedy decoding、40,928-token 安全 context 上限。每种方法处理
相同的 16 条确定性 long-continuation prompt（每种方法生成 654,042 tokens）。注册的
exactness-safe policy 关闭 CUDA Graph 与 Radix Cache。TTS 与 L0 均使用 drafter LoRA
rank 8、Adam 学习率 `1e-3`、update stride 80，二者只在发布时间上不同。

| 方法 | Decode goodput | 相对 Static | p99 ITL | Peak HBM |
|---|---:|---:|---:|---:|
| Static DFlash | 1,342.0 tok/s | 1.00x | 45.04 ms | 90.36 GiB |
| TTS | 2,497.9 tok/s | 1.86x | 45.94 ms | 90.52 GiB |
| L0 | **2,519.5 tok/s** | **1.88x** | 46.21 ms | 90.53 GiB |

本次 snapshot 中 L0 比 TTS 快 0.86%。三种方法的完整 token-ID 轨迹完全一致；
exactness violation、version mismatch、fallback、non-finite update、OOM 与 retraction
计数均为零。
复现身份为 `LightCone-Spec@0db2ff4`、patched SGLang tree `e795ecc`、
execution-policy SHA `231ca579` 与 tuning-window SHA `132019ee`。

这些数据来自单个 tuning-window timing block（`n=1`），没有 BCa 置信区间，也不是自然
任务结果。重复型受控 prompt 用于让单请求内 adaptation 机制充分显现，不能据此声称复现了
论文的 LiveCodeBench、数学任务或 OnlineSPEC 结果。旧重复实验只对 decoded text 做
hash，未保存完整 token IDs，因此刻意不纳入本表。

## 架构

- `lightcone_spec` 管理严格的 schema-v2 配置、确定性数据窗口、配置选择、证据记录和
  统计门槛；
- `patches/sglang` 是针对唯一 upstream commit 的十个可复现 mail patch；仓库既不
  vendoring SGLang，也不原地修改 SGLang；
- cohort runtime 将 optimizer 状态保持在 GPU，在固定地址的 inference tensor 上发布，
  并用 epoch、slot generation 和 source version 绑定每个 candidate；Adam、AdamW、
  SGDm、NAG、Muon 与 Lion 共用这一 functional propose-then-commit 路径；
- headline 遥测只使用异步 CUDA event；需要同步的诊断和 profiler 在计时区间之后或
  独立 run 中执行。

详见[架构](docs/zh-CN/architecture.md)与[数学方法](docs/zh-CN/mathematical-method.md)。

## 安装

仅开发框架：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

从固定 upstream 创建一次性 patched SGLang checkout：

```bash
git clone https://github.com/sgl-project/sglang.git /path/to/sglang
git -C /path/to/sglang checkout 3312645a307453893a00778592f105581e3d1c3d
patches/sglang/apply.sh /path/to/sglang
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream --compile-only
```

GPU 环境合同见[安装](docs/zh-CN/installation.md)。

## 快速开始

先生成不可变源协议。正式启动还会绑定仓库中的 sampling profile 与分角色执行策略：

```bash
lightcone-spec build-speed-study \
  --output artifacts/protocol/static_tts_l0.json
```

下载前锁定模型 revision：

```bash
lightcone-spec lock-models --output artifacts/locks/models.json \
  Qwen/Qwen3-8B z-lab/Qwen3-8B-DFlash-b16
lightcone-spec prepare-models --lockfile artifacts/locks/models.json \
  --model-cache /path/to/model-cache \
  --output artifacts/locks/model-roots.json
```

Tuning 前为每个已注册 concurrency 生成一个零 adaptation 分配的 Static 端点
（下面以 `C=48` 为例）：

```bash
lightcone-spec render-static-load-runtime --concurrency 48 \
  --sglang-checkout /path/to/patched-sglang \
  --model-lock artifacts/locks/models.json \
  --model-roots artifacts/locks/model-roots.json \
  --sampling-profile manifests/speed-study/sampling_profile_v2.json \
  --mem-fraction-static MEMORY_FRACTION \
  --output-root artifacts/load/c48
```

独立 Static 负载扫描和 tuning 生成 selection artifact 后，渲染独占 GPU 的启动计划。
三种方法复用同一个端口和同一张 GPU，必须**顺序执行**，不得同时启动三个 server argv：

```bash
lightcone-spec render-runtime \
  --sglang-checkout /path/to/patched-sglang \
  --selection artifacts/selection.json \
  --model-lock artifacts/locks/models.json \
  --model-roots artifacts/locks/model-roots.json \
  --sampling-profile manifests/speed-study/sampling_profile_v2.json \
  --adaptation-group-id formal-a --adaptation-reserve-mb RESERVE_MB \
  --mem-fraction-static MEMORY_FRACTION \
  --output-root artifacts/runtime

lightcone-spec build-confirmation-queue \
  --manifest manifests/speed-study/static_tts_l0_v2.json \
  --selection artifacts/selection.json \
  --model-lock artifacts/locks/models.json \
  --sampling-profile manifests/speed-study/sampling_profile_v2.json \
  --launch-plan artifacts/runtime/launch-plan.json \
  --evidence-root artifacts/confirmation \
  --output artifacts/confirmation/queue.json
```

对每个 queue job：启动它的 `launch_argv`，等待健康检查，通过后执行 `run_argv`，再关闭
server，随后进入下一项。之后先生成相同 load 的 target-only 启动计划，再单独执行一次
`run-target-reference`，并把该 artifact 传给 `collect-speed-study` 派生正式表：

```bash
lightcone-spec render-target-runtime --concurrency SELECTED_CONCURRENCY \
  --sglang-checkout /path/to/patched-sglang \
  --model-lock artifacts/locks/models.json \
  --model-roots artifacts/locks/model-roots.json \
  --sampling-profile manifests/speed-study/sampling_profile_v2.json \
  --mem-fraction-static MEMORY_FRACTION \
  --output-root artifacts/target-runtime
```

注册策略固定模型 context 与 server seed，并在 target 与 DFlash 两类端点关闭 radix cache
与 CUDA graph。Target-only reference 额外关闭 overlap schedule，正式计时的 DFlash 端点保留
原生 overlap；两个角色的赋值共同进入同一内容身份。修改任一控制项都会形成不同的非正式
runtime identity。方法彼此一致
并不充分：每种方法的每个 block 都必须匹配 target-only greedy token-ID 摘要。仅比较
解码文本不能证明 exactness，因为不同 token 序列可能解码成相同文本。Queue 是数据而不是
shell 脚本；调度器必须保留注册顺序与 clean-server 边界。

`RESERVE_MB` 与 `MEMORY_FRACTION` 不设源码默认值；它们必须来自硬件预检和已选择的
参数布局。

## 更新模式与缓存合同

公共 schema 接受：

- `residual`：仅用于 tail 的低秩 logit correction；
- `lora`：用于 drafter 或 tail 消融的低秩因子，在发布时合并到固定地址权重；
- `full`：drafter scope 下更新全部 DFlash 自有浮点参数，或作为 full-rank tail 消融。
  target embedding、target LM head 与 target model 始终冻结。

DFlash 支持 drafter Full/LoRA 与三种 tail mode。DSpark 和 EAGLE/EAGLE3 刻意只支持
tail scope：residual、tail LoRA 与 full-rank tail。DSpark 在 Markov correction 之前，
对 post-normalization 的真实 LM-head 输入施加 tail；EAGLE 从 draft-extend 到 verify
持续钉住 proposal version，存在尚未验证的 proposal 时禁止发布。

在线可选 optimizer 为 `adam`、`adamw`、`sgdm`、`nag`、`muon` 与 `lion`。Muon 对二维
参数使用 matrix orthogonalization，对非 matrix 参数使用显式配置的辅助 AdamW。
Optimizer 身份及全部专属字段都进入 cohort、selection、layout 与 evidence hash。

历史 drafter KV 不可变。发布前的 KV 不重建、不参与梯度；发布后新建的 KV 记录新
source version。实际 proposal distribution 始终用于 exact speculative rejection
sampling。

## 证据与安全

正式 confirmation 在每种方法中将 32 个不同的 held-out controlled prompt 以一次有序的原生
batch 请求全部提交给 SGLang，并使用八个独立方法顺序 block 与一个已选择负载。锁定的 server
admission limit 控制 active batch，队列排空前不 reset cohort；每个 method/block 只拥有其
active decode 区间的并集。Request 诊断独立记录，避免把共享 batch 时间伪重复成 32 个
goodput 样本。40,960 token 模型上限包含 tokenized prompt；正式的
prompt-plus-generated context 在 40,928 停止，为 DFlash 保留两个 block-16
speculative KV reservation。Headline 区域从 16K generated tokens 开始。TTS 与 L0
必须分别通过已注册的速度阈值、repetition-block 配对 BCa 区间和零安全事件要求。

缺少内容绑定的 GPU attestation 时，gate 不能输出 `PASS`。Attestation 同时绑定
manifest、tuning selection、patched SGLang tree、模型 revision、硬件报告和准确的
Parquet 输入。本地或合成数据即使算术上为正，也仍是 `UNMEASURED`。

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

## 限制与路线图

- 正式 GPU 状态仍为 `UNMEASURED`；上表明确是单 block 受控 snapshot，不是经过
  attestation 的论文复现结果；
- Adaptation 要求 TP=DP=1 与未量化 drafter/KV。Drafter-scope Full/LoRA 仅用于
  DFlash；DSpark 必须使用 verify-all；EAGLE/EAGLE3 必须 single layer、fixed depth、
  top-k one，并使用 exact full-vocabulary rejection sampling。其他组合在分配
  adaptation 显存前 fail closed；
- 历史 KV 按设计冻结；重算旧 KV 会定义另一种方法和显存边界；
- 正式 DFlash 模型对之外的 GPU 认证与多 GPU adaptation 属于未来工作；实现兼容不
  代表已经实测加速。
- OnlineSPEC 对比是对论文在线 learner 公式的 clean-room 实现。固定审计 commit 上的
  官方仓库没有项目级许可证文件，因此本项目没有重新分发或复制其上游源码。

## 贡献与许可证

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)、
[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md) 与 [SECURITY.md](SECURITY.md)。
LightCone-Spec 使用 [Apache-2.0](LICENSE)；外部模型、数据集与 SGLang 仍遵循各自许可。
