# LightCone-Spec

[English](README.md) · [中文文档](docs/zh-CN/architecture.md) · [许可证](LICENSE)

LightCone-Spec 是一个以证据为先的 speculative decoding 测试时 drafter 更新研究框架。
0.2.0 只回答一个刻意收窄的问题：严格按论文实现的 TTS，以及在更新就绪后立即发布的
策略，能否都比不更新的 Static baseline 获得更高的 decode goodput？

> 当前为 Alpha 软件。在不可变的正式 GPU 实验通过统计与安全门槛前，GPU 状态始终为
> `UNMEASURED`。本仓库不发布 benchmark 数值或性能结论。

## 研究范围

正式实验只包含三种方法：

| 方法 | Candidate 计算 | 发布策略 |
|---|---|---|
| Static | 无 | 原生 SGLang speculative decoding |
| TTS | 每个 stride 在 side CUDA stream 更新 | 在下一个固定更新边界前同步并发布 |
| L0（`naive_async`） | 与 TTS 逐项相同的更新 | ready event 完成后的首个合法 graph 边界发布 |

TTS 遵循 [Test-Time Speculation 论文](https://arxiv.org/abs/2605.09329)中的调度定义。
L0 只改变发布时间，不改变 loss 或 optimizer。

目前在线 adaptation 仅认证 Qwen3-8B + DFlash，且要求 TP=DP=1。未提供 adaptation
配置时，其他后端仍走原生 SGLang 路径。三个 clean-room OnlineSpec 公式隔离在
`baselines` 包中，不进入默认实验。

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

## 架构

- `lightcone_spec` 管理严格的 schema-v2 配置、确定性数据窗口、配置选择、证据记录和
  统计门槛；
- `patches/sglang` 是针对唯一 upstream commit 的六层可复现 mail patch；仓库既不
  vendoring SGLang，也不原地修改 SGLang；
- cohort runtime 将 optimizer 状态保持在 GPU，在固定地址的 inference tensor 上发布，
  并用 epoch、slot generation 和 source version 绑定每个 candidate；
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

先生成不可变源协议与 sampling profile：

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
server，随后进入下一项。最后用 `collect-speed-study` 派生正式表。Queue 是数据而不是 shell
脚本；调度器必须保留注册顺序与 clean-server 边界。

`RESERVE_MB` 与 `MEMORY_FRACTION` 不设源码默认值；它们必须来自硬件预检和已选择的
参数布局。

## 更新模式与缓存合同

公共 schema 接受：

- `residual`：仅用于 tail 的低秩 logit correction；
- `lora`：用于 drafter 或 tail 消融的低秩因子，在发布时合并到固定地址权重；
- `full`：drafter scope 下更新全部 DFlash 自有浮点参数，或作为 full-rank tail 消融。
  target embedding、target LM head 与 target model 始终冻结。

历史 drafter KV 不可变。发布前的 KV 不重建、不参与梯度；发布后新建的 KV 记录新
source version。实际 proposal distribution 始终用于 exact speculative rejection
sampling。

## 证据与安全

正式 confirmation 使用 32 个 held-out controlled prompts、八个独立方法顺序 block、
一个已选择负载，并按生成 token 位置记录到每个 prompt 的 checkpoint 安全终点。40,960
token 的模型上限始终包含 tokenized prompt；headline 区域从 16K generated tokens 开始。
TTS 与 L0 必须分别通过已注册的速度阈值、prompt-cluster 配对
BCa 区间和零安全事件要求。

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
- [故障排查](docs/zh-CN/troubleshooting.md)

## 限制与路线图

- GPU 状态目前为 `UNMEASURED`，不声明任何加速；
- 在线 adaptation 仅支持 DFlash、TP=DP=1、未量化 drafter/KV；其他组合在分配
  adaptation 显存前 fail closed；
- 历史 KV 按设计冻结；重算旧 KV 会定义另一种方法和显存边界；
- 多 GPU 认证及更多 speculative backend 属于未来工作，不存在隐藏或半启用实现。

## 贡献与许可证

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)、
[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md) 与 [SECURITY.md](SECURITY.md)。
LightCone-Spec 使用 [Apache-2.0](LICENSE)；外部模型、数据集与 SGLang 仍遵循各自许可。
