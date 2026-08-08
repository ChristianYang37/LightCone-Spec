# LightCone-Spec

[English](README.md) · [中文文档](docs/zh-CN/architecture.md) · [许可证](LICENSE)

LightCone-Spec 是一个面向 speculative decoding 的、版本安全的异步测试时
adaptation 研究框架。它将 proposal correction 与 drafter backbone 分离，只在
合法解码边界发布更新，并将 controller 和实验绑定到可复现的模型、运行时与数据
身份。

> 当前为 Alpha 软件。本仓库公开实现与评测协议，不公开性能结论或实验结果。

## 架构

系统刻意分为三层：

1. `lightcone_spec` 负责候选更新、controller、不可变 manifest 与证据 artifact；
2. 固定 patch series 为临时 SGLang checkout 增加通用 proposal 信号和版本化 tail bank；
3. 实验执行器强制 exactness、模型锁、请求级数据切分、遥测完整性和 fail-closed 科学门槛。

Adaptation tensor 保持 GPU resident。active buffer 地址固定以支持 CUDA graph replay；
staging、optimizer 与 candidate 工作位于 side stream。请求 epoch、slot generation 和
source version 共同阻止 stale/ABA 发布。

## 支持的后端

| 后端 | 支持范围 | 当前边界 |
|---|---|---|
| DSpark | residual、LoRA、full-rank tail；Markov 与 confidence 信号 | checkpoint 与 proposal depth 在模型加载前校验 |
| DFlash | residual、LoRA、full-rank tail | 仅确定性 proposal 和已认证的 rejection-sampling 路径 |
| EAGLE / EAGLE3 | residual、LoRA、full-rank tail | 单层、`topk=1`；tree/多层等未支持组合会 fail closed |

关闭 adaptation 时保留原生 SGLang 路径。本仓库不 vendoring SGLang 或模型权重。

## 安装

仅开发框架代码：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
lightcone-spec --help
```

GPU 环境先执行只读预检，再运行原生安装器。安装器从固定 upstream commit 创建临时
checkout 并应用已验证 patchset；它不修改系统 Python/CUDA，且没有 `--execute` 时
不会执行安装。

```bash
lightcone-spec doctor --output doctor.json
python scripts/install_native.py --runtime-root ~/lightcone-spec-runtime
python scripts/install_native.py --runtime-root ~/lightcone-spec-runtime --execute
```

详见[安装](docs/zh-CN/installation.md)与
[SGLang patch 工作流](docs/zh-CN/sglang-patches.md)。

## 快速开始

下载或加载模型前必须先生成不可变输入锁：

```bash
lightcone-spec lock --output lightcone.lock.json \
  --pairs qwen3_4b_dflash16 --datasets livecodebench
lightcone-spec prepare-models --lockfile lightcone.lock.json \
  --model-cache /path/to/cache --pairs qwen3_4b_dflash16 \
  --output model-roots.json
lightcone-spec run-manifest --manifest manifests/smoke/smoke_gpu_qwen3_4b.json \
  --lockfile lightcone.lock.json --model-roots model-roots.json \
  --runtime-root /path/to/runtime --model-cache /path/to/cache
```

当 manifest 需要 selection/controller artifact 时，真实运行必须显式提供。缺失、身份
不匹配或以结果派生配置作为隐式默认值都会在模型加载前被拒绝。

## 更新模式与调度器

`--weight-update-mode` 只接受：

- `residual`：压缩的输出 logit residual；
- `lora`：proposal tail 上的低秩更新；
- `full`：full-rank proposal-tail 更新，不代表全 drafter 微调。

L0 在首个合法边界发布完成的候选；L1 对候选做 gate；L2 缩放候选幅度；L3 评估
transport 后的候选。controller 阶段只有在不可变证据门槛通过后才具有正式含义。

## 正确性、显存与证据链

- 生成 proposal 与 rejection sampling 使用同一 corrected distribution `q`；
- semantic mask 排除 rejected suffix、stop/EOS 后 token、越过请求边界的 bonus token，
  以及超过 `max_new_tokens` 的 token；
- adaptation 显存在 KV pool sizing 前保留，不静默 offload 或驱逐；压力交给 admission
  control 与 KV retraction；
- 轻量 CUDA event 遥测与同步式详细 profiling 分离；
- artifact 绑定模型 revision、参数布局、运行时源码、数据窗口、seed 和 patch 身份。

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

GPU 与模型兼容范围有意小于 SGLang 本身。未支持的 speculative tree、参数范围、
checkpoint context window 和 sampling 组合会 fail closed。后续计划包括扩展后端、探索
更多 cache-safe 可训练范围、完成多 GPU 认证，并仅在证据门槛完成后公开实验结论。

## 贡献与安全

参见 [CONTRIBUTING.md](CONTRIBUTING.md)、
[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md) 和 [SECURITY.md](SECURITY.md)。
LightCone-Spec 使用 [Apache-2.0](LICENSE) 许可证。
