# 安装

[English](../en/installation.md) · [首页](../../README_zh-CN.md)

## 框架环境

使用隔离环境。核心 Python 包不依赖本地 SGLang 源码：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
lightcone-spec doctor
pytest -q
```

可选的 `gpu` extra 安装自然任务副表使用的数据集加载器。确定性 controlled prompts
和全部 CPU 测试不依赖它。

## Patched SGLang checkout

SGLang 必须位于本仓库之外。Clone 精确 pin，并对 clean checkout 应用 mail series：

```bash
git clone https://github.com/sgl-project/sglang.git /path/to/sglang
git -C /path/to/sglang checkout --detach \
  3312645a307453893a00778592f105581e3d1c3d
patches/sglang/apply.sh /path/to/sglang
```

以下命令可在不修改源码 checkout 的前提下审计 patch series：

```bash
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream --compile-only
```

只有在环境已安装固定 SGLang 测试依赖时才移除 `--compile-only`。Verifier 会在临时目录
clone、应用全部 patch、核对 expected Git tree、编译修改过的 Python surface、按需执行
聚焦测试、反向卸载，并确认传入的 upstream checkout 始终 clean。

## GPU 环境合同

加载模型前运行 `lightcone-spec doctor --path /path/to/patched-sglang`。最终证据必须记录
NVIDIA driver、toolkit、PyTorch runtime、compiler、可用磁盘和 patched tree。不要替换
系统 Python/CUDA，不使用 `sudo`，也不复用身份不明的环境。

仓库刻意不提供一键 GPU 安装器：CUDA 与 wheel 兼容性取决于实测主机。请创建用户级
隔离环境、锁定解析后的依赖，并把模型缓存和 run artifact 放在源码树外。预检失败意味着
停止，不能静默改变精度、更新模式或显存策略。

## 模型准备

下载前解析不可变 revision：

```bash
lightcone-spec lock-models --output artifacts/locks/models.json \
  Qwen/Qwen3-8B z-lab/Qwen3-8B-DFlash-b16
lightcone-spec prepare-models \
  --lockfile artifacts/locks/models.json \
  --model-cache /path/to/model-cache \
  --output artifacts/locks/model-roots.json
```

凭据只通过临时 `HF_TOKEN` 环境变量或其他安全通道提供。不得把 token 写入命令参数、
manifest、日志或仓库文件。
