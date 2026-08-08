# 安装

[English](../en/installation.md) · [首页](../../README_zh-CN.md)

## 框架开发

使用隔离的 Python 环境并以 editable 模式安装：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

## GPU 运行时

先运行 `lightcone-spec doctor`。原生安装器要求兼容的 NVIDIA driver/toolkit 和用户级
编译工具。默认模式只读；加入 `--execute` 后，它在全新 runtime root 中 clone 固定
SGLang commit、应用 `patches/sglang/series`、创建虚拟环境并记录依赖与源码 provenance。

```bash
python scripts/install_native.py --runtime-root /absolute/runtime/path
python scripts/install_native.py --runtime-root /absolute/runtime/path --execute
```

安装器不使用 `sudo`、不替换系统 CUDA、不接受 dirty SGLang tree，也不复用未知虚拟
环境。模型准备是独立且受 lock 控制的操作。
