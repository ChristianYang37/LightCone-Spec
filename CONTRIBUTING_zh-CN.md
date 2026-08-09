# 为 LightCone-Spec 贡献

[English](CONTRIBUTING.md)

感谢参与 LightCone-Spec。所有改动都应保持可复现、可审查，并对运行大模型的用户安全。

## 开发环境

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

提交 PR 前运行 `python -m compileall -q src scripts tests`、`git diff --check` 和相关
聚焦测试。GPU 行为应使用独立的 GPU marker，不能让 CPU 测试依赖 CUDA。

## 设计原则

- 优先选择最小、可复用的抽象，避免复制 lifecycle、version bank、optimizer 或证据逻辑；
- 无法建立身份、exactness 或兼容性时，在模型加载前 fail closed；
- 关闭 adaptation 时保持 allocation-free 并与 upstream 路径兼容；
- 正式方法面只保留 Static、TTS 与 L0；新的研究想法应先独立评审，不能藏在 feature flag；
- TTS 与 L0 必须复用同一 candidate 计算实现，只有发布策略可以不同；
- 保持冻结的历史 drafter KV 及其分段 source version；修改该合同即定义新方法；
- 不提交模型、数据集、遥测、profile、实验结果、凭据、provider 状态或机器绝对路径；
- 不添加 SGLang checkout/submodule。针对固定 upstream 编写语义清晰的 mail patch，并
  同步更新 manifest、checksum、测试、expected tree 与文档。

## Pull request

保持 commit 聚焦并说明被修改的不变量。请包含 CPU 测试、GPU 测试需求、兼容性影响、
显存账本影响和 exactness 证据。性能结论必须来自批准的实验协议，不接受临时测量。

完整项目规范见 [`.codex/project-standards.md`](.codex/project-standards.md)。

除非另有明确说明，贡献均使用 Apache-2.0 许可证。
