# 故障排查

[English](../en/troubleshooting.md) · [首页](../../README_zh-CN.md)

- **Patch 被拒绝：**核对完整 upstream commit、detached clean 状态、patch checksum 和
  `patches/sglang/manifest.json`；
- **Runtime fingerprint 不匹配：**不要修改已经 screen 的运行时；创建新的临时 checkout
  和证据根目录；
- **模型加载被拒绝：**重新生成 lock 与 model-roots，确认所需模型对完整存在，不要降低
  hash 校验；
- **Controller 被拒绝：**核对模型对、parameter-layout hash、数据窗口、sampling、
  lifecycle 与 runtime 身份；
- **显存不足：**降低 admission、context 或 load；adaptation buffer 不会被自动 offload
  或静默降档；
- **队列科学阻塞：**检查带 attestation 的 terminal receipt；只有 final gate 的递归
  `verify-resume` 接口可以恢复旧队列。

报告 bug 时请提供脱敏命令、平台版本、准确的 upstream/patch tree ID 和最小复现
manifest。不要公开 token、密码、私有模型路径或原始私有 prompt。
