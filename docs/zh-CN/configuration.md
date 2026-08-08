# 配置

[English](../en/configuration.md) · [首页](../../README_zh-CN.md)

配置按 schema 默认值、不可变 manifest、少量显式 CLI override 分层解析。
`--weight-update-mode` 是 `residual`、`lora`、`full` 的唯一公共 override；显式覆盖会
重新计算 unit 和 artifact 身份。

真实 manifest 必须绑定 lockfile、model-roots digest、runtime fingerprint、数据窗口、
sampling profile 与 seed。controller 方法还要求 artifact 的模型对、normalization、
distance weight、数据 hash、参数布局和 lifecycle 与运行一致。

Static unit 不分配 adaptation 状态。未支持组合或新旧配置来源冲突会在模型加载前被
拒绝。禁止将 access token、密码、机器路径或由结果选出的超参数写进待提交 manifest。
