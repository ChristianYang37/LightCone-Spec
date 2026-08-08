# CLI

[English](../en/cli.md) · [首页](../../README_zh-CN.md)

`lightcone-spec` 提供以下工作流：

- `doctor`：只读的 host/GPU 兼容性报告；
- `lock`：解析可变的模型、数据集和环境输入；
- `prepare-models` / `prepare-datasets`：物化并验证锁定输入；
- `serve`：启动一个已验证的 adaptation server；
- `run-manifest`：以隔离输出路径执行不可变 unit；
- `replay`：oracle replay 与 controller 拟合；
- `exactness`：检查 proposal/accept/reject 正确性；
- `analyze`：生成协议表格与图；
- `validate-artifacts`：递归验证证据。

以 `lightcone-spec COMMAND --help` 为参数权威来源。缺少 lock、controller、selection
artifact 或 runtime binding 时会产生可操作错误，而不是空的成功运行。队列退出码区分
成功、科学阻塞、锁冲突和可恢复的工程失败。
