# 架构

[English](../en/architecture.md) · [README](../../README_zh-CN.md)

## 系统边界

LightCone-Spec 将三类职责分开：Python 包负责配置、实验身份、证据与统计推断；mail
patch 只向一次性 SGLang checkout 加入必要 runtime hook；模型、数据集、CUDA 库和
run artifact 全部位于仓库之外。

任何本地 SGLang 目录都不是依赖或事实来源。唯一集成身份是 exact upstream commit 与
`patches/sglang/manifest.json` 中的 expected final Git tree。

## 解码生命周期

Static 使用原生 DFlash 路径，不导入 adaptation module。TTS 与 L0 在每个 server
process 中创建一个同质 cohort runtime：

1. Verification round 暴露当前 teacher logits、drafter hidden、semantic mask 和
   source version；
2. 到达配置 stride 时，每个活跃请求只保留最近一次合法信号，并归一化为 cohort batch；
3. 低优先级 CUDA stream 重建当前 DFlash canvas，与 inference 输出核对，计算 KL
   gradient，并在不修改 active state 的前提下生成 optimizer proposal；
4. Candidate 绑定 cohort epoch、slot generation、source round 与 source version；
5. TTS 等待下一个固定边界；L0 在每个合法边界查询 ready event。发布把 staging 值
   copy 到固定地址 tensor；
6. Stale、cancelled、non-finite、conflicting 或未认证 candidate 都是 no-op；原生解码
   继续执行，并记录原因。

最多只有一个 in-flight candidate，从协议上限制 side-stream 工作并明确取消语义。

## 参数布局

Drafter `full` 覆盖全部 DFlash 自有浮点参数，包括 transformer linear、norm、final
norm 与 `fc`；target embedding、LM head 和 target model 是借用且冻结的组件。Drafter
`lora` 选择 DFlash 自有二维 `fc`、QKV、output、gate/up 与 down matrix。A/B 因子作为
FP32 optimizer state；发布时只合并已选择 matrix，并保持原模型 storage 不变。

Tail 消融先修正 hidden，再只执行一次 target-head projection。Residual tail correction
复用 proposal 输出，不再运行第二次 target-head projection。

## KV 与版本

历史 drafter KV 是不可变状态输入，不是 trainable tensor。仅 update round 使用的可微
路径把受限 paged history gather 到 scratch 后 detach；当前 canvas K/V 保持可微。发布
后产生的新 KV 使用新权重版本；遥测以 half-open segment 记录 source version。

该合同避免在长请求中使全部历史 KV 失效。它不声称等价于用新权重重建旧 KV；后者是
另一种算法。

## 显存与并发

Adaptation 显存在 KV-pool sizing 前分配。账本拆分 active/base、FP32 master、gradient、
一阶与二阶动量、staging、training activation、KV gather scratch、candidate scratch、
graph buffer 和 telemetry。Resident adaptation state 不可驱逐，也不静默 offload 或降档；
剩余 HBM 决定 KV capacity 与 admission。

只有模型 revision、算法、sampling profile、tenant、实验组、参数布局和 optimizer identity
全部相同的请求才能共享更新。Batch 对每个请求只保留最近一次合法信号，不积累跳过轮次
的 stale supervision。

## 证据链

每个 run 写入 process-unique 的 run、request、round、update 和 performance Parquet
shard。缺失计数是错误，不会补零。Run-scope speculative count 不会复制到 generated-token
bucket。派生的 `speed_study.parquet` 只写入 ignored artifact root。

正式 gate 必须有 attestation，同时绑定源 manifest、selection、模型 revision、patched
runtime tree、硬件报告与精确 evidence file digest；否则状态保持 `UNMEASURED`。
