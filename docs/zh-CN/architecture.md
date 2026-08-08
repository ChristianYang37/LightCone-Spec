# 架构

[English](../en/architecture.md) · [首页](../../README_zh-CN.md)

LightCone-Spec 将不可变编排与 GPU 执行分离。host 层解析 lock、manifest、参数布局、
controller 与证据；SGLang patch 暴露 proposal 信号并持有 graph-visible buffer；side
stream 计算候选，只在合法请求边界将其复制到固定地址的 active bank。

每个候选携带 `(request_epoch, slot_generation, source_version)`。请求取消、slot 重用、
非有限值、显存压力或版本冲突只会丢弃候选，不会改变 active bank。ready event 约束
main-to-side 输入，publish event 约束 side-to-main 可见性。

关闭 adaptation 时不分配相关 buffer，并沿用 upstream SGLang 路径；开启时在 KV pool
sizing 前保留常驻 adaptation 显存。KV admission/retraction 承担负载压力，adaptation
状态不会被静默驱逐或自动移到 CPU。
