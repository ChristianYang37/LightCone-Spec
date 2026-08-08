# SGLang patch 工作流

[English](../en/sglang-patches.md) · [首页](../../README_zh-CN.md)

SGLang 是外部 Apache-2.0 依赖。LightCone-Spec 固定 upstream commit
`3312645a307453893a00778592f105581e3d1c3d`，只在 `patches/sglang/` 发布 mail-format
patch series。

只对 exact、clean checkout 应用：

```bash
git clone https://github.com/sgl-project/sglang.git /tmp/sglang
git -C /tmp/sglang checkout --detach 3312645a307453893a00778592f105581e3d1c3d
patches/sglang/apply.sh /tmp/sglang
```

`apply.sh` 验证每个 patch 的 SHA-256 与最终 Git tree。`verify.sh` 在临时 clone 中重新
应用、编译修改的 Python surface、运行聚焦测试并检查反向 checkout 后保持 clean。
后续 SGLang 改动必须 patch-first；不得在本仓库提交源码 checkout 或 submodule。
