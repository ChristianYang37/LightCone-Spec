# SGLang patch 工作流

[English](../en/sglang-patches.md) · [首页](../../README_zh-CN.md)

## 源码边界

SGLang 是外部 Apache-2.0 项目。LightCone-Spec 固定 upstream commit
`3312645a307453893a00778592f105581e3d1c3d`，只分发 mail-format patch。仓库不包含
SGLang 源码、submodule 或修改后的 checkout。本地 `sglang/` 目录被 ignore，永远不作为
集成身份。

## Patch 分层

十个 patch 具有单向的语义依赖：

1. 严格的跨 backend schema、preflight 与 disabled fast path；
2. cohort、resident optimizer 与 OnlineSPEC learner state、source version、CUDA event
   和发布 runtime；
3. 可微 DFlash drafter Full/LoRA 与 OnlineSPEC 更新路径；
4. cache-safe DFlash、DSpark、EAGLE 与 EAGLE3 tail 路径；
5. 显存账本、lifecycle、遥测与 profiling 集成；
6. 跨 backend optimizer、proposal、exactness 与回归测试；
7. 请求边界的 speculative KV headroom 与有界 reservation 校验；
8. 显存受限的 LoRA 坐标 OnlineSPEC Hedge 决策类及其协议回归测试；
9. overlap scheduling 下由已提交 prefix 在设备端推导请求剩余预算；
10. 最终 overlap 与发布轮之间的逻辑 prefix 连续性。

只支持完整 series。中间 patch 状态是 review 边界，不是可运行的产品变体。

OnlineSPEC 被折叠到既有语义层，而不是创建一套平行 runtime patch：patch 一负责 schema，
patch 二负责 learner state，patch 三负责 DFlash gradient，patch 四负责跨 backend tail
routing，patch 五负责显存与诊断，patch 六负责协议测试，patch 七负责严格的请求边界 KV
lifecycle，patch 八负责可选的低显存 Hedge 决策类，patch 九负责已提交 prefix 的请求预算，
patch 十负责逻辑 prefix 连续性。这样版本、event、exactness 与 disabled path 始终只有一份实现。

## 应用

`patches/sglang/apply.sh` 只接受位于精确 upstream HEAD 的 clean checkout。它校验每个
patch digest，使用 `git am` 应用，并核对 `manifest.json` 记录的最终 Git tree：

```bash
patches/sglang/apply.sh /path/to/clean-sglang
```

Dirty state、错误 commit、被修改的 patch、mail 应用失败或最终 tree 不匹配都会立即停止。
脚本不会通过 rebase、stash、reset 或编辑输入 checkout 来掩盖错误。

## 验证与编写

发布前运行一次性 verifier：

```bash
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream
```

新的集成改动必须 patch-first：从 pin 创建临时分支，完成单一语义 commit 和聚焦测试，
用 `git format-patch` 导出，然后同时更新 `series`、patch SHA-256、修改文件列表、expected
final tree、Python pin 常量、NOTICE 与文档。最后验证应用与反向卸载，并确认原始 upstream
源码仍保持 clean。
