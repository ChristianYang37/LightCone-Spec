# 故障排查

[English](../en/troubleshooting.md) · [首页](../../README_zh-CN.md)

## Patch 或 Runtime 身份失败

- 核对准确 detached upstream commit、clean status、series order、逐 patch digest、
  modified-file inventory 与 `patches/sglang/manifest.json` 中 expected final tree；
- 运行完整 disposable apply/compile/test/reverse verifier。CPU package test 成功不能验证
  pending SGLang migration；
- Run 不得隐式引用 workspace checkout。显式传入 verified disposable checkout，并绑定其
  tree receipt；
- Source、package、model、sampling、trace、topology 或 launch argument 的任何 drift 都
  要求新 runtime identity 与下游 evidence root。

## 配置或 Backend 被拒绝

- Schema v3 使用 canonical `target_only`、`static`、`tts`、`l0`。未知 method name 与已
  退役 adaptation 字段属于错误；
- Target-only 要求关闭 speculation 且无 adaptation；Static 启用 speculation 但无
  adaptation；移除 method 字段后，TTS/L0 的 adaptation 必须逐字节相同；
- 对 industrial executor 而言，当前只有 TP1/DP1 Target-only 为 READY。固定 tree 已实现
  `sglang.schema_v3.content_bound_terminal_speculative_evidence.v1`，但没有配置
  release-trusted hardware signer，因此 Static/TTS/L0 会在 mutation 前失败；这是预期的
  fail-closed release boundary；
- Adaptation 是 `last1`、`last3`、`last5` 或 `all` 上的 Full/LoRA。LoRA 要求注册 rank 与
  `alpha/r=1`。借用 target parameter、量化或不属于 backend 的 coordinate 不能训练；
- DSpark layer-only 与 `*_native_heads` hybrid 字段是目标 contract；当前 adaptive schema
  会在 model loading 前拒绝 DSpark。未来实现必须解析真实 W1/W2/confidence state，不得使用
  placeholder Markov feature 或推断 predecessor；
- EAGLE/EAGLE3 与 NEXTN validator/hook 是目标 contract，不是当前实现。它们的 adaptive
  config 会在 model loading 前被拒绝；
- Optimizer 专属字段严格校验：SGDm/NAG/Muon 要求 momentum；Muon 还要求 Newton--Schulz
  与 auxiliary AdamW value。未使用字段不会被忽略。

## Distributed Rank 被拒绝或 Collective 失败

- 当前 `RunConfig` 会在 model loading 前拒绝全部 TP2/DP2 value；caller 自己填写
  `patched_two_gpu_v1` digest 也不能启用 multi-rank 工作；
- 单节点/双 rank identity、sticky DP routing 与 all-rank receipt 字段只是 CPU coordinator
  目标 contract。它们描述未来固定实现必须证明的内容，不是当前 SGLang support；
- 缺 prepare vote、foreign topology receipt、generation mismatch、non-finite candidate、
  unsafe boundary 或不完整 post-copy receipt 会让全部 rank abort update。Transport/split-
  decision failure 还会关闭 admission，直至显式 same-topology restart；
- CPU `gloo` harness 必须使用 live gloo process group。它刻意拒绝 NCCL，不能当作 GPU
  capability evidence；
- Multi-node、超过两个 rank、Kubernetes、elastic membership 与 automatic failover 不受
  支持，也不存在隐藏 flag。

## 显存压力

检查 per-category、per-rank HBM ledger。Admission 由 headroom 最小 rank 决定，且先计入
model/KV、FP32 master、gradient、真实 moment、candidate/staging、activation、graph、
telemetry 与 safety margin。只按 trainable parameter count 估计是不完整的。

Pressure handling 保留 active correctness state，只允许驱逐 native inactive prefix，随后
取消 pending adaptation，并 queue 或拒绝新工作。可选 inactive-cohort offload 必须显式，
且排在最后。不要通过改变 Full/LoRA、rank、scope、optimizer、precision、context 或
admission 隐藏 OOM；每次改变都要求新 config 与 load screen。

Slab exhausted 时检查 tenant quota、active reference、replica identity、generation counter
与可选 cold-offload timer。只有 inactive cohort 可回收。没有 matching reclamation/transfer
receipt 就复用 slab 属于 stale-state bug。

## Telemetry Backpressure 或中断

Evidence writer 有 queued-row 上限，并把 queue drain 批量写入 Parquet WAL row group。已
注册 row/time checkpoint 与 terminal boundary 执行 durable fsync；queue 暂时清空不是
fsync 请求。`backpressure` 下 producer 等待；显式 `drop` 下 triggering negative row 仍
保持 durable，drop counter 增加，且 attempt 不能满足 complete evidence。提高 bound 会
改变 runtime identity，必须在 measurement 前决定。

Durable index 拒绝重复 request/round/update/performance key。Checkpoint、WAL row-group
coverage、final shard schema/digest 与 completion-receipt counter 必须一致。不要 rename、
combine、truncate 或手工编辑 segment。

中断后保留 WAL/checkpoint/aborted marker 供审计，并从相同 immutable cell resume。只跳过
一个有效 exclusive terminal receipt。没有 receipt 的 final Parquet file，或同一 run/rank
存在多个 completed attempt，都不是有效证据。

## Registry 或 Dispatch 被拒绝

- 通过 CLI 加载 generated registry；generator、parameter、embedded declaration 或 SHA-256
  不匹配表示内容被编辑；
- Registry 接受两个 logical rank slot，绝不接受 physical device argument。向 pool planner
  提供 strict content-bound inventory、准确 per-cell budget sequence 与 interference
  envelope；physical UUID 只出现在 frozen assignment；
- 只提供 ready stage 声明的准确 receipt。Locked-output name 必须匹配 definition，并使用
  lowercase SHA-256；
- E1/E2 completion 需要准确 reducer activation，并为每个 template 提供 disposition。
  Confirmation completion 需要 family-local pilot/final activation 及准确 power plan。Forged
  或 cross-round survivor 是错误，不是 runnable work；
- Completed serving row 需要 terminal receipt、physical assignment、`ExperimentBudget` 与
  terminal-bound `BudgetObservationReceipt`。所有 observed phase component 及 measured/billed
  GPU-time closure 必须一致；目录存在不表示完成；
- Concurrency 由准确 matching `InterferenceEnvelope` 限制。更多 idle GPU 不会授权未经校准
  co-run class。Gang job 必须 atomic；profiler/download/compile 为 exclusive-host。不能仅因
  assignment 出现在同一个 plan 就并发启动全部工作；
- Resume 会重放 path-bound append-only attempt journal，并现场重开每个成功 terminal
  authority。Failed sibling 不会抹去 durable successful receipt，且只有失败 sibling 会消耗
  retry。Intent 若没有 finish，会保持
  `dispatch_attempt_intent_without_finish_cost_unresolved`；不要删除它或估算其成本。

## 意外的 `UNMEASURED`、`BLOCKED` 或 `UNDERPOWERED`

`UNMEASURED` 表示不存在合格 content-bound GPU evidence/attestation。正向 diagnostics、
CPU mock、历史 v2 result 或 acceptance change 都不能改变它。当前新 GPU phase 保持
`UNMEASURED`；Stage B 因缺少 trusted hardware signer、provider credential、immutable
model/data/trace lock、已注册硬件、GPU smoke 与 interference evidence 而 `BLOCKED`。
Static/TTS/L0、全部 DSpark/EAGLE/EAGLE3/NEXTN adaptive cell 与全部 TP2/DP2 cell 都
blocked；只有 TP1/DP1 Target-only 可端到端执行。

`BLOCKED` 也是 registered cell prerequisite 或 attested criterion 失败时的正确结果。
`UNDERPOWERED` 表示四个 excluded pilot block 无法按注册 power 选择 12--20 个 final block，
因此 confirmation 不能开始。`UNRESOLVED` p99 表示 completion 少于 10,000。不得重命名、
省略、impute 或优化掉这些状态。

## 安全与 Bug Report

报告脱敏 command、package/platform version、准确 upstream/patched tree ID、registry/cell/
trace digest、topology receipt 与最小可复现 schema-v3 config。不得公开 token、password、
provider key、temporary URL、private prompt、instance address、model path 或 raw provider
state。
