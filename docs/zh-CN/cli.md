# CLI

[English](../en/cli.md) · [首页](../../README_zh-CN.md)

## 命令

参数以 `lightcone-spec --help` 为准。0.2.0 提供：

| 命令 | 用途 |
|---|---|
| `doctor` | 只读检查 host、Python、CUDA 与源码树 |
| `validate-config` | 解析一份严格 schema-v2 run config |
| `build-speed-study` | 物化不可变源协议 |
| `lock-models` | 把模型 ID 解析为不可变 revision |
| `prepare-models` | 下载或离线核验锁定 snapshot |
| `list-tuning-candidates` | 物化已注册的 Full/LoRA 搜索网格 |
| `render-static-load-runtime` | 生成一个不分配 adaptation 状态的 Static 负载端点 |
| `render-tuning-runtime` | 为单个 candidate 渲染独占的 TTS/L0 slice |
| `run-controlled-slice` | 测量一个 load-screen 或 tuning slice |
| `collect-static-load-screen` | 验证 Static 负载网格并选择负载 |
| `advance-tuning-stage` | 验证 tuning stage 并写入 survivor 集合 |
| `select-speed-config` | 执行只读取 tuning 的 maximin 规则 |
| `select-anchor-config` | 锁定一个 terminal tuning anchor，用于 held-out 复现 |
| `render-runtime` | 输出顺序执行的匹配 confirmation 配置与 argv |
| `build-confirmation-queue` | 注册 24 个 clean-server confirmation job |
| `run-confirmation` | 执行一个 method/block confirmation slice |
| `collect-speed-study` | 从完整 receipt 派生正式表 |
| `render-replication-runtime` | 渲染自然任务或仅 profiler 使用的 slice |
| `run-natural-slice` | 运行一个锁定的自然 EOS 副表 slice |
| `build-profiler-plan` | 构建隔离的 Nsight/device-monitor 计划 |
| `attest-speed-study` | 绑定 GPU、runtime、模型、selection 和证据身份 |
| `analyze-speed-study` | 计算已注册的配对速度门槛 |

重要但与核心 gate 隔离的 OnlineSPEC 对比拥有一组平行命令：

| 命令 | 用途 |
|---|---|
| `build-onlinespec-study` | 物化绑定 provenance 的对比协议 |
| `verify-onlinespec-source` | 按已注册源码审计验证外部 clean checkout |
| `list-onlinespec-candidates` | 写出 OGD、optimistic 与 Hedge 调参网格 |
| `render-onlinespec-tuning-runtime` | 渲染配对 Static/candidate tuning endpoint |
| `run-onlinespec-tuning-slice` | 测量一个 learner tuning slice |
| `advance-onlinespec-tuning-stage` | 在每个 learner 内独立减半 candidate |
| `select-onlinespec-config` | 为每个 learner 选择一个安全 terminal candidate |
| `render-onlinespec-runtime` | 渲染 Static 与三个互斥对比 endpoint |
| `build-onlinespec-queue` | 注册随机化 clean-server 对比 job |
| `run-onlinespec-confirmation` | 执行一个 method/block 对比 slice |
| `collect-onlinespec-study` | 派生配对对比表 |
| `attest-onlinespec-study` | 把对比证据绑定到 GPU 与源码身份 |
| `analyze-onlinespec-study` | 生成 learner 对 Static 的诊断区间 |

项目不提供通用 method override 或 replay 命令。正式流程刻意保持狭窄，源码 manifest
永远不包含实测结果。

## 身份链

最小依赖链为：

```text
manifest + model lock + sampling profile + registered grid
                            |
                            v
              load screen + staged tuning
                            |
                            v
                   selection artifact
                            |
                            v
       sequential launch plan + completed evidence
                            |
                            v
          derived table + attestation + speed gate
```

每个 artifact 都有 sidecar 或内嵌 digest。身份不匹配是可操作错误；CLI 不会生成空的
成功 run，也不会用默认配置替代缺失输入。`select-speed-config` 必须同时接收 terminal
tuning artifact、完整 Static load-screen artifact、源 manifest、controlled sampling
profile 与 model lock。生成的 selection 会绑定这些身份、完整 tuning grid 与 patched
SGLang tree。

## OnlineSPEC 流程

OnlineSPEC 复用 model lock、host preflight、controlled sampling 与已选择的 Static 负载，
但绝不复用核心 tuning 或 confirmation row。正式比较前，
`verify-onlinespec-source` 会验证外部 checkout 的精确 commit/tree、clean
状态、全部已注册关键文件哈希与许可证文件清单。其内容绑定 receipt 应写入 ignored artifact
root；upstream 源码继续保留在本仓库之外。从已跟踪的
`manifests/speed-study/onlinespec_baseline_v2.json` 开始，每个 stage 为一个 candidate
及其唯一配对 Static reference 渲染运行时，并在 selection 前完成全部注册 stage。
Successive halving 在每个 learner 内独立执行。

Terminal selection 绑定完整 terminal-stage artifact 的 SHA-256，而不只是重新序列化的
winner row。它还强制要求 `--core-selection`，继承其中选定的 Static 并发量，并绑定其
SHA-256；OnlineSPEC 不再有独立并发覆盖。由于 confirmation 只有 32 个唯一 prompt，
正式 selection 不可能把并发 48 写成真实负载。随后 `render-onlinespec-runtime` 输出四个
共享端口与设备的描述；Static、OGD、optimistic 与 Hedge 必须按 manifest 的随机顺序依次执行。

```bash
lightcone-spec select-onlinespec-config \
  --measurements artifacts/onlinespec/terminal-tuning.json \
  --manifest manifests/speed-study/onlinespec_baseline_v2.json \
  --model-lock artifacts/locks/models.json \
  --sampling-profile manifests/speed-study/sampling_profile_v2.json \
  --core-selection artifacts/selection.json \
  --output artifacts/onlinespec/selection.json
```

`analyze-onlinespec-study` 与核心分析使用相同的内容绑定 attestation 纪律。缺少
attestation 时状态为 `UNMEASURED` 且退出码为 42；attested 证据若有任何安全失败则为
`BLOCKED`，同样退出 42。安全的实测输出仍然只是诊断，并固定
`core_speed_gate_affected=false`。完整公式、源码审计边界和显存账本见
[OnlineSPEC baseline](onlinespec-baseline.md)。

## 负载扫描与调参

两阶段都使用 controlled sampling profile。Static 负载扫描必须为每个注册 concurrency
生成一份 `run-controlled-slice --phase static-load` measurement。
`collect-static-load-screen` 会拒绝覆盖不全、重复、OOM、retraction 或身份不匹配的证据。

每个负载点使用 `render-static-load-runtime --concurrency C`，它只生成一个原生
Static 端点，不接受 adaptation group 或 adaptation HBM reserve，启动参数也不包含
adaptation flag。进入下一个负载点前必须停止当前 server。所有 renderer 还必须显式提供
`--sglang-checkout /path/to/patched-sglang`；该 checkout 必须 clean，且 Git tree 与本版本
记录的精确 patchset tree 一致。

Tuning 从 `list-tuning-candidates` 开始。先按选定 concurrency 只渲染一个 Static
端点，并在每个 stage 中复用其不可变启动描述作为唯一 Static baseline。每个 active
candidate/stage 使用 `render-tuning-runtime` 在同一端口只生成 TTS 与 L0 两个互斥
slice，因此不会意外重复 Static 证据。只启动当前要测量的 slice，
执行 `run-controlled-slice --phase tune`，关闭 server 后再继续。
Candidate JSON 包含完整 optimizer identity：Adam/AdamW 的 beta 与 decay、SGDm/NAG
momentum、Lion beta，或 Muon momentum、Newton--Schulz steps 与辅助 AdamW 字段。
不得删除字段，也不得手工把两个 optimizer candidate 归一化为同一 identity。
`advance-tuning-stage`
强制检查注册的 prompt 数与 context 上限、Static/TTS/L0 配对覆盖、安全计数与 survivor
身份。后续 stage 必须引用前一 stage 的 survivor artifact；confirmation 数据永不作为输入。

`select-anchor-config` 是更窄的复现入口。它要求完整的 terminal tuning-window
Static/TTS/L0 三元组、完整 Static load screen，并且 anchor 必须属于已注册 grid。生成的
selection 会标记为 `heldout_anchor`：它可以验证这个锁定配置在 held-out 数据上的速度，
但绝不声称它是全网格最优配置。该入口不能读取 confirmation 证据，也不能绕过任何
confirmation、attestation 或安全门槛。

## Confirmation 与恢复

`render-runtime` 输出的三个方法描述刻意共用一个端口，并要求独占设备。
`build-confirmation-queue` 把 manifest 中由 seed 固定的方法顺序展开为 24 个 job。对每个
job，启动其 `launch_argv`、等待健康检查、执行 `run_argv`，然后关闭该 server，才能开始
下一项。同时运行三个 endpoint 会改变 HBM、KV 容量、batching 与竞争，因此属于无效实验。

启动 wrapper 只从已验证 checkout 导入 SGLang。服务端 diagnostics 必须报告精确
adaptation config 的 SHA-256；method、optimizer、学习率、rank、stride 或 cohort 任一
不匹配，都会在提交证据前使该 slice 失效。

`run-confirmation` 每次只执行一个 method/block slice；它只 reset 一次 engine/cohort，默认
执行不计时 warmup，然后将 32 个不同 prompt 以一次有序的原生 batch 请求全部提交。
SGLang 锁定的 `max_running_requests` 负责 admission，cohort 在队列排空期间保持连续。它记录
active decode 区间的并集以及 request 级绝对 streaming arrival time。模型的 40,960-token
上限包含 tokenized prompt；每个 prompt 独立截断到已注册的 40,928 安全上限，从而保留
两个 block-16 KV reservation。

每个完整 slice 最后写入 SHA-256 绑定的 receipt。重复执行相同 job 时，只有在 manifest、
config、method、block、batch window、load 和全部 shard 均验证通过后才跳过。缺少 receipt 的
中断 shard 永不进入 `speed_study.parquet`；被修改或重复的 terminal 会 fail closed。
`--no-warmup` 只能用于诊断，不能用于正式协议。

## 自然任务复现与 Profiling

自然副表使用 `render-replication-runtime --phase natural` 和独立的 EOS-enabled sampling
profile，再对每个方法和锁定 dataset revision 执行 `run-natural-slice`。它们报告 at-risk
request，但不能影响 selection 或正式 gate。

详细 profiling 使用 `--phase profile` 与 `build-profiler-plan`。计划中固定
`headline_evidence_forbidden=true`；带同步开销的 profiler 输出不得合并进 headline 计时证据。

## Attestation 与退出状态

未提供 attestation 时，`analyze-speed-study` 可以计算诊断，但状态固定为 `UNMEASURED`
且退出码为 42。提供有效的内容绑定 GPU attestation 后，只有两种 adaptation 方法都
通过才退出零；有效的实测失败是 `BLOCKED`，同样退出 42。输入、身份、runtime 或证据
错误属于普通非零失败，不能表述为科学结果。

Artifact、模型根目录、profiler trace 和生成的 selection 应位于 ignored 输出根目录。
不得把 secret 放入 CLI 参数；模型访问只使用临时 `HF_TOKEN` 环境变量。
