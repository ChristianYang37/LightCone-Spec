# Installation

[中文](../zh-CN/installation.md) · [Home](../../README.md)

## Framework environment

Use CPython 3.12 in an isolated user-space environment. The project pins
PyTorch 2.11.0 to the exact requirement of the pinned SGLang checkout; allowing
the package resolver to upgrade PyTorch creates an unsupported runtime. The
core package does not import a vendored SGLang tree:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
lightcone-spec doctor
pytest -q
```

The optional `gpu` extra installs external dataset support. Controlled traces,
registry generation, statistics, evidence durability, and CPU/gloo tests do not
require it. CPU success does not imply a GPU measurement.

The release dependency audit is currently `BLOCKED`. PyTorch 2.11.0 is the
exact version exercised by the pinned SGLang patch tree, but the strict audit
reports `PYSEC-2025-194` (fixed in PyTorch 2.13.0). PyTorch 2.11.0 also pins
Setuptools below 82, so the resolved runtime Setuptools 81.0.0 reports
`PYSEC-2026-3447` even though isolated package builds use Setuptools 83.0.0.
Do not silently upgrade either runtime dependency: the package is not
releaseable until the SGLang patch set is migrated and requalified on the fixed
PyTorch line.

On a restricted China network, set a temporary organization-approved package
index or Hugging Face endpoint in the shell that performs the download, record
that endpoint in the sanitized environment receipt, and unset it afterward.
Do not commit mirror credentials or silently replace a locked artifact when a
mirror is missing it.

## SGLang patch gate

SGLang must remain outside this repository. Clone the exact pin and apply the
complete mail series to a clean disposable checkout:

```bash
git clone https://github.com/sgl-project/sglang.git /path/to/sglang
git -C /path/to/sglang checkout --detach \
  3312645a307453893a00778592f105581e3d1c3d
patches/sglang/apply.sh /path/to/sglang
```

Audit from a separate clean upstream checkout:

```bash
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream --compile-only
```

The schema-v3 patch has passed clean-HEAD, patch-digest, expected-tree,
changed-source compile/focused-test, and reverse-removal verification. Running
with `--compile-only` checks patch integrity but is not the release gate; the CI
gate installs the pinned dependencies and runs the focused patched-tree tests.
Record a fresh verifier output and final-tree receipt before GPU work. This
result does not constitute GPU validation.

## GPU inventory and rank contract

Run `lightcone-spec doctor --project-root /path/to/lightcone-spec
--sglang-root /path/to/patched-sglang` before loading a model. Capture driver,
toolkit, PyTorch/CUDA runtime, compiler, GPU UUIDs, clocks, temperature, power
state, storage, background processes, and patched tree. Materialize a
content-bound `GpuInventory` with PCI/NUMA/interconnect and allowed topology
groups before planning. Do not replace system Python/CUDA, use `sudo`, or reuse
an unidentified environment.

The pool scheduler accepts arbitrary same-host inventory size, with 1/2/4/8/16
GPU regression coverage. The scientific registry still carries two logical
rank slots; a frozen assignment resolves them to physical UUIDs for each cell.
This inventory scaling supports more independent jobs and topology-aware gangs,
not an executable larger-rank method. The tracked compatibility manifest
continues to describe its exact reference host separately.

The target registry and CPU coordinator describe one-node TP2 and sticky DP2
identities, but the current release accepts only TP1/DP1 and rejects every
TP2/DP2 `RunConfig` before model loading. A future multi-rank release would
require a content-bound `patched_two_gpu_v1` capability receipt and matching
receipts from every rank. The real CPU `gloo` harness tests collective state
transitions only; it cannot enable that release surface or supply GPU/NCCL
evidence.

HBM preflight must measure every rank and uses the least feasible rank. Choose
adaptation reserve, KV pool, safety margin, fixed cohort-slab capacity, and
telemetry queue bounds from the registered memory ledger. Never make a run fit
by silently changing Full to LoRA, precision, scope, optimizer, or context.

## Provider staging

The repository deliberately has no one-command cloud installer. Provider
images, drivers, mounts, and firewall behavior are external state. Before
creating an instance, obtain credentials through the provider's secure channel,
confirm the requested inventory and storage are available, and record a
sanitized provisioning receipt. Never place provider secrets, temporary URLs,
instance addresses, or access tokens in commands, manifests, evidence, handoff
documents, or Git.

At present, empirical Stage B is `BLOCKED` because no trusted hardware signer,
provider credentials, immutable model/data/trace locks, or registered hardware
and interference envelope are available. The pinned integration already
implements the exact native begin/reset/finalize hook, but the industrial
executor can release-run only TP1/DP1 Target-only; Static/TTS/L0 fail preflight
before mutation. Hardware access or a test signer alone does not unblock
speculative work. DSpark/EAGLE/EAGLE3/NEXTN adaptation and all TP2/DP2 work need
additional implementations and remain blocked.

## Model and data preparation

Resolve immutable revisions before downloading:

```bash
lightcone-spec lock-models --output artifacts/locks/models.json \
  Qwen/Qwen3-8B z-lab/Qwen3-8B-DFlash-b16
lightcone-spec prepare-models \
  --lockfile artifacts/locks/models.json \
  --model-cache /path/to/model-cache \
  --output artifacts/locks/model-roots.json
```

Use a temporary `HF_TOKEN` environment variable or another secure credential
channel. Lock tokenizer, dataset revision, split, prompt compiler, and trace
identity separately. A BurstGPT-shaped synthetic trace remains labelled
synthetic unless an immutable external corpus is supplied and digested.

## Evidence roots

Keep model snapshots, runtime caches, provider state, traces, WAL segments,
Parquet shards, receipts, profiles, selections, attestations, and handoff files
under ignored external roots. Each rank/process gets a unique evidence prefix.
Interrupted WALs are retained for audit but cannot enter analysis without an
exclusive terminal receipt.

Build compile caches as verified content-addressed immutable bases and give
each process a private writable overlay. Do not share a writable cache or move
a captured CUDA Graph between processes or devices. The immutable session-key
and boundary-receipt schemas are audit-only in this release: shared-session
execution is blocked before mutation until a release-owned trusted boundary,
durable session receipt binding, and continuous whole-inventory accounting are
available. The high-level block executor uses a validated clean-process
fallback, with a distinct official HTTP pool and native provider for every
logical trace. The pool is reused only within that one single-trace execution.

Do not commit model/data payloads, experimental results, selected
hyperparameters derived from results, machine paths, credentials, or provider
metadata. A source checkout should remain clean except for deliberate code and
documentation changes.
