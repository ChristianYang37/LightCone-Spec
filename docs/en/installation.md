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

The source runtime implements three registered single-host modes:
`tp1_dp1`, `tp2_dp1`, and sticky-replica `tp1_dp2`. A distributed `RunConfig`
is accepted only when it carries the exact source-owned
`patched_two_gpu_v1` capability identity and a runtime-envelope receipt. That
schema support is not GPU authority. TP2 and DP2 remain fail-closed until a
fresh root-authorized deployment policy permits the observed homogeneous
two-GPU inventory and the matching GPU qualification artifact proves the
rank, UUID, rendezvous, ownership, publication, and terminal contracts. The
real CPU `gloo` harness tests collective state transitions only; it cannot
supply GPU/NCCL evidence.

HBM preflight must measure every rank and uses the least feasible rank. Choose
adaptation reserve, KV pool, safety margin, fixed cohort-slab capacity, and
telemetry queue bounds from the registered memory ledger. Never make a run fit
by silently changing Full to LoRA, precision, scope, optimizer, or context.

## Multi-host preparation

Prepare every host as a separate security and contention domain. Run `doctor`,
collect a nonce-bound single-host `GpuInventory`, and derive or calibrate that
host's own `InterferenceEnvelope` before fleet assembly. Do not reuse another
host's envelope, even for the same GPU model. Ports, compile-cache overlays,
evidence roots, and materialization manifests must be collision-free within
each host's resource domain. Their literal values may repeat on different hosts.

Pair repeated inventory and envelope arguments in the same order:

```bash
lightcone-spec assemble-gpu-fleet-inventory \
  --inventory /external/host-a/inventory.json \
  --interference-envelope /external/host-a/interference.json \
  --inventory /external/host-b/inventory.json \
  --interference-envelope /external/host-b/interference.json \
  --output /external/fleet/inventory.json
```

This artifact enables scheduling of independent host-local cells. It does not
authorize a gang across hosts. TP/DP gangs must fit one host; otherwise planning
returns `cross_host_collectives_unvalidated`. Heterogeneous hardware envelopes
remain separate for analysis.

For remote waves, provision a coordinator-local SSH agent socket and an
absolute, fixed, non-writable `known_hosts` file. The transport disables
password and interactive authentication, agent forwarding, port forwarding,
and user SSH configuration. Host address, user, agent socket, and host-key path
are routing state and must not appear in manifests or evidence. The coordinator
is currently a Python-library API; do not invent a fleet execution CLI. The
remote worker command is exactly `lightcone-spec execute-dispatch-wave
--host-request-stdin` and receives canonical JSON only on standard input.
Fleet transport concurrency is explicitly bounded. Once a request may have
reached the worker, any timeout, connection loss, truncated response, or missing
authority is `REMOTE_OUTCOME_UNKNOWN` and must not be retried. Reconcile it only
through the independent fetch bound to the exact original destination, port,
and known-host key; it reopens and validates the exact receipt and evidence
bytes. Endpoint values, host-key bytes, and credentials are not persisted.
Otherwise leave the attempt unknown.

## Provider staging

The repository deliberately has no one-command cloud installer. Provider
images, drivers, mounts, and firewall behavior are external state. Before
creating an instance, obtain credentials through the provider's secure channel,
confirm the requested inventory and storage are available, and record a
sanitized provisioning receipt. Never place provider secrets, temporary URLs,
instance addresses, or access tokens in commands, manifests, evidence, handoff
documents, or Git.

The source tree does not itself contain a runnable empirical Stage B. A formal
session remains `BLOCKED` until it has fresh provider state, root-authorized
deployment/hardware policy, immutable prepared-model and workload content
receipts, exact compile/exactness/interference terminals, and a stage capacity
control. The pinned integration contains first-party compile and non-serving
terminal contracts, but their absence from a checkout is intentional: source
capability is not execution evidence. Hardware access or a test signer alone
does not unblock speculative work. Formal TTS/L0-naive additionally require a
sealed TTS-Cal winner, and LightCone requires the exact sealed E2 winner;
neither is inferred from defaults. TP2/DP2, DSpark, NEXTN, native ITL, and
session reuse are implemented pending their exact dynamic GPU proofs. EAGLE3
requires a separate signed official model/selector compatibility decision;
unsupported or incompatible combinations stay N/A or `BLOCKED`.

## Trusted attester bundle

The package pins one offline Ed25519 **public** root and its fingerprint in
`manifests/runtime/release_ed25519_root_v1.json`, with a raw-file SHA-256
sidecar. The installed wheel and sdist carry the same public resource. No
private key, signing seed, credential, host route, or hardware digest is stored
in the repository.

After a real inventory is observed, the offline root signs a short-lived,
challenge-bound deployment policy containing the exact homogeneous hardware
allowlist and typed control-attester keys. This dynamic layer avoids guessing a
future GPU topology or changing source HEAD. Loaders verify the pinned root,
policy signature and validity, exact policy digest, hardware membership,
challenge replay reservation, and path/content identity. They reject wrong
keys, expiry, replay, TOCTOU changes, symlinks, hard links, noncanonical bytes,
and caller-selected trust roots. The external private key is never copied to a
remote instance or passed through argv, environment variables, or logs.

The presence of the public root does not make a run `MEASURED`: each formal
session still needs a fresh root-signed deployment policy and the appropriate
locally controlled terminal/aggregate attestations.

### Local offline-signing ceremony

Run signing only on the trusted local signing host. The source-owned signer
accepts a private key through an inherited file descriptor, or prompts for an
absolute key path through an unechoed TTY. It has no private-key path, private
bytes, or passphrase argument and never reads those values from the
environment. Key files must be single-link, current-user-owned mode `0600`
files in a private directory; the public key must match the pinned root or the
root-authorized signing policy.

```bash
python -m lightcone_spec.runtime.offline_signer sign-deployment \
  --bundle /safe/public/deployment-bundle.json \
  --inventory-sha256 "$INVENTORY_SHA256" \
  --challenge-id deployment-2026-08-17-001 \
  --output /safe/evidence/deployment-authorization.json

python -m lightcone_spec.runtime.offline_signer sign-control \
  --subject /safe/public/compile-control-subject.json \
  --deployment-authorization /safe/evidence/deployment-authorization.json \
  --hardware-envelope-sha256 "$HARDWARE_ENVELOPE_SHA256" \
  --attester-id release-signer \
  --key-id release-signer-key \
  --challenge-id compile-control-2026-08-17-001 \
  --output /safe/evidence/compile-control.json

python -m lightcone_spec.runtime.offline_signer sign-scientific \
  --artifact-type stage-materialization \
  --payload /safe/public/stage-materialization.json \
  --deployment-authorization /safe/evidence/deployment-authorization.json \
  --attester-id release-signer \
  --key-id release-signer-key \
  --challenge-id stage-materialization-2026-08-17-001 \
  --output /safe/evidence/stage-materialization.candidate.json

python -m lightcone_spec.runtime.offline_signer finalize-scientific \
  --artifact-type stage-materialization \
  --candidate /safe/evidence/stage-materialization.candidate.json \
  --deployment-authorization /safe/evidence/deployment-authorization.json \
  --challenge-ledger /safe/private/single-use-challenge-ledger \
  --output /safe/evidence/signed-stage-materialization.json
```

Every signing command creates a fresh challenge and publishes one canonical
file with no-replace semantics. Scientific signing is a closed, typed
two-phase ceremony: the first command accepts only a registered payload type;
the finalizer revalidates its deployment policy and reserves the challenge in
a private, single-use ledger before publishing the signed wrapper. It has no
generic JSON signing mode. For automation, pass only a numeric inherited descriptor
with `--key-fd`; never place a key or key path in argv. Copy only the public
authorization/control artifact to the execution workflow. Never copy the
private key or signer input FD to the GPU host.

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
and boundary-receipt schemas now have a first-party source producer. On the
supported single-tokenizer HTTP/1.1 uvicorn paths, continuous HTTP accounting
is measured at the real protocol lifecycle and bound to one process generation;
Granian HTTP/2 and multiple-tokenizer HTTP-process paths fail closed before
producing that capability. This covers reset-state accounting only; native
warm-up/trace/close receipts are absent and GPU semantics remain `PENDING`.
Shared-session execution is blocked before
mutation until GPU reset validation, durable session receipt binding, and
continuous whole-inventory accounting are available. The high-level block
executor uses a validated clean-process
fallback, with a distinct official HTTP pool and native provider for every
logical trace. The pool is reused only within that one single-trace execution.

Do not commit model/data payloads, experimental results, selected
hyperparameters derived from results, machine paths, credentials, or provider
metadata. A source checkout should remain clean except for deliberate code and
documentation changes.
