from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import formal_stage_execution as execution
from lightcone_spec.experiments.e6_stage_authority import (
    E6_MODELS,
    E6NextnModelAuthorityInput,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _Root:
    def __init__(self, root_sha256: str) -> None:
        self.deployment_policy_authorization = SimpleNamespace(
            root_manifest_sha256=root_sha256
        )


class _NativeToken:
    def __init__(
        self,
        suite_id: str,
        *,
        topology: str,
        gpu_uuids: tuple[str, ...],
        source: str,
        inventory: str,
    ) -> None:
        self.suite_id = suite_id
        self.topology_mode = topology
        self.gpu_uuids = gpu_uuids
        self.source_identity_sha256 = source
        self.inventory_sha256 = inventory
        self.hardware_envelope_sha256 = _sha("hardware")
        self.receipt_sha256 = _sha(f"native-receipt:{suite_id}")
        self.sha256 = _sha(f"native:{suite_id}")
        assert self.receipt_sha256 != self.sha256
        self.backend_capabilities = {
            "native_hot_path_tp1": ("graph_hot_path", "native_itl"),
            "dspark_tp1": ("dspark", "graph_hot_path", "native_itl"),
            "dspark_tp2": (
                "dspark",
                "graph_hot_path",
                "native_itl",
                "tp2_all_rank_publication",
            ),
            "dspark_dp2": (
                "dspark",
                "graph_hot_path",
                "native_itl",
                "dp2_sticky_isolation",
            ),
            "nextn_tp1": ("graph_hot_path", "native_itl", "nextn"),
            "nextn_tp2": ("graph_hot_path", "native_itl", "nextn"),
            "eagle3_tp1": ("eagle3", "graph_hot_path", "native_itl"),
        }[suite_id]


class _DistributedToken:
    def __init__(
        self,
        *,
        topology: str,
        gpu_uuids: tuple[str, str],
        source: str,
        inventory: str,
    ) -> None:
        self.topology_mode = topology
        self.gpu_uuids = gpu_uuids
        self.source_identity_sha256 = source
        self.inventory_sha256 = inventory
        self.hardware_envelope_sha256 = _sha("hardware")
        self.sha256 = _sha(f"distributed:{topology}")


def _install_fake_artifacts(monkeypatch, *, root: str, source: str, inventory: str):
    class NativeArtifact:
        def __init__(self, raw: dict[str, object]) -> None:
            self.raw = raw
            self.control_attestation = _Root(root)

        @classmethod
        def from_dict(cls, value: object):
            assert type(value) is dict
            return cls(value)

        def revalidate(self, *, now_ns: int):
            assert now_ns > 0
            return _NativeToken(
                str(self.raw["suite_id"]),
                topology=str(self.raw["topology"]),
                gpu_uuids=tuple(self.raw["gpu_uuids"]),
                source=source,
                inventory=inventory,
            )

    class DistributedArtifact:
        def __init__(self, raw: dict[str, object]) -> None:
            self.raw = raw
            self.control_attestation = _Root(root)

        @classmethod
        def from_dict(cls, value: object):
            assert type(value) is dict
            return cls(value)

        def revalidate(self, *, now_ns: int):
            assert now_ns > 0
            gpu_uuids = tuple(self.raw["gpu_uuids"])
            assert len(gpu_uuids) == 2
            return _DistributedToken(
                topology=str(self.raw["topology"]),
                gpu_uuids=gpu_uuids,  # type: ignore[arg-type]
                source=source,
                inventory=inventory,
            )

    monkeypatch.setattr(execution, "NativeRuntimeGpuProofArtifact", NativeArtifact)
    monkeypatch.setattr(
        execution, "DistributedRuntimeGpuProofArtifact", DistributedArtifact
    )
    monkeypatch.setattr(execution, "VerifiedNativeRuntimeGpuProof", _NativeToken)
    monkeypatch.setattr(
        execution, "VerifiedDistributedRuntimeGpuProof", _DistributedToken
    )


def _bindings(tmp_path, rows: tuple[tuple[str, str], ...], gpu_uuids):
    tmp_path.mkdir(parents=True, exist_ok=True)
    result = []
    for index, (kind, suite_or_topology) in enumerate(rows):
        path = tmp_path / f"proof-{index}.json"
        value = {
            "schema_version": 1,
            "kind": kind,
            "gpu_uuids": list(gpu_uuids),
        }
        if kind == "lightcone_native_runtime_gpu_proof_artifact":
            value.update(
                {
                    "suite_id": suite_or_topology,
                    "topology": (
                        "tp1_dp1"
                        if suite_or_topology.endswith("tp1")
                        or suite_or_topology == "native_hot_path_tp1"
                        else (
                            "tp1_dp2"
                            if suite_or_topology == "dspark_dp2"
                            else "tp2_dp1"
                        )
                    ),
                }
            )
        else:
            value["topology"] = suite_or_topology
        publish_canonical_json_no_replace(path, value)
        result.append(CanonicalJsonProofBinding.bind(path))
    return tuple(result)


def _revalidate(monkeypatch, tmp_path, *, algorithm, topology, suites):
    root = _sha("root")
    source = _sha("source")
    inventory = _sha("inventory")
    gpu_uuids = ("GPU-0",) if topology == "tp1_dp1" else ("GPU-0", "GPU-1")
    _install_fake_artifacts(
        monkeypatch,
        root=root,
        source=source,
        inventory=inventory,
    )
    subject = SimpleNamespace(
        runtime_gpu_proof_artifacts=_bindings(tmp_path, suites, gpu_uuids),
        inventory_sha256=inventory,
        gpu_uuids=gpu_uuids,
        topology_mode=topology,
    )
    lock = SimpleNamespace(
        offline_release_trust_root_sha256=root,
        native_runtime_qualification_source_identity_sha256=source,
    )
    config = SimpleNamespace(
        model=SimpleNamespace(algorithm=algorithm),
        adaptation=None,
    )
    return execution._revalidate_gpu_proofs(
        subject,
        protocol_lock=lock,
        run_config=config,
        now_ns=10,
    )


def test_dspark_tp1_requires_base_and_backend_native_suites(
    monkeypatch, tmp_path
) -> None:
    rows = (
        ("lightcone_native_runtime_gpu_proof_artifact", "native_hot_path_tp1"),
        ("lightcone_native_runtime_gpu_proof_artifact", "dspark_tp1"),
    )
    _ids, _hardware, native, distributed = _revalidate(
        monkeypatch,
        tmp_path,
        algorithm="DSPARK",
        topology="tp1_dp1",
        suites=rows,
    )
    assert tuple(row.suite_id for row in native) == (
        "dspark_tp1",
        "native_hot_path_tp1",
    )
    assert distributed == ()

    with pytest.raises(ValueError, match="native GPU proof coverage"):
        _revalidate(
            monkeypatch,
            tmp_path / "missing",
            algorithm="DSPARK",
            topology="tp1_dp1",
            suites=rows[:1],
        )


@pytest.mark.parametrize(
    ("algorithm", "topology", "native_suite"),
    (
        ("DSPARK", "tp2_dp1", "dspark_tp2"),
        ("DSPARK", "tp1_dp2", "dspark_dp2"),
        ("NEXTN", "tp2_dp1", "nextn_tp2"),
    ),
)
def test_distributed_backend_requires_exact_native_and_distributed_union(
    monkeypatch,
    tmp_path,
    algorithm,
    topology,
    native_suite,
) -> None:
    native_kind = "lightcone_native_runtime_gpu_proof_artifact"
    distributed_kind = "lightcone_distributed_runtime_gpu_proof_artifact"
    _revalidate(
        monkeypatch,
        tmp_path / "valid",
        algorithm=algorithm,
        topology=topology,
        suites=((native_kind, native_suite), (distributed_kind, topology)),
    )
    with pytest.raises(ValueError, match="native GPU proof coverage"):
        _revalidate(
            monkeypatch,
            tmp_path / "missing",
            algorithm=algorithm,
            topology=topology,
            suites=((distributed_kind, topology),),
        )
    foreign = "nextn_tp2" if native_suite != "nextn_tp2" else "dspark_tp2"
    with pytest.raises(ValueError, match="native GPU proof coverage"):
        _revalidate(
            monkeypatch,
            tmp_path / "foreign",
            algorithm=algorithm,
            topology=topology,
            suites=((native_kind, foreign), (distributed_kind, topology)),
        )


def test_nextn_tp2_joins_two_model_authority_to_live_proof_union(
    monkeypatch, tmp_path
) -> None:
    dynamic_path = tmp_path / "nextn-dynamic.json"
    publish_canonical_json_no_replace(
        dynamic_path,
        {"schema_version": 1, "kind": "test-nextn-dynamic-authority"},
    )
    interface = _sha("nextn-interface")
    topology_authority = _sha("nextn-topology")
    source = E6NextnModelAuthorityInput.bind(
        model=E6_MODELS[0],
        target_member_id="target-member",
        drafter_member_id="drafter-member",
        artifact_path=str(dynamic_path),
        expected_interface_sha256=interface,
        expected_topology_sha256=topology_authority,
        expected_source_adapter_version=3,
    )
    native = SimpleNamespace(suite_id="nextn_tp2", sha256=_sha("nextn-native"))
    distributed = SimpleNamespace(sha256=_sha("nextn-distributed"))
    content = SimpleNamespace(sha256=_sha("nextn-content"))
    verified_sha256 = _sha("nextn-verified")

    class VerifiedNextN:
        def __init__(self) -> None:
            self.sha256 = verified_sha256
            self.native_gpu_proof_sha256 = native.sha256
            self.distributed_gpu_proof_sha256 = distributed.sha256
            self.content_verification_receipt_sha256 = content.sha256
            self.gpu_uuids = ("GPU-0", "GPU-1")

    monkeypatch.setattr(execution, "VerifiedNextNTp2Authority", VerifiedNextN)
    monkeypatch.setattr(
        execution,
        "validate_nextn_tp2_dynamic_authority_artifact",
        lambda *_args, **_kwargs: VerifiedNextN(),
    )
    cell_id = _sha("nextn-cell")
    cell = SimpleNamespace(
        cell_id=cell_id,
        task="LiveCodeBench",
        backend="NEXTN",
        model=source.model,
        dimensions=(
            ("content_verification_receipt_sha256", content.sha256),
            ("distributed_gpu_proof_sha256", distributed.sha256),
            ("drafter_member_id", source.drafter_member_id),
            ("e6_verified_authority_sha256", verified_sha256),
            ("interface_sha256", interface),
            ("native_gpu_proof_sha256", native.sha256),
            ("source_adapter_version", 3),
            ("target_member_id", source.target_member_id),
            ("topology_authority_sha256", topology_authority),
        ),
    )
    subject = SimpleNamespace(
        topology_mode="tp2_dp1",
        materialized_cell_id=cell_id,
        inventory_sha256=_sha("inventory"),
        gpu_uuids=("GPU-0", "GPU-1"),
    )
    config = SimpleNamespace(
        model=SimpleNamespace(algorithm="NEXTN", target=source.model)
    )
    lock = SimpleNamespace(
        registry_sha256=_sha("registry"),
        offline_release_trust_root_sha256=_sha("root"),
    )
    materialization = SimpleNamespace(stage="E6", cells=(cell,))
    assert (
        type(
            execution._revalidate_nextn_tp2_authority(
                source_input=source,
                subject=subject,
                protocol_lock=lock,
                materialization=materialization,
                run_config=config,
                content_verification_receipt=content,
                native_gpu_proofs=(native,),
                distributed_gpu_proofs=(distributed,),
                now_ns=10,
            )
        )
        is VerifiedNextN
    )

    with pytest.raises(TypeError, match="requires its exact authority input"):
        execution._revalidate_nextn_tp2_authority(
            source_input=None,
            subject=subject,
            protocol_lock=lock,
            materialization=materialization,
            run_config=config,
            content_verification_receipt=content,
            native_gpu_proofs=(native,),
            distributed_gpu_proofs=(distributed,),
            now_ns=10,
        )
    with pytest.raises(ValueError, match="live proof/content union"):
        execution._revalidate_nextn_tp2_authority(
            source_input=source,
            subject=subject,
            protocol_lock=lock,
            materialization=materialization,
            run_config=config,
            content_verification_receipt=content,
            native_gpu_proofs=(
                SimpleNamespace(suite_id="nextn_tp2", sha256=_sha("foreign-native")),
            ),
            distributed_gpu_proofs=(distributed,),
            now_ns=10,
        )
