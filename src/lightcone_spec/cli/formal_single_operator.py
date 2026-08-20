"""CLI surface for the trusted ``formal_single_operator_v1`` workflow."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path


def add_formal_single_operator_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    command = commands.add_parser("formal-single-operator", allow_abbrev=False)
    operations = command.add_subparsers(
        dest="single_operator_operation",
        required=True,
    )

    status = operations.add_parser("status", allow_abbrev=False)
    status.set_defaults(_formal_single_operator=True)

    v03_model_lock = operations.add_parser(
        "publish-v03-model-lock",
        allow_abbrev=False,
    )
    v03_model_lock.add_argument("--output", required=True)
    v03_model_lock.set_defaults(_formal_single_operator=True)

    v03_e0_path_inputs = operations.add_parser(
        "write-v03-e0-raw-source-path-inputs",
        allow_abbrev=False,
    )
    v03_e0_path_inputs.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=ABSOLUTE_PATH",
    )
    v03_e0_path_inputs.add_argument("--output", required=True)
    v03_e0_path_inputs.set_defaults(_formal_single_operator=True)

    v03_e0_sources = operations.add_parser(
        "publish-v03-e0-source-authorities",
        allow_abbrev=False,
    )
    v03_e0_sources.add_argument("--inputs", required=True)
    v03_e0_sources.add_argument("--output-directory", required=True)
    v03_e0_sources.set_defaults(_formal_single_operator=True)

    v03_content_path_inputs = operations.add_parser(
        "write-v03-content-path-inputs",
        allow_abbrev=False,
    )
    v03_content_path_inputs.add_argument("--repository-root", required=True)
    v03_content_path_inputs.add_argument(
        "--model-snapshot",
        action="append",
        required=True,
        metavar="KEY=ABSOLUTE_DIRECTORY",
    )
    v03_content_path_inputs.add_argument("--livecodebench-raw", required=True)
    v03_content_path_inputs.add_argument("--math500-raw", required=True)
    v03_content_path_inputs.add_argument(
        "--burstgpt-asset",
        action="append",
        required=True,
        metavar="NAME=ABSOLUTE_PATH",
    )
    v03_content_path_inputs.add_argument(
        "--e0-source-authority",
        action="append",
        required=True,
        metavar="NAME=ABSOLUTE_PATH",
    )
    v03_content_path_inputs.add_argument("--inventory", required=True)
    v03_content_path_inputs.add_argument("--doctor-output", required=True)
    v03_content_path_inputs.add_argument(
        "--content-replay-authority-output",
        required=True,
    )
    v03_content_path_inputs.add_argument("--output", required=True)
    v03_content_path_inputs.set_defaults(_formal_single_operator=True)

    v03_content_spec = operations.add_parser(
        "publish-v03-content-path-spec",
        allow_abbrev=False,
    )
    v03_content_spec.add_argument("--inputs", required=True)
    v03_content_spec.add_argument("--output", required=True)
    v03_content_spec.set_defaults(_formal_single_operator=True)

    content_replay = operations.add_parser(
        "publish-content-replay-authority",
        allow_abbrev=False,
    )
    content_replay.add_argument("--spec", required=True)
    content_replay.add_argument("--output", required=True)
    content_replay.set_defaults(_formal_single_operator=True)

    stage_capacity = operations.add_parser(
        "publish-stage-capacity",
        allow_abbrev=False,
    )
    stage_capacity.add_argument("--content-path-spec", required=True)
    stage_capacity.add_argument("--run-root", required=True)
    stage_capacity.add_argument("--output", required=True)
    stage_capacity.set_defaults(_formal_single_operator=True)

    trusted_content = operations.add_parser(
        "publish-trusted-content",
        allow_abbrev=False,
    )
    trusted_content.add_argument("--spec", required=True)
    trusted_content.add_argument("--output", required=True)
    trusted_content.set_defaults(_formal_single_operator=True)

    trusted_workload = operations.add_parser(
        "publish-preflight-workload",
        allow_abbrev=False,
    )
    trusted_workload.add_argument("--content-source", required=True)
    trusted_workload.add_argument("--output", required=True)
    trusted_workload.set_defaults(_formal_single_operator=True)

    for name in (
        "publish-tts-cal-trainable-plan",
        "publish-e1-anchor-trainable-plan",
    ):
        trainable_plan = operations.add_parser(name, allow_abbrev=False)
        trainable_plan.add_argument("--trusted-content-bundle", required=True)
        trainable_plan.add_argument("--output", required=True)
        trainable_plan.set_defaults(_formal_single_operator=True)

    onlinespec_source = operations.add_parser(
        "publish-onlinespec-source-authority",
        allow_abbrev=False,
    )
    onlinespec_source.add_argument("--checkout", required=True)
    onlinespec_source.add_argument("--audit", required=True)
    onlinespec_source.add_argument("--output", required=True)
    onlinespec_source.set_defaults(_formal_single_operator=True)

    driver_config = operations.add_parser(
        "write-dag-driver-config",
        allow_abbrev=False,
    )
    driver_config.add_argument("--repository-root", required=True)
    driver_config.add_argument("--run-root", required=True)
    driver_config.add_argument("--protocol-lock", required=True)
    driver_config.add_argument("--content-source", required=True)
    driver_config.add_argument("--runtime-authority-manifest", required=True)
    driver_config.add_argument("--inventory", required=True)
    driver_config.add_argument("--doctor-report", required=True)
    driver_config.add_argument("--preflight-workload-authority", required=True)
    driver_config.add_argument(
        "--profiler-tool",
        action="append",
        default=[],
        metavar="ABSOLUTE_PATH",
    )
    driver_config.add_argument("--prerequisite-catalog", required=True)
    driver_config.add_argument("--session-reset-authority-directory")
    driver_config.add_argument("--output", required=True)
    driver_config.set_defaults(_formal_single_operator=True)

    bootstrap_config = operations.add_parser(
        "write-bootstrap-config",
        allow_abbrev=False,
    )
    bootstrap_config.add_argument("--driver-config", required=True)
    bootstrap_config.add_argument("--onlinespec-source-authority")
    bootstrap_config.add_argument("--output", required=True)
    bootstrap_config.set_defaults(_formal_single_operator=True)
    for name in ("bootstrap-once", "bootstrap-run"):
        bootstrap = operations.add_parser(name, allow_abbrev=False)
        bootstrap.add_argument("--config", required=True)
        bootstrap.set_defaults(_formal_single_operator=True)

    trusted_lock = operations.add_parser(
        "build-trusted-protocol-lock",
        allow_abbrev=False,
    )
    trusted_lock.add_argument("--protocol-id", required=True)
    trusted_lock.add_argument("--trusted-content-bundle", required=True)
    trusted_lock.add_argument("--runtime-authority-manifest", required=True)
    trusted_lock.add_argument("--tts-calibration-authority", required=True)
    trusted_lock.add_argument("--chronobelief-authority", required=True)
    trusted_lock.add_argument("--e1-recipe-anchor-authority", required=True)
    trusted_lock.add_argument("--output", required=True)
    trusted_lock.set_defaults(_formal_single_operator=True)

    materialize = operations.add_parser("materialize-node", allow_abbrev=False)
    materialize.add_argument("--node", required=True)
    materialize.add_argument("--predecessor-completion")
    materialize.add_argument("--protocol-lock")
    materialize.add_argument(
        "--content-source",
        help="root-only runtime-BOUND trusted content bundle",
    )
    materialize.add_argument(
        "--auxiliary-source",
        action="append",
        default=[],
        metavar="KIND=ABSOLUTE_PATH",
        help=(
            "repeatable exact scientific auxiliary; E6 uses e6_interface_fit "
            "and E0 uses e0_compatibility"
        ),
    )
    materialize.add_argument("--materialization-output", required=True)
    materialize.add_argument("--node-materialization-output", required=True)
    materialize.add_argument("--created-ns", type=int)
    materialize.set_defaults(_formal_single_operator=True)

    reduce_node = operations.add_parser("reduce-node", allow_abbrev=False)
    reduce_node.add_argument("--node-materialization", required=True)
    reduce_node.add_argument(
        "--actual",
        action="append",
        default=[],
        metavar="CELL_ID=PATH",
        help="one actual result path for each materialized cell",
    )
    reduce_node.add_argument("--repository-root")
    reduce_node.add_argument("--decision-output", required=True)
    reduce_node.add_argument("--completion-output", required=True)
    reduce_node.add_argument("--completed-ns", type=int)
    reduce_node.set_defaults(_formal_single_operator=True)

    execution_source = operations.add_parser(
        "publish-execution-source",
        allow_abbrev=False,
    )
    execution_source.add_argument("--node-materialization", required=True)
    execution_source.add_argument("--output", required=True)
    execution_source.set_defaults(_formal_single_operator=True)

    profiler_subject = operations.add_parser(
        "publish-profiler-subject",
        allow_abbrev=False,
    )
    profiler_subject.add_argument("--execution-source", required=True)
    profiler_subject.add_argument("--repository-root", required=True)
    profiler_subject.add_argument("--output", required=True)
    profiler_subject.set_defaults(_formal_single_operator=True)

    preflight_actual = operations.add_parser(
        "publish-preflight-actual",
        allow_abbrev=False,
    )
    preflight_actual.add_argument("--final-evidence", required=True)
    preflight_actual.add_argument("--protocol-lock", required=True)
    preflight_actual.add_argument("--output", required=True)
    preflight_actual.add_argument("--started-ns", type=int, required=True)
    preflight_actual.add_argument("--finished-ns", type=int, required=True)
    preflight_actual.add_argument("--verified-ns", type=int)
    preflight_actual.set_defaults(_formal_single_operator=True)

    preflight_inputs = operations.add_parser(
        "build-preflight-inputs",
        allow_abbrev=False,
    )
    preflight_inputs.add_argument("--execution-source", required=True)
    preflight_inputs.add_argument("--repository-root", required=True)
    preflight_inputs.add_argument("--runtime-authority-manifest", required=True)
    preflight_inputs.add_argument("--inventory", required=True)
    preflight_content = preflight_inputs.add_mutually_exclusive_group(required=True)
    preflight_content.add_argument("--content-verification-receipt")
    preflight_content.add_argument("--content-source")
    preflight_inputs.add_argument("--workload-authority", required=True)
    preflight_inputs.add_argument("--doctor-report", required=True)
    preflight_inputs.add_argument("--private-output-root", required=True)
    preflight_inputs.add_argument("--current-ns", type=int)
    preflight_inputs.set_defaults(_formal_single_operator=True)

    preflight_completion = operations.add_parser(
        "publish-preflight-completion",
        allow_abbrev=False,
    )
    preflight_completion.add_argument("--execution-inputs", required=True)
    preflight_completion.add_argument("--compile-result", required=True)
    preflight_completion.add_argument("--exactness-result", required=True)
    for name in (
        "interference-terminal",
        "interference-lifecycle",
        "interference-junit",
    ):
        preflight_completion.add_argument(
            f"--{name}",
            action="append",
            default=[],
            required=True,
            metavar="CELL_ID=PATH",
        )
    preflight_completion.add_argument("--output", required=True)
    preflight_completion.add_argument("--current-ns", type=int)
    preflight_completion.set_defaults(_formal_single_operator=True)

    execute_preflight = operations.add_parser(
        "execute-preflight",
        allow_abbrev=False,
    )
    execute_preflight.add_argument("--execution-inputs", required=True)
    execute_preflight.add_argument("--current-ns", type=int)
    execute_preflight.set_defaults(_formal_single_operator=True)

    reduce_preflight = operations.add_parser(
        "reduce-preflight",
        allow_abbrev=False,
    )
    reduce_preflight.add_argument("--node-materialization", required=True)
    reduce_preflight.add_argument("--preflight-execution", required=True)
    reduce_preflight.add_argument("--repository-root", required=True)
    reduce_preflight.add_argument("--decision-output", required=True)
    reduce_preflight.add_argument("--completion-output", required=True)
    reduce_preflight.add_argument("--completed-ns", type=int)
    reduce_preflight.set_defaults(_formal_single_operator=True)

    pre_hours = operations.add_parser("gpu-hours-pre", allow_abbrev=False)
    pre_hours.add_argument("--materialization", required=True)
    pre_hours.add_argument("--output", required=True)
    pre_hours.set_defaults(_formal_single_operator=True)

    post_hours = operations.add_parser("gpu-hours-post", allow_abbrev=False)
    post_hours.add_argument("--repository-root", required=True)
    post_hours.add_argument("--pilot-materialization", required=True)
    post_hours.add_argument("--final-materialization")
    post_hours.add_argument("--inventory", required=True)
    post_hours.add_argument(
        "--run-manifest",
        action="append",
        default=[],
        help="legacy fresh-process manifest input; repeat per pilot cell",
    )
    post_hours.add_argument(
        "--actual-result",
        action="append",
        default=[],
        help=(
            "fresh or resident root actual result; repeat per pilot cell and use "
            "for mixed single-charge accounting"
        ),
    )
    post_hours.add_argument("--source-output", required=True)
    post_hours.add_argument("--output", required=True)
    post_hours.set_defaults(_formal_single_operator=True)

    prepare_e6_interface = operations.add_parser(
        "prepare-e6-interface-fit",
        allow_abbrev=False,
    )
    prepare_e6_interface.add_argument("--protocol-lock", required=True)
    prepare_e6_interface.add_argument(
        "--predecessor-completion",
        required=True,
    )
    prepare_e6_interface.add_argument("--content-source", required=True)
    prepare_e6_interface.add_argument(
        "--launch",
        action="append",
        default=[],
        required=True,
        metavar="MODEL=ABSOLUTE_PATH",
    )
    prepare_e6_interface.add_argument("--output-root", required=True)
    prepare_e6_interface.set_defaults(_formal_single_operator=True)

    execute_e6_interface = operations.add_parser(
        "execute-e6-interface-fit",
        allow_abbrev=False,
    )
    execute_e6_interface.add_argument("--plan", required=True)
    execute_e6_interface.set_defaults(_formal_single_operator=True)

    finalize_e6_interface = operations.add_parser(
        "finalize-e6-interface-fit",
        allow_abbrev=False,
    )
    finalize_e6_interface.add_argument("--campaign", required=True)
    finalize_e6_interface.add_argument("--output", required=True)
    finalize_e6_interface.set_defaults(_formal_single_operator=True)

    prepare_run = operations.add_parser("prepare-run", allow_abbrev=False)
    prepare_run.add_argument("--repository-root", required=True)
    prepare_run.add_argument("--execution-source", required=True)
    prepare_run.add_argument("--cell", required=True)
    prepare_run.add_argument(
        "--preflight-inputs",
        help="exact retained preflight source required by every serving/failure plan",
    )
    prepare_run.add_argument(
        "--prepared-launch-bundle",
        help=(
            "post-materialization source-owned launch envelope; required for "
            "E4 profiler and later serving/failure cells"
        ),
    )
    prepare_run.add_argument(
        "--profiler-tool",
        help="absolute onsite nsys/ncu path; required only for E4 profiler cells",
    )
    prepare_run.add_argument("--output-root", required=True)
    prepare_run.set_defaults(_formal_single_operator=True)

    execute_run = operations.add_parser("execute-run", allow_abbrev=False)
    execute_run.add_argument("--repository-root", required=True)
    execute_run.add_argument("--run-plan", required=True)
    execute_run.set_defaults(_formal_single_operator=True)

    finalize = operations.add_parser("finalize-run", allow_abbrev=False)
    finalize.add_argument("--repository-root", required=True)
    finalize.add_argument("--run-plan", required=True)
    finalize.add_argument(
        "--execution-source",
        help="legacy schema-1 compatibility; schema-2 plans carry this source",
    )
    finalize.add_argument(
        "--inventory",
        help="legacy schema-1 compatibility; schema-2 plans carry this source",
    )
    finalize.set_defaults(_formal_single_operator=True)


def _timestamp(value: int | None) -> int:
    observed = time.time_ns() if value is None else value
    if type(observed) is not int or observed < 1:
        raise ValueError("single-operator timestamp must be a positive integer")
    return observed


def _named_paths(values: list[str], *, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if separator != "=" or not name or not raw_path or name in result:
            raise ValueError(
                f"single-operator {label} rows must be unique NAME=ABSOLUTE_PATH values"
            )
        path = Path(raw_path)
        if not path.is_absolute() or path != path.resolve(strict=False):
            raise ValueError(f"single-operator {label} paths must be absolute")
        result[name] = str(path)
    return result


def _actual_paths(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        cell_id, separator, raw_path = value.partition("=")
        if separator != "=" or not cell_id or not raw_path or cell_id in result:
            raise ValueError(
                "single-operator --actual values must be unique CELL_ID=PATH rows"
            )
        path = Path(raw_path)
        if not path.is_absolute() or path != path.resolve(strict=False):
            raise ValueError("single-operator actual paths must be absolute")
        result[cell_id] = str(path)
    if not result:
        raise ValueError("single-operator reduction requires actual result paths")
    return result


def _auxiliary_paths(values: list[str]) -> dict[str, str]:
    allowed = {"e6_interface_fit", "e0_compatibility"}
    result: dict[str, str] = {}
    for value in values:
        kind, separator, raw_path = value.partition("=")
        if separator != "=" or kind not in allowed or not raw_path or kind in result:
            raise ValueError(
                "single-operator --auxiliary-source rows must be unique "
                "registered KIND=ABSOLUTE_PATH values"
            )
        path = Path(raw_path)
        if not path.is_absolute() or path != path.resolve(strict=False):
            raise ValueError("single-operator auxiliary source paths must be absolute")
        result[kind] = str(path)
    return result


def _e6_launch_paths(values: list[str]) -> dict[str, str]:
    from lightcone_spec.experiments.formal_protocol import E6_MODELS

    result: dict[str, str] = {}
    for value in values:
        model, separator, raw_path = value.partition("=")
        if (
            separator != "="
            or model not in E6_MODELS
            or not raw_path
            or model in result
        ):
            raise ValueError(
                "E6 --launch rows must cover each exact MODEL=ABSOLUTE_PATH once"
            )
        path = Path(raw_path)
        if not path.is_absolute() or path != path.resolve(strict=False):
            raise ValueError("E6 launch manifest paths must be absolute")
        result[model] = str(path)
    if tuple(model for model in E6_MODELS if model in result) != E6_MODELS:
        raise ValueError("E6 --launch rows must cover both exact NEXTN models")
    return result


def _load_materialization(path: str):
    from lightcone_spec.experiments.formal_registry import (
        stage_materialization_receipt_from_dict,
    )
    from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

    binding = CanonicalJsonProofBinding.bind(path)
    materialization = stage_materialization_receipt_from_dict(binding.reopen())
    return materialization


def _load_inventory(path: str):
    from lightcone_spec.experiments.gpu_pool import GpuInventory
    from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

    binding = CanonicalJsonProofBinding.bind(path)
    inventory = GpuInventory.from_dict(binding.reopen())
    return inventory


def handle_formal_single_operator_command(args: argparse.Namespace) -> int | None:
    if not getattr(args, "_formal_single_operator", False):
        return None

    from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
    from lightcone_spec.experiments.formal_single_operator_gpu_hours import (
        derive_formal_single_operator_post_pilot_gpu_hours_from_run_manifests,
        derive_formal_single_operator_post_pilot_gpu_hours_from_serving_actuals,
        derive_formal_single_operator_pre_pilot_gpu_hours,
        publish_formal_single_operator_gpu_hours,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        formal_single_operator_node_readiness,
        materialize_formal_single_operator_node,
        publish_formal_single_operator_execution_source,
        publish_formal_single_operator_preflight_actual,
        reduce_formal_single_operator_node,
    )
    from lightcone_spec.runtime.formal_single_operator import (
        FORMAL_SINGLE_OPERATOR_MODE,
        FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256,
        finalize_formal_single_operator_run,
    )
    from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

    operation = args.single_operator_operation
    if operation == "publish-v03-model-lock":
        from lightcone_spec.experiments.formal_single_operator_model_registry import (
            publish_formal_v03_model_lock,
        )

        output = str(Path(args.output).resolve())
        lock = publish_formal_v03_model_lock(output_path=output)
        print(
            json.dumps(
                {
                    "model_count": len(lock.models),
                    "path": output,
                    "sha256": lock.sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "write-v03-e0-raw-source-path-inputs":
        from lightcone_spec.experiments.formal_single_operator_model_registry import (
            publish_formal_v03_e0_raw_source_path_inputs,
        )

        binding = publish_formal_v03_e0_raw_source_path_inputs(
            source_paths=_named_paths(args.source, label="E0 raw-source"),
            output_path=args.output,
        )
        print(json.dumps(binding.to_dict(), sort_keys=True))
        return 0
    if operation == "publish-v03-e0-source-authorities":
        from lightcone_spec.experiments.formal_single_operator_model_registry import (
            publish_formal_v03_e0_source_authorities_from_inputs,
        )

        index = publish_formal_v03_e0_source_authorities_from_inputs(
            inputs_path=args.inputs,
            output_directory=args.output_directory,
        )
        print(
            json.dumps(
                {
                    "index_path": index.absolute_path,
                    "raw_sha256": index.raw_sha256,
                    "semantic_sha256": index.semantic_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "write-v03-content-path-inputs":
        from lightcone_spec.experiments.formal_single_operator_model_registry import (
            publish_formal_v03_content_path_inputs,
        )

        binding = publish_formal_v03_content_path_inputs(
            repository_root=args.repository_root,
            model_snapshot_paths=_named_paths(
                args.model_snapshot,
                label="model-snapshot",
            ),
            livecodebench_raw_path=args.livecodebench_raw,
            math500_raw_path=args.math500_raw,
            burstgpt_asset_paths=_named_paths(
                args.burstgpt_asset,
                label="BurstGPT asset",
            ),
            e0_source_authority_paths=_named_paths(
                args.e0_source_authority,
                label="E0 source-authority",
            ),
            inventory_path=args.inventory,
            doctor_output_path=args.doctor_output,
            content_replay_authority_output_path=(args.content_replay_authority_output),
            output_path=args.output,
        )
        print(json.dumps(binding.to_dict(), sort_keys=True))
        return 0
    if operation == "publish-v03-content-path-spec":
        from lightcone_spec.experiments.formal_protocol import content_sha256
        from lightcone_spec.experiments.formal_single_operator_model_registry import (
            publish_formal_v03_content_path_spec_from_inputs,
        )

        spec = publish_formal_v03_content_path_spec_from_inputs(
            inputs_path=args.inputs,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "path": str(Path(args.output)),
                    "semantic_sha256": content_sha256(spec.to_dict()),
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "publish-content-replay-authority":
        from lightcone_spec.experiments.formal_single_operator_content import (
            publish_trusted_single_operator_content_replay_authority_from_spec,
        )

        binding = publish_trusted_single_operator_content_replay_authority_from_spec(
            spec_path=args.spec,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "path": binding.absolute_path,
                    "semantic_sha256": binding.semantic_sha256,
                    "protocol_sha256": binding.protocol_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "publish-stage-capacity":
        from lightcone_spec.experiments.formal_single_operator_capacity import (
            publish_trusted_single_operator_stage_capacity_authority,
        )

        authority = publish_trusted_single_operator_stage_capacity_authority(
            content_path_spec_path=args.content_path_spec,
            run_root_path=args.run_root,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "path": str(Path(args.output)),
                    "authority_sha256": authority.sha256,
                    "status": authority.status,
                    "formal_measured_authorization": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "publish-onlinespec-source-authority":
        from lightcone_spec.experiments.formal_registry import (
            publish_e0_onlinespec_source_authority,
        )

        binding = publish_e0_onlinespec_source_authority(
            checkout_path=args.checkout,
            audit_path=args.audit,
            output_path=args.output,
        )
        print(
            json.dumps(
                {"semantic_sha256": binding.semantic_sha256},
                sort_keys=True,
            )
        )
        return 0
    if operation == "write-dag-driver-config":
        from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
            publish_path_bound_formal_dag_driver_config,
        )

        config = publish_path_bound_formal_dag_driver_config(
            repository_root=args.repository_root,
            run_root=args.run_root,
            protocol_lock_path=args.protocol_lock,
            content_source_path=args.content_source,
            runtime_authority_manifest_path=args.runtime_authority_manifest,
            inventory_path=args.inventory,
            doctor_report_path=args.doctor_report,
            preflight_workload_authority_path=(args.preflight_workload_authority),
            profiler_tool_paths=tuple(args.profiler_tool),
            prerequisite_index_catalog_directory=args.prerequisite_catalog,
            session_reset_authority_directory=(args.session_reset_authority_directory),
            output_path=args.output,
        )
        print(json.dumps({"config_sha256": config.sha256}, sort_keys=True))
        return 0
    if operation == "write-bootstrap-config":
        from lightcone_spec.orchestration.formal_single_operator_bootstrap import (
            publish_path_bound_formal_bootstrap_config,
        )

        config = publish_path_bound_formal_bootstrap_config(
            driver_config_path=args.driver_config,
            onlinespec_source_authority_path=args.onlinespec_source_authority,
            output_path=args.output,
        )
        print(json.dumps({"config_sha256": config.sha256}, sort_keys=True))
        return 0
    if operation in {"bootstrap-once", "bootstrap-run"}:
        from lightcone_spec.orchestration.formal_single_operator_bootstrap import (
            FormalSingleOperatorBootstrapSupervisor,
        )

        supervisor = FormalSingleOperatorBootstrapSupervisor(args.config)
        try:
            cycle = (
                supervisor.run_once()
                if operation == "bootstrap-once"
                else supervisor.run_until_event()
            )
        finally:
            supervisor.close()
        print(json.dumps(cycle.to_dict(), sort_keys=True))
        if cycle.controller_action == "COMPLETE":
            return 43
        return 42 if cycle.controller_action == "BLOCKED" else 0
    if operation == "publish-trusted-content":
        from lightcone_spec.experiments.formal_single_operator_content import (
            publish_runtime_bound_trusted_single_operator_content_from_spec,
        )

        binding = publish_runtime_bound_trusted_single_operator_content_from_spec(
            spec_path=args.spec,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "path": binding.absolute_path,
                    "raw_sha256": binding.raw_sha256,
                    "runtime_binding_status": binding.runtime_binding_status,
                    "semantic_sha256": binding.semantic_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "publish-preflight-workload":
        from lightcone_spec.experiments.formal_single_operator_content import (
            publish_trusted_preflight_workload_authority_from_content,
        )

        binding = publish_trusted_preflight_workload_authority_from_content(
            trusted_content_bundle_path=args.content_source,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "path": binding.absolute_path,
                    "raw_sha256": binding.raw_sha256,
                    "semantic_sha256": binding.semantic_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation in {
        "publish-tts-cal-trainable-plan",
        "publish-e1-anchor-trainable-plan",
    }:
        from lightcone_spec.experiments.formal_single_operator_trainable_plan import (
            publish_trusted_e1_recipe_anchor_trainable_plan_authority,
            publish_trusted_tts_calibration_trainable_plan_authority,
        )

        publisher = (
            publish_trusted_tts_calibration_trainable_plan_authority
            if operation == "publish-tts-cal-trainable-plan"
            else publish_trusted_e1_recipe_anchor_trainable_plan_authority
        )
        binding = publisher(
            trusted_content_bundle_path=args.trusted_content_bundle,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "cell_id": binding.cell_id,
                    "path": str(Path(args.output)),
                    "semantic_sha256": binding.sha256,
                    "trainable_plan_sha256": binding.trainable_plan_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "build-trusted-protocol-lock":
        from lightcone_spec.experiments.formal_single_operator_protocol_lock import (
            build_trusted_single_operator_protocol_lock,
            publish_trusted_single_operator_protocol_lock,
        )

        lock = build_trusted_single_operator_protocol_lock(
            protocol_id=args.protocol_id,
            trusted_content_bundle_path=args.trusted_content_bundle,
            formal_runtime_authority_manifest_path=(args.runtime_authority_manifest),
            tts_calibration_authority_path=args.tts_calibration_authority,
            chronobelief_authority_path=args.chronobelief_authority,
            e1_recipe_anchor_authority_path=args.e1_recipe_anchor_authority,
        )
        binding = publish_trusted_single_operator_protocol_lock(lock, args.output)
        print(
            json.dumps(
                {
                    "content_source_mode": lock.content_source_mode,
                    "path": binding.absolute_path,
                    "protocol_lock_sha256": lock.sha256,
                    "schema_version": lock.schema_version,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "status":
        from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
            formal_single_operator_dag_code_capabilities,
        )

        capabilities = {
            capability.node: capability
            for capability in formal_single_operator_dag_code_capabilities()
        }
        nodes = []
        for readiness in formal_single_operator_node_readiness():
            row = asdict(readiness)
            structural_ready = row["status"] == "READY"
            capability = capabilities[readiness.node]
            executable = structural_ready and capability.ready
            row["structural_status"] = row["status"]
            row["code_materializer_available"] = capability.materializer
            row["producer_available"] = capability.producer
            row["physical_mapper_available"] = capability.mapper
            row["executor_available"] = capability.executor
            row["finalizer_available"] = capability.finalizer
            row["code_capability_ready"] = capability.ready
            row["status"] = "READY" if executable else "BLOCKED"
            if not executable:
                if not structural_ready:
                    blocker = row["blocker"] or "structural_stage_adapter_unavailable"
                else:
                    blocker = capability.blocker or "code_capability_unavailable"
                row["blocker"] = blocker
            nodes.append(row)
        print(
            json.dumps(
                {
                    "mode": FORMAL_SINGLE_OPERATOR_MODE,
                    "protocol_sha256": FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256,
                    "readiness_scope": "code_capability_only",
                    "nodes": nodes,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "materialize-node":
        rebuilt = materialize_formal_single_operator_node(
            node=args.node,
            predecessor_completion_path=args.predecessor_completion,
            protocol_lock_path=args.protocol_lock,
            content_source_path=args.content_source,
            materialization_output_path=args.materialization_output,
            node_materialization_output_path=args.node_materialization_output,
            created_ns=_timestamp(args.created_ns),
            auxiliary_source_paths=_auxiliary_paths(args.auxiliary_source),
        )
        print(rebuilt.artifact.sha256)
        return 0
    if operation == "reduce-node":
        rebuilt = reduce_formal_single_operator_node(
            node_materialization_path=args.node_materialization,
            actual_result_paths=_actual_paths(args.actual),
            repository_root=args.repository_root,
            decision_output_path=args.decision_output,
            completion_output_path=args.completion_output,
            completed_ns=_timestamp(args.completed_ns),
        )
        print(rebuilt.artifact.sha256)
        return 0
    if operation == "publish-execution-source":
        source = publish_formal_single_operator_execution_source(
            node_materialization_path=args.node_materialization,
            output_path=args.output,
        )
        print(source.sha256)
        return 0
    if operation == "publish-profiler-subject":
        from lightcone_spec.experiments.formal_single_operator_profiler_subject_producer import (
            publish_formal_single_operator_profiler_subject_requirement,
        )

        requirement = publish_formal_single_operator_profiler_subject_requirement(
            execution_source_path=args.execution_source,
            repository_root=args.repository_root,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "path": str(Path(args.output)),
                    "requirement_sha256": requirement.sha256,
                    "selected_configuration_sha256": (
                        requirement.selected_configuration_sha256
                    ),
                    "source_headline_cell_id": (requirement.source_headline_cell_id),
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "publish-preflight-actual":
        lock_binding = CanonicalJsonProofBinding.bind(args.protocol_lock)
        lock = protocol_lock_from_dict(lock_binding.reopen())
        if lock.sha256 != lock_binding.semantic_sha256:
            raise ValueError("single-operator ProtocolLock digest differs")
        actual = publish_formal_single_operator_preflight_actual(
            final_evidence_source_path=args.final_evidence,
            protocol_lock=lock,
            verified_ns=_timestamp(args.verified_ns),
            started_ns=args.started_ns,
            finished_ns=args.finished_ns,
            output_path=args.output,
        )
        print(actual.sha256)
        return 0
    if operation == "build-preflight-inputs":
        from lightcone_spec.experiments.formal_preflight_inputs import (
            materialize_formal_single_operator_preflight_execution_inputs,
        )

        binding = materialize_formal_single_operator_preflight_execution_inputs(
            execution_source_path=args.execution_source,
            repository_root=args.repository_root,
            formal_runtime_authority_manifest_path=(args.runtime_authority_manifest),
            inventory_path=args.inventory,
            content_verification_receipt_path=(args.content_verification_receipt),
            content_source_path=args.content_source,
            workload_authority_path=args.workload_authority,
            doctor_report_path=args.doctor_report,
            private_output_root=args.private_output_root,
            current_ns=_timestamp(args.current_ns),
        )
        print(binding.semantic_sha256)
        return 0
    if operation == "publish-preflight-completion":
        from lightcone_spec.experiments.formal_preflight_execution import (
            FormalPreflightInterferenceExecutionManifest,
        )
        from lightcone_spec.experiments.formal_preflight_inputs import (
            load_formal_preflight_execution_inputs,
            publish_formal_single_operator_preflight_completion,
        )

        inputs = load_formal_preflight_execution_inputs(args.execution_inputs)
        manifest = FormalPreflightInterferenceExecutionManifest.from_dict(
            inputs.interference_manifest.reopen()
        )
        registry_cell_ids = tuple(row.registry_cell_id for row in manifest.inputs)
        sources = {
            "terminal": _actual_paths(args.interference_terminal),
            "lifecycle": _actual_paths(args.interference_lifecycle),
            "junit": _actual_paths(args.interference_junit),
        }
        expected = set(registry_cell_ids)
        if any(set(rows) != expected for rows in sources.values()):
            raise ValueError(
                "single-operator preflight completion requires exact eight CELL_ID paths"
            )
        binding = publish_formal_single_operator_preflight_completion(
            execution_inputs_path=args.execution_inputs,
            compile_result_path=args.compile_result,
            exactness_result_path=args.exactness_result,
            interference_terminal_result_proof_paths=tuple(
                sources["terminal"][cell_id] for cell_id in registry_cell_ids
            ),
            interference_lifecycle_timing_paths=tuple(
                sources["lifecycle"][cell_id] for cell_id in registry_cell_ids
            ),
            interference_junit_paths=tuple(
                sources["junit"][cell_id] for cell_id in registry_cell_ids
            ),
            output_path=args.output,
            current_ns=_timestamp(args.current_ns),
        )
        print(binding.semantic_sha256)
        return 0
    if operation == "execute-preflight":
        import asyncio

        from lightcone_spec.experiments.formal_preflight_inputs import (
            execute_formal_single_operator_preflight_exact_ten,
            revalidate_formal_single_operator_preflight_exact_ten_execution,
        )

        current_ns = _timestamp(args.current_ns)
        binding = asyncio.run(
            execute_formal_single_operator_preflight_exact_ten(
                args.execution_inputs,
                current_ns=current_ns,
            )
        )
        execution = revalidate_formal_single_operator_preflight_exact_ten_execution(
            binding.absolute_path,
            current_ns=current_ns,
        )
        print(
            json.dumps(
                {
                    "completion_path": (
                        None
                        if execution.completion is None
                        else execution.completion.absolute_path
                    ),
                    "execution_path": binding.absolute_path,
                    "execution_sha256": execution.sha256,
                    "status": execution.status,
                },
                sort_keys=True,
            )
        )
        return 0 if execution.status == "COMPLETE" else 42
    if operation == "reduce-preflight":
        from lightcone_spec.experiments.formal_preflight_inputs import (
            revalidate_formal_single_operator_preflight_exact_ten_execution,
        )
        from lightcone_spec.experiments.formal_single_operator_stages import (
            rebuild_formal_single_operator_node_materialization,
        )

        completed_ns = _timestamp(args.completed_ns)
        execution = revalidate_formal_single_operator_preflight_exact_ten_execution(
            args.preflight_execution,
            current_ns=completed_ns,
        )
        if execution.status != "COMPLETE" or execution.completion is None:
            raise RuntimeError(
                "single-operator preflight reduction requires COMPLETE exact ten"
            )
        node = rebuild_formal_single_operator_node_materialization(
            args.node_materialization
        )
        if node.artifact.node != "preflight":
            raise ValueError(
                "single-operator preflight reduction names another DAG node"
            )
        rebuilt = reduce_formal_single_operator_node(
            node_materialization_path=args.node_materialization,
            actual_result_paths={
                cell.cell_id: execution.completion.absolute_path
                for cell in node.materialization.cells
            },
            repository_root=args.repository_root,
            decision_output_path=args.decision_output,
            completion_output_path=args.completion_output,
            completed_ns=completed_ns,
        )
        print(rebuilt.artifact.sha256)
        return 0
    if operation == "gpu-hours-pre":
        output = derive_formal_single_operator_pre_pilot_gpu_hours(
            _load_materialization(args.materialization)
        )
        publish_formal_single_operator_gpu_hours(output, args.output)
        print(output.sha256)
        return 0
    if operation == "gpu-hours-post":
        if bool(args.run_manifest) == bool(args.actual_result):
            raise ValueError(
                "gpu-hours-post requires exactly one of --run-manifest or "
                "--actual-result"
            )
        common = {
            "repository_root": args.repository_root,
            "pilot_materialization": _load_materialization(args.pilot_materialization),
            "final_materialization": (
                None
                if args.final_materialization is None
                else _load_materialization(args.final_materialization)
            ),
            "inventory": _load_inventory(args.inventory),
            "source_manifest_output_path": args.source_output,
        }
        if args.actual_result:
            output = (
                derive_formal_single_operator_post_pilot_gpu_hours_from_serving_actuals(
                    **common,
                    pilot_actual_result_paths=tuple(args.actual_result),
                )
            )
        else:
            output = (
                derive_formal_single_operator_post_pilot_gpu_hours_from_run_manifests(
                    **common,
                    pilot_run_manifest_paths=tuple(args.run_manifest),
                )
            )
        publish_formal_single_operator_gpu_hours(output, args.output)
        print(output.sha256)
        return 0
    if operation == "prepare-e6-interface-fit":
        from lightcone_spec.experiments.formal_single_operator_e6_interface import (
            materialize_formal_single_operator_e6_interface_fit_campaign,
        )

        campaign = materialize_formal_single_operator_e6_interface_fit_campaign(
            protocol_lock_path=args.protocol_lock,
            predecessor_completion_path=args.predecessor_completion,
            trusted_content_bundle_path=args.content_source,
            launch_manifest_paths=_e6_launch_paths(args.launch),
            output_root=args.output_root,
        )
        print(
            json.dumps(
                {
                    "campaign": str(
                        Path(args.output_root) / "e6-interface-fit-campaign.json"
                    ),
                    "campaign_sha256": campaign.sha256,
                    "exclusive_dual_gpu": True,
                    "gpu_uuids": list(campaign.gpu_uuids),
                    "models": list(campaign.models),
                    "physical_run_count": campaign.physical_run_count,
                    "plans": [row.absolute_path for row in campaign.plans],
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "execute-e6-interface-fit":
        from lightcone_spec.experiments.formal_single_operator_e6_interface import (
            execute_formal_single_operator_e6_interface_fit_plan,
        )

        terminal = execute_formal_single_operator_e6_interface_fit_plan(args.plan)
        print(
            json.dumps(
                {
                    "model": terminal.model,
                    "exclusive_dual_gpu": True,
                    "physical_execution_count": (terminal.physical_execution_count),
                    "status": terminal.status,
                    "terminal": str(
                        Path(args.plan).parent / "e6-interface-fit-terminal.json"
                    ),
                    "terminal_sha256": terminal.sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "finalize-e6-interface-fit":
        from lightcone_spec.experiments.formal_single_operator_e6_interface import (
            finalize_formal_single_operator_e6_interface_fit_bundle,
        )

        bundle = finalize_formal_single_operator_e6_interface_fit_bundle(
            campaign_path=args.campaign,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "bundle": str(Path(args.output)),
                    "bundle_sha256": bundle.sha256,
                    "models": list(bundle.models),
                    "physical_execution_count": bundle.physical_execution_count,
                    "reuse_scope": bundle.reuse_scope,
                    "trust_mode": bundle.trust_mode,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "prepare-run":
        from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
            FormalSingleOperatorDispatchBlocked,
            materialize_formal_single_operator_e4_direct_run_plan_inputs,
            materialize_formal_single_operator_prepared_downstream_run_plan_inputs,
            revalidate_formal_single_operator_e0_compatibility_decision,
            route_formal_single_operator_cell,
        )
        from lightcone_spec.runtime.formal_single_operator import (
            create_formal_single_operator_run_directory,
        )

        started_ns = time.time_ns()
        source, _cell, route = route_formal_single_operator_cell(
            execution_source_path=args.execution_source,
            materialized_cell_id=args.cell,
        )
        early = source.node in {
            "e3a",
            "tts_cal",
            "e1",
            "e2_r0",
            "e2_r1",
            "e2_r2",
            "e2_r3",
        }
        e4_headline = source.node in {"e4_screen", "e4_local"} and (
            route.physical_kind == "serving"
        )
        prepared_serving = (
            not early and not e4_headline and route.physical_kind == "serving"
        )
        profiler = route.physical_kind == "profiler"
        e5_failure = route.physical_kind == "e5_failure"
        e6_interface = route.physical_kind == "e6_interface_preflight"
        if (
            early or e4_headline or prepared_serving or e5_failure
        ) and args.preflight_inputs is None:
            raise ValueError("serving/failure preparation requires --preflight-inputs")
        if (prepared_serving or e5_failure) and args.prepared_launch_bundle is None:
            from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
                FormalSingleOperatorPreparedLaunchBlocked,
            )

            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_prepared_launch_bundle_missing"
            )
        if profiler:
            if args.prepared_launch_bundle is None:
                from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
                    FormalSingleOperatorPreparedLaunchBlocked,
                )

                raise FormalSingleOperatorPreparedLaunchBlocked(
                    "source_owned_prepared_launch_bundle_missing"
                )
            if args.profiler_tool is None:
                raise ValueError("E4 profiler preparation requires --profiler-tool")
        elif args.profiler_tool is not None:
            raise ValueError("--profiler-tool is valid only for E4 profiler cells")
        if (
            not early
            and not e4_headline
            and not prepared_serving
            and not profiler
            and not e5_failure
            and not e6_interface
        ):
            if route.physical_kind == "e0_compatibility_decision":
                decision = revalidate_formal_single_operator_e0_compatibility_decision(
                    execution_source_path=args.execution_source,
                    materialized_cell_id=args.cell,
                )
                print(
                    json.dumps(
                        {
                            "cell_id": args.cell,
                            "compatibility_evidence": (decision.evidence.absolute_path),
                            "compatibility_evidence_sha256": (
                                decision.evidence.semantic_sha256
                            ),
                            "decision_id": decision.decision_id,
                            "disposition": decision.disposition,
                            "physical_kind": route.physical_kind,
                            "stage": source.stage,
                        },
                        sort_keys=True,
                    )
                )
                return 0
            reason = {
                "e6_interface_preflight": (
                    "source_owned_e6_interface_preflight_plan_unavailable"
                ),
                "serving": ("private_downstream_stage_source_rebuild_unavailable"),
                "e0_compatibility_decision": (
                    "source_owned_e0_compatibility_evidence_unavailable"
                ),
            }[route.physical_kind]
            raise FormalSingleOperatorDispatchBlocked(reason)
        run_root = create_formal_single_operator_run_directory(
            repository_root=args.repository_root,
            base_output_root=args.output_root,
            stage=source.stage,
            cell_id=args.cell,
            attempt="attempt-0",
            started_ns=started_ns,
        )
        if e6_interface:
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                materialize_formal_single_operator_e6_interface_replay_plan,
            )

            replay_path = run_root / "e6-interface-replay-plan.json"
            replay = materialize_formal_single_operator_e6_interface_replay_plan(
                execution_source_path=args.execution_source,
                materialized_cell_id=args.cell,
                output_path=replay_path,
            )
            print(
                json.dumps(
                    {
                        "additional_gpu_runs": replay.additional_gpu_runs,
                        "cell_id": args.cell,
                        "physical_execution_reused": (replay.physical_execution_reused),
                        "physical_kind": route.physical_kind,
                        "run_directory": str(run_root),
                        "run_plan": str(replay_path),
                        "run_plan_sha256": replay.sha256,
                        "shared_terminal": replay.shared_terminal.absolute_path,
                        "stage": source.stage,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if profiler:
            from lightcone_spec.experiments.formal_single_operator_profiler import (
                materialize_formal_single_operator_profiler_plan,
            )

            assert args.prepared_launch_bundle is not None
            assert args.profiler_tool is not None
            profiler_plan = materialize_formal_single_operator_profiler_plan(
                execution_source_path=args.execution_source,
                materialized_cell_id=args.cell,
                prepared_launch_bundle_path=args.prepared_launch_bundle,
                repository_root=args.repository_root,
                private_output_root=run_root,
                tool_path=args.profiler_tool,
                current_ns=started_ns,
            )
            print(
                json.dumps(
                    {
                        "cell_id": args.cell,
                        "physical_kind": route.physical_kind,
                        "run_directory": str(run_root),
                        "run_plan": str(
                            run_root / "formal-single-operator-profiler-plan.json"
                        ),
                        "run_plan_sha256": profiler_plan.sha256,
                        "stage": source.stage,
                        "subject_run_plan": (
                            profiler_plan.subject_run_plan.absolute_path
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if e5_failure:
            from lightcone_spec.experiments.formal_failure_execution import (
                materialize_formal_single_operator_e5_failure_execution_descriptor,
            )
            from lightcone_spec.orchestration.formal_physical_dispatch import (
                materialize_formal_single_operator_e5_failure_run_plan,
            )
            from lightcone_spec.orchestration.formal_single_operator_admission import (
                publish_formal_single_operator_admission,
            )

            assert args.prepared_launch_bundle is not None
            inputs = materialize_formal_single_operator_e5_failure_execution_descriptor(
                execution_source_path=args.execution_source,
                materialized_cell_id=args.cell,
                prepared_launch_bundle_path=args.prepared_launch_bundle,
                repository_root=args.repository_root,
                private_output_root=run_root,
                current_ns=started_ns,
            )
            descriptor_path = (
                run_root / "formal-single-operator-e5-failure-execution.json"
            )
            plan = materialize_formal_single_operator_e5_failure_run_plan(
                failure_execution_descriptor_path=descriptor_path,
                preflight_inputs_path=args.preflight_inputs,
            )
            admission = publish_formal_single_operator_admission(
                plan_path=run_root / "formal-serving-run-plan.json",
                inventory_path=inputs.inventory.absolute_path,
            )
            print(
                json.dumps(
                    {
                        "cell_id": args.cell,
                        "exclusive_timing": True,
                        "failure_execution_descriptor": str(descriptor_path),
                        "failure_execution_binding_sha256": (
                            inputs.expected_failure_execution_binding_sha256
                        ),
                        "formal_launch_admission": admission.absolute_path,
                        "physical_kind": route.physical_kind,
                        "retry_allowance": 0,
                        "run_directory": str(run_root),
                        "run_plan": str(run_root / "formal-serving-run-plan.json"),
                        "run_plan_sha256": plan.sha256,
                        "stage": source.stage,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if early:
            from lightcone_spec.experiments.formal_single_operator_early_execution import (
                materialize_formal_single_operator_early_run_plan_inputs,
            )
            from lightcone_spec.orchestration.formal_physical_dispatch import (
                materialize_formal_single_operator_serving_run_plan,
            )

            inputs = materialize_formal_single_operator_early_run_plan_inputs(
                execution_source_path=args.execution_source,
                materialized_cell_id=args.cell,
                preflight_inputs_path=args.preflight_inputs,
                private_output_root=run_root,
            )
            plan = materialize_formal_single_operator_serving_run_plan(
                early_run_plan_inputs_path=(
                    run_root / "formal-single-operator-early-run-plan-inputs.json"
                ),
            )
        elif e4_headline:
            from lightcone_spec.orchestration.formal_physical_dispatch import (
                materialize_formal_single_operator_downstream_serving_run_plan,
            )

            inputs = materialize_formal_single_operator_e4_direct_run_plan_inputs(
                execution_source_path=args.execution_source,
                materialized_cell_id=args.cell,
                repository_root=args.repository_root,
                preflight_inputs_path=args.preflight_inputs,
                private_output_root=run_root,
            )
            plan = materialize_formal_single_operator_downstream_serving_run_plan(
                downstream_run_plan_inputs_path=(
                    run_root / "formal-single-operator-downstream-run-plan-inputs.json"
                ),
            )
        else:
            from lightcone_spec.orchestration.formal_physical_dispatch import (
                materialize_formal_single_operator_prepared_downstream_serving_run_plan,
            )

            assert args.prepared_launch_bundle is not None
            inputs = (
                materialize_formal_single_operator_prepared_downstream_run_plan_inputs(
                    execution_source_path=args.execution_source,
                    materialized_cell_id=args.cell,
                    prepared_launch_bundle_path=args.prepared_launch_bundle,
                    private_output_root=run_root,
                    current_ns=started_ns,
                )
            )
            plan = (
                materialize_formal_single_operator_prepared_downstream_serving_run_plan(
                    prepared_downstream_run_plan_inputs_path=(
                        run_root
                        / "formal-single-operator-prepared-downstream-inputs.json"
                    ),
                    preflight_inputs_path=args.preflight_inputs,
                )
            )
        print(
            json.dumps(
                {
                    "run_directory": str(run_root),
                    "run_plan": str(run_root / "formal-serving-run-plan.json"),
                    "run_plan_sha256": plan.sha256,
                    "stage": inputs.stage,
                    "cell_id": inputs.materialized_cell_id,
                    "physical_kind": route.physical_kind,
                },
                sort_keys=True,
            )
        )
        return 0
    if operation == "execute-run":
        import asyncio
        import os
        import shutil

        from lightcone_spec.orchestration.formal_physical_dispatch import (
            FormalServingRunPlan,
            execute_formal_single_operator_serving_run_plan,
        )
        from lightcone_spec.orchestration.live_sglang import PinnedNvidiaSmiTool
        from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

        plan_binding = CanonicalJsonProofBinding.bind(args.run_plan)
        plan_value = plan_binding.reopen()
        if plan_value.get("kind") == (
            "formal_single_operator_e6_interface_replay_plan"
        ):
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                revalidate_formal_single_operator_e6_interface_fit_terminal,
                revalidate_formal_single_operator_e6_interface_replay_plan,
            )

            replay = revalidate_formal_single_operator_e6_interface_replay_plan(
                plan_binding.absolute_path
            )
            terminal = revalidate_formal_single_operator_e6_interface_fit_terminal(
                replay.shared_terminal.absolute_path
            )
            print(
                json.dumps(
                    {
                        "actual_result": replay.shared_terminal.absolute_path,
                        "additional_gpu_runs": replay.additional_gpu_runs,
                        "cell_id": replay.materialized_cell_id,
                        "physical_execution_reused": (replay.physical_execution_reused),
                        "status": terminal.status,
                        "terminal_sha256": terminal.sha256,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if plan_value.get("kind") == "formal_single_operator_profiler_capture_plan":
            from lightcone_spec.experiments.formal_single_operator_profiler import (
                run_formal_single_operator_profiler,
            )

            terminal = run_formal_single_operator_profiler(
                profiler_plan_path=plan_binding.absolute_path,
            )
            print(
                json.dumps(
                    {
                        "cell_id": terminal.cell_id,
                        "status": terminal.status,
                        "terminal_path": str(
                            Path(args.run_plan).parent
                            / "capture"
                            / "profiler-terminal.json"
                        ),
                        "terminal_sha256": terminal.sha256,
                        "variant": terminal.variant,
                    },
                    sort_keys=True,
                )
            )
            return 0 if terminal.status == "COMPLETE" else 42
        plan = FormalServingRunPlan.from_dict(plan_value)
        if (
            plan.schema_version != 2
            or plan.sha256 != plan_binding.semantic_sha256
            or plan.single_operator_execution_rebuild_source is None
        ):
            raise ValueError(
                "trusted single-operator execution requires one schema-2 run plan"
            )
        launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
        executable = shutil.which(
            "nvidia-smi",
            path=os.pathsep.join(launch.path_entries),
        )
        if executable is None:
            raise FileNotFoundError(
                "source-owned launch PATH does not contain nvidia-smi"
            )
        nvidia_smi_tool = PinnedNvidiaSmiTool.bind(executable)
        rebuild_kind = plan.single_operator_execution_rebuild_source.reopen().get(
            "kind"
        )
        if rebuild_kind == ("formal_single_operator_e5_failure_execution_descriptor"):
            from lightcone_spec.orchestration.formal_failure_physical import (
                execute_formal_e5_failure_run_plan,
                validate_formal_single_operator_e5_physical_outcome,
            )

            result = asyncio.run(
                execute_formal_e5_failure_run_plan(
                    plan_path=plan_binding.absolute_path,
                    launch_admission_path=(
                        Path(plan.private_output_root)
                        / "formal-single-operator-admission.json"
                    ),
                    failure_execution_descriptor_path=(
                        plan.single_operator_execution_rebuild_source.absolute_path
                    ),
                    nvidia_smi_tool=nvidia_smi_tool,
                )
            )
            outcome = validate_formal_single_operator_e5_physical_outcome(
                plan_path=plan_binding.absolute_path,
                run_receipt_path=plan.live_run_receipt_output_path,
                lifecycle_receipt_path=result.lifecycle_receipt.absolute_path,
            )
            print(
                json.dumps(
                    {
                        "cell_id": plan.materialized_cell_id,
                        "lifecycle_receipt": (outcome.lifecycle_receipt.absolute_path),
                        "physical_kind": "e5_failure",
                        "retry_allowance": 0,
                        "run_receipt": outcome.run_receipt.absolute_path,
                        "run_receipt_sha256": (outcome.run_receipt.semantic_sha256),
                        "status": outcome.status,
                    },
                    sort_keys=True,
                )
            )
            return 0
        asyncio.run(
            execute_formal_single_operator_serving_run_plan(
                plan_path=plan_binding.absolute_path,
                nvidia_smi_tool=nvidia_smi_tool,
            )
        )
        assert plan.single_operator_execution_rebuild_source is not None
        if rebuild_kind == ("formal_single_operator_profiler_subject_run_plan_inputs"):
            print(
                json.dumps(
                    {
                        "profile_subject_completed": True,
                        "run_plan_sha256": plan.sha256,
                    },
                    sort_keys=True,
                )
            )
            return 0
        manifest = finalize_formal_single_operator_run(
            repository_root=args.repository_root,
            run_plan_path=plan_binding.absolute_path,
        )
        print(
            json.dumps(
                {
                    "manifest_path": str(
                        Path(plan.private_output_root)
                        / "formal-single-operator-manifest.json"
                    ),
                    "manifest_sha256": manifest.sha256,
                    "status": manifest.completion_status,
                },
                sort_keys=True,
            )
        )
        return 0 if manifest.completion_status == "COMPLETE" else 42
    if operation == "finalize-run":
        plan_binding = CanonicalJsonProofBinding.bind(args.run_plan)
        plan_value = plan_binding.reopen()
        if plan_value.get("kind") == (
            "formal_single_operator_e6_interface_replay_plan"
        ):
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                revalidate_formal_single_operator_e6_interface_fit_terminal,
                revalidate_formal_single_operator_e6_interface_replay_plan,
            )

            replay = revalidate_formal_single_operator_e6_interface_replay_plan(
                plan_binding.absolute_path
            )
            terminal = revalidate_formal_single_operator_e6_interface_fit_terminal(
                replay.shared_terminal.absolute_path
            )
            print(terminal.sha256)
            return 0
        if plan_value.get("kind") == "formal_serving_run_plan":
            from lightcone_spec.orchestration.formal_physical_dispatch import (
                FormalServingRunPlan,
            )

            plan = FormalServingRunPlan.from_dict(plan_value)
            source_kind = (
                None
                if plan.single_operator_execution_rebuild_source is None
                else plan.single_operator_execution_rebuild_source.reopen().get("kind")
            )
            if source_kind == (
                "formal_single_operator_e5_failure_execution_descriptor"
            ):
                from lightcone_spec.orchestration.formal_failure_physical import (
                    validate_formal_single_operator_e5_physical_outcome,
                )

                outcome = validate_formal_single_operator_e5_physical_outcome(
                    plan_path=plan_binding.absolute_path,
                    run_receipt_path=plan.live_run_receipt_output_path,
                    lifecycle_receipt_path=plan.lifecycle_timing_output_path,
                )
                print(outcome.run_receipt.semantic_sha256)
                return 0
        manifest = finalize_formal_single_operator_run(
            repository_root=args.repository_root,
            run_plan_path=args.run_plan,
            execution_source_path=args.execution_source,
            inventory_path=args.inventory,
        )
        print(manifest.sha256)
        return 0
    raise AssertionError(operation)


__all__ = [
    "add_formal_single_operator_parser",
    "handle_formal_single_operator_command",
]
