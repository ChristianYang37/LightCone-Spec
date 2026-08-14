"""Reducer-derived scientific execution semantics for activated registry cells.

This module is deliberately an overlay, not an execution authority.  It binds
an already completed raw-activation replay, registered load, registry
declaration, and source-owned adaptation recipe needed to render one scientific
cell later.  A :class:`BudgetActivationAuthorityResult` remains a publicly
constructible Python value; its type alone does not prove that raw paths were
reopened.  Formal callers must first obtain it from the existing onsite replay
path.  This module neither authenticates that provenance nor authorizes a
release launch, and it is not consulted by the formal release gate in this
source release.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

from lightcone_spec.config.schema import RunConfig, RuntimeConfig
from lightcone_spec.experiments.budget_authority import (
    BudgetActivationAuthorityResult,
)
from lightcone_spec.experiments.load import cohort_assignments
from lightcone_spec.experiments.planning import (
    BudgetLoadBinding,
    E1ActivationAuthorityBinding,
    ReducerActivationArtifact,
    SealedE3aSelection,
)
from lightcone_spec.experiments.registry import (
    AdaptationRecipeDeclaration,
    AdaptationRecipeLookupKey,
    ExperimentCell,
    ExperimentRegistry,
    content_sha256,
    scientific_role_for_cell,
)
from lightcone_spec.experiments.sampling import SamplingProfile

EXECUTION_SEMANTICS_UNSUPPORTED_EXPERIMENT_REASON = (
    "cell_execution_semantics_experiment_unsupported"
)
EXECUTION_SEMANTICS_RAW_ACTIVATION_UNAVAILABLE_REASON = (
    "cell_execution_semantics_raw_activation_unavailable"
)
EXECUTION_SEMANTICS_ACTIVATION_IDENTITY_MISMATCH_REASON = (
    "cell_execution_semantics_activation_identity_mismatch"
)
EXECUTION_SEMANTICS_FOREIGN_CELL_REASON = "cell_execution_semantics_foreign_cell"
EXECUTION_SEMANTICS_CELL_NOT_ACTIVATED_REASON = (
    "cell_execution_semantics_cell_not_activated"
)
EXECUTION_SEMANTICS_E3A_SELECTION_UNAVAILABLE_REASON = (
    "cell_execution_semantics_e3a_selection_unavailable"
)
EXECUTION_SEMANTICS_E3A_SELECTION_MISMATCH_REASON = (
    "cell_execution_semantics_e3a_selection_mismatch"
)
EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON = (
    "cell_execution_semantics_registered_load_mismatch"
)
EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON = (
    "cell_execution_semantics_recipe_unavailable"
)
EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON = (
    "cell_execution_semantics_run_config_mismatch"
)


class CellExecutionSemanticsBlockedError(RuntimeError):
    """Stable named BLOCK for missing or conflicting scientific inputs."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"cell execution semantics are BLOCKED: {reason_code}")


def _block(reason_code: str) -> None:
    raise CellExecutionSemanticsBlockedError(reason_code)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_scalar_mapping(
    actual: dict[str, object], expected: dict[str, object]
) -> bool:
    return set(actual) == set(expected) and all(
        type(actual[name]) is type(expected[name]) and actual[name] == expected[name]
        for name in expected
    )


@dataclass(frozen=True)
class CellExecutionSemantics:
    """Immutable, content-bound E1 semantics derived after caller-owned replay.

    The declaration, activation-binding identity, sealed E3a selection,
    registered load, and full adaptive recipe are retained rather than replaced
    by a caller-authored plan summary.  Redundant digests make each boundary
    explicit and are rechecked during construction; they do not authenticate
    the binding's raw paths.
    """

    schema_version: int
    registry_sha256: str
    cell_declaration: ExperimentCell
    cell_declaration_sha256: str
    activation_authority_binding: E1ActivationAuthorityBinding
    activation_authority_binding_sha256: str
    activation_semantic_sha256: str
    e3a_selection: SealedE3aSelection
    e3a_selection_sha256: str
    load_binding: BudgetLoadBinding
    load_binding_sha256: str
    registered_load_plan_sha256: str
    registered_load_source_sha256: str
    registered_corpus_sha256: str
    registered_request_ids_sha256: str
    registered_sampling_parameters_sha256: str
    registered_request_count: int
    adaptation_recipe: AdaptationRecipeDeclaration | None
    adaptation_recipe_sha256: str | None
    expected_method: str
    expected_model: str
    expected_backend: str
    expected_task: str
    expected_model_max_context_length: int
    expected_runtime_context_length: int
    expected_concurrency: int
    expected_draft_width: int | None
    expected_draft_depth: int | None
    expected_speculation_enabled: bool
    expected_scope: str | None
    expected_parameterization: str
    expected_rank: int | None
    expected_optimizer: str | None
    expected_learning_rate: float | None
    expected_schedule: str | None
    expected_cohort: str
    expected_cohort_count: int
    expected_workload_seed: int
    expected_runtime_random_seed: int
    expected_sampling_profile_sha256: str
    expected_regime: str
    expected_arrival: str
    expected_slo: str
    expected_topology: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only cell execution semantics schema 1 is supported")
        if type(self.cell_declaration) is not ExperimentCell:
            raise TypeError("execution semantics require an exact cell declaration")
        if type(self.activation_authority_binding) is not E1ActivationAuthorityBinding:
            raise TypeError("execution semantics require an exact E1 binding value")
        if type(self.e3a_selection) is not SealedE3aSelection:
            raise TypeError("execution semantics require an exact E3a selection")
        if type(self.load_binding) is not BudgetLoadBinding:
            raise TypeError("execution semantics require an exact load binding")
        if (
            self.adaptation_recipe is not None
            and type(self.adaptation_recipe) is not AdaptationRecipeDeclaration
        ):
            raise TypeError("execution semantics require an exact recipe declaration")
        for name in (
            "registry_sha256",
            "cell_declaration_sha256",
            "activation_authority_binding_sha256",
            "activation_semantic_sha256",
            "e3a_selection_sha256",
            "load_binding_sha256",
            "registered_load_plan_sha256",
            "registered_load_source_sha256",
            "registered_corpus_sha256",
            "registered_request_ids_sha256",
            "registered_sampling_parameters_sha256",
            "expected_sampling_profile_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"execution semantics {name} is not a SHA-256")
        if self.adaptation_recipe_sha256 is not None and not _is_sha256(
            self.adaptation_recipe_sha256
        ):
            raise ValueError("execution semantics recipe identity is not a SHA-256")
        positive_integers = (
            "registered_request_count",
            "expected_model_max_context_length",
            "expected_runtime_context_length",
            "expected_concurrency",
            "expected_cohort_count",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 1
            for name in positive_integers
        ):
            raise ValueError("execution semantics positive integer fields are invalid")
        for name in ("expected_workload_seed", "expected_runtime_random_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"execution semantics {name} is invalid")
        for name in (
            "expected_draft_width",
            "expected_draft_depth",
            "expected_rank",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"execution semantics {name} is invalid")
        if type(self.expected_speculation_enabled) is not bool:
            raise ValueError("execution semantics speculation flag must be boolean")
        if self.expected_learning_rate is not None and (
            type(self.expected_learning_rate) not in {int, float}
            or not math.isfinite(self.expected_learning_rate)
            or self.expected_learning_rate <= 0
        ):
            raise ValueError("execution semantics learning rate is invalid")
        if self.cell_declaration.sha256 != self.cell_declaration_sha256:
            raise ValueError("cell declaration differs from its bound identity")
        if (
            self.activation_authority_binding.sha256
            != self.activation_authority_binding_sha256
            or self.activation_authority_binding.activation_sha256
            != self.activation_semantic_sha256
        ):
            raise ValueError("activation authority differs from its bound identity")
        if (
            self.e3a_selection.sha256 != self.e3a_selection_sha256
            or self.e3a_selection.registry_sha256 != self.registry_sha256
            or self.activation_authority_binding.selection_sha256
            != self.e3a_selection_sha256
        ):
            raise ValueError("E3a selection differs from its bound identity")
        registered = self.load_binding.registered_load
        if (
            self.load_binding.sha256 != self.load_binding_sha256
            or self.load_binding.cell_id != self.cell_declaration.cell_id
            or registered.paired_replay_sha256 != self.registered_load_plan_sha256
            or registered.scored.source_identity_sha256
            != self.registered_load_source_sha256
            or registered.scored.hashes.corpus_sha256 != self.registered_corpus_sha256
            or registered.scored.hashes.request_ids_sha256
            != self.registered_request_ids_sha256
            or registered.scored.hashes.sampling_parameters_sha256
            != self.registered_sampling_parameters_sha256
            or len(registered.scored.requests) != self.registered_request_count
        ):
            raise ValueError("registered load differs from its bound identity")

        identity = self.cell_declaration.identity
        expected = (
            self.expected_method,
            self.expected_model,
            self.expected_backend,
            self.expected_task,
            self.expected_model_max_context_length,
            self.expected_concurrency,
            self.expected_scope,
            self.expected_parameterization,
            self.expected_rank,
            self.expected_optimizer,
            self.expected_schedule,
            self.expected_cohort,
            self.expected_cohort_count,
            self.expected_workload_seed,
            self.expected_regime,
            self.expected_arrival,
            self.expected_slo,
            self.expected_topology,
        )
        actual = (
            identity.method,
            identity.model,
            identity.backend,
            identity.task,
            identity.context,
            identity.concurrency,
            identity.scope,
            identity.parameterization,
            identity.rank,
            identity.optimizer,
            identity.schedule,
            identity.cohort,
            identity.cohort_count,
            identity.seed,
            identity.regime,
            identity.arrival,
            identity.slo,
            identity.topology,
        )
        if actual != expected or any(
            type(actual_value) is not type(expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        ):
            raise ValueError("scientific fields differ from the cell declaration")
        if (
            self.expected_runtime_context_length
            != RuntimeConfig.model_fields["context_length"].default
        ):
            raise ValueError("runtime capacity context differs from the source schema")
        if (
            self.expected_runtime_random_seed
            != RuntimeConfig.model_fields["random_seed"].default
        ):
            raise ValueError("runtime random seed differs from the source schema")
        if self.expected_method == "target_only":
            if (
                self.expected_draft_width is not None
                or self.expected_draft_depth is not None
                or self.expected_speculation_enabled
            ):
                raise ValueError("target-only execution cannot bind draft semantics")
        elif (
            self.expected_draft_width != identity.width
            or self.expected_draft_width is None
            or self.expected_draft_depth != self.expected_draft_width - 1
            or not self.expected_speculation_enabled
        ):
            raise ValueError("speculative execution width/depth semantics differ")

        if self.expected_method in {"target_only", "static"}:
            if (
                self.adaptation_recipe is not None
                or self.adaptation_recipe_sha256 is not None
                or self.expected_learning_rate is not None
            ):
                raise ValueError("baseline semantics cannot bind adaptation")
        elif self.expected_method == "l0":
            recipe = self.adaptation_recipe
            if (
                self.cell_declaration.identity.experiment != "E1"
                or recipe is None
                or recipe.status != "AVAILABLE"
                or recipe.sha256 != self.adaptation_recipe_sha256
                or recipe.lookup_key
                != AdaptationRecipeLookupKey.from_cell(self.cell_declaration)
                or recipe.optimizer.learning_rate != self.expected_learning_rate
                or recipe.adaptation_group_id != self.expected_cohort
            ):
                raise ValueError("adaptive semantics differ from the full recipe")
        else:
            raise ValueError("cell execution semantics method is unsupported")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def validate_run_config(self, config: RunConfig) -> None:
        """Require one RunConfig to realize these scientific semantics exactly.

        Rank-local topology, model revision receipts, policy identity, and other
        release authorities remain the responsibility of their existing gates.
        """

        if type(config) is not RunConfig:
            raise TypeError("execution semantics require an exact RunConfig")
        if (
            config.method != self.expected_method
            or config.model.target != self.expected_model
            or (
                self.expected_backend != "NONE"
                and config.model.algorithm != self.expected_backend
            )
            or config.model.max_context_length != self.expected_model_max_context_length
            or config.runtime.context_length != self.expected_runtime_context_length
            or config.runtime.random_seed != self.expected_runtime_random_seed
            or config.runtime.sampling_profile_sha256
            != self.expected_sampling_profile_sha256
            or config.runtime.max_running_requests != self.expected_concurrency
            or config.runtime.speculation_enabled
            is not self.expected_speculation_enabled
        ):
            _block(EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON)
        if self.expected_method == "target_only":
            if config.adaptation is not None:
                _block(EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON)
            return
        if (
            config.runtime.speculative_num_draft_tokens != self.expected_draft_width
            or config.model.draft_depth != self.expected_draft_depth
        ):
            _block(EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON)
        if self.expected_method == "static":
            if config.adaptation is not None:
                _block(EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON)
            return
        recipe = self.adaptation_recipe
        if recipe is None:  # pragma: no cover - guarded by construction
            _block(EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON)
        expected_adaptation = recipe.to_adaptation_config()
        if (
            config.adaptation != expected_adaptation
            or config.adaptation.adaptation_group_id != self.expected_cohort
        ):
            _block(EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON)


def _validate_e1_activation(
    authority: BudgetActivationAuthorityResult,
) -> tuple[ExperimentRegistry, ReducerActivationArtifact, SealedE3aSelection]:
    if authority.experiment != "E1":
        _block(EXECUTION_SEMANTICS_UNSUPPORTED_EXPERIMENT_REASON)
    registry = authority.registry
    artifact = authority.activation_artifact
    selection = authority.e3a_selection
    binding = authority.binding
    if (
        type(binding) is not E1ActivationAuthorityBinding
        or type(artifact) is not ReducerActivationArtifact
    ):
        _block(EXECUTION_SEMANTICS_RAW_ACTIVATION_UNAVAILABLE_REASON)
    if type(selection) is not SealedE3aSelection:
        _block(EXECUTION_SEMANTICS_E3A_SELECTION_UNAVAILABLE_REASON)
    if (
        authority.family_activations
        or authority.family_power_reductions
        or authority.e1_pareto is not None
        or authority.prior_e2_reductions
        or authority.prior_e2_stage_authorities
        or artifact.plan.experiment != "E1"
        or artifact.plan.status != "AVAILABLE"
        or artifact.plan.registry_sha256 != registry.sha256
        or binding.generated_registry.semantic_sha256 != registry.sha256
        or binding.activation_sha256 != artifact.sha256
    ):
        _block(EXECUTION_SEMANTICS_ACTIVATION_IDENTITY_MISMATCH_REASON)
    if (
        selection.registry_sha256 != registry.sha256
        or selection.runtime_sha256 != artifact.plan.runtime_sha256
        or selection.split_sha256 != artifact.plan.split_sha256
        or selection.sha256 != artifact.plan.source_selection_sha256
        or binding.selection_sha256 != selection.sha256
        or binding.runtime.semantic_sha256 != selection.runtime_sha256
        or binding.split.semantic_sha256 != selection.split_sha256
    ):
        _block(EXECUTION_SEMANTICS_E3A_SELECTION_MISMATCH_REASON)
    return registry, artifact, selection


def _validate_registered_load(
    *,
    cell: ExperimentCell,
    selection: SealedE3aSelection,
    load_binding: BudgetLoadBinding,
) -> None:
    if load_binding.cell_id != cell.cell_id:
        _block(EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON)
    registered = load_binding.registered_load
    try:
        registered.validate()
    except (OverflowError, TypeError, ValueError) as error:
        raise CellExecutionSemanticsBlockedError(
            EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON
        ) from error
    scored = registered.scored
    source = dict(scored.source_parameters)
    exact_source_fields = {
        "cohort_count",
        "cohort_popularity",
        "cohort_seed",
        "concurrency",
        "generator",
        "request_count",
        "zipf_exponent",
    }
    request_count = len(scored.requests)
    if (
        scored.split != "tuning"
        or scored.source_kind != "closed_loop"
        or set(source) != exact_source_fields
        or type(source.get("concurrency")) is not int
        or source["concurrency"] != selection.concurrency
        or source["concurrency"] != cell.identity.concurrency
        or type(source.get("cohort_count")) is not int
        or source["cohort_count"] != cell.identity.cohort_count
        or type(source.get("cohort_seed")) is not int
        or source["cohort_seed"] != cell.identity.seed
        or source.get("generator") != "closed_loop_zero_think_v1"
        or type(source.get("request_count")) is not int
        or source["request_count"] != request_count
        or source.get("cohort_popularity") not in {"uniform", "zipf"}
        or cell.identity.cohort
        != f"K={source['cohort_count']}:{source['cohort_popularity']}"
        or load_binding.minimum_completed_requests > request_count
        or tuple(request.ordinal for request in scored.requests)
        != tuple(range(request_count))
        or len({request.namespace for request in scored.requests}) != 1
    ):
        _block(EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON)
    zipf_exponent = source["zipf_exponent"]
    if type(zipf_exponent) not in {int, float}:
        _block(EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON)
    try:
        expected_cohorts = cohort_assignments(
            request_count,
            cohort_count=source["cohort_count"],
            popularity=source["cohort_popularity"],
            seed=source["cohort_seed"],
            zipf_exponent=float(zipf_exponent),
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise CellExecutionSemanticsBlockedError(
            EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON
        ) from error
    if tuple(request.cohort_id for request in scored.requests) != expected_cohorts:
        _block(EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON)
    context = cell.identity.context
    if context is None:
        _block(EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON)
    corpora = (scored,) + (() if registered.warmup is None else (registered.warmup,))
    for corpus in corpora:
        for request in corpus.requests:
            sampling = dict(request.sampling.items)
            if len(
                request.input_token_ids
            ) + request.requested_output_tokens > context or not _exact_scalar_mapping(
                sampling,
                SamplingProfile().parameters(
                    seed=cell.identity.seed,
                    max_new_tokens=request.requested_output_tokens,
                ),
            ):
                _block(EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON)


def resolve_cell_execution_semantics(
    *,
    activation: BudgetActivationAuthorityResult,
    load_binding: BudgetLoadBinding,
    cell: ExperimentCell,
) -> CellExecutionSemantics:
    """Derive one E1 overlay from an activation replay performed by the caller.

    This function exact-compares the supplied immutable values but does not
    reopen their raw paths.  Formal callers must pass the result returned by
    ``replay_budget_activation_authority`` in the same onsite execution flow.
    """

    if type(activation) is not BudgetActivationAuthorityResult:
        raise TypeError("execution semantics require an exact replay-result value")
    if type(load_binding) is not BudgetLoadBinding:
        raise TypeError("execution semantics require an exact load binding")
    if type(cell) is not ExperimentCell:
        raise TypeError("execution semantics require an exact registry cell")
    registry, artifact, selection = _validate_e1_activation(activation)
    owned = tuple(
        candidate for candidate in registry.cells if candidate.cell_id == cell.cell_id
    )
    if len(owned) != 1 or owned[0] != cell:
        _block(EXECUTION_SEMANTICS_FOREIGN_CELL_REASON)
    if cell.identity.experiment != "E1":
        _block(EXECUTION_SEMANTICS_UNSUPPORTED_EXPERIMENT_REASON)
    scientific_role = scientific_role_for_cell(registry, cell)
    if scientific_role not in {"target_only", "static", "lc_candidate"}:
        _block(EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON)
    if cell.cell_id not in artifact.plan.activated_cell_ids:
        _block(EXECUTION_SEMANTICS_CELL_NOT_ACTIVATED_REASON)
    identity = cell.identity
    if (
        identity.concurrency != selection.concurrency
        or f"width={selection.width}:concurrency={selection.concurrency}"
        not in identity.variant
        or (
            identity.width is not None
            if identity.method == "target_only"
            else identity.width != selection.width
        )
    ):
        _block(EXECUTION_SEMANTICS_E3A_SELECTION_MISMATCH_REASON)
    _validate_registered_load(
        cell=cell,
        selection=selection,
        load_binding=load_binding,
    )

    recipe: AdaptationRecipeDeclaration | None = None
    if scientific_role == "lc_candidate":
        try:
            recipe = registry.adaptation_recipe_for_cell(cell)
        except (TypeError, ValueError) as error:
            raise CellExecutionSemanticsBlockedError(
                EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON
            ) from error
        if recipe.status != "AVAILABLE":
            _block(EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON)
    elif scientific_role not in {"target_only", "static"}:
        _block(EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON)

    registered = load_binding.registered_load
    expected_width = None if identity.method == "target_only" else selection.width
    return CellExecutionSemantics(
        schema_version=1,
        registry_sha256=registry.sha256,
        cell_declaration=cell,
        cell_declaration_sha256=cell.sha256,
        activation_authority_binding=activation.binding,
        activation_authority_binding_sha256=activation.binding.sha256,
        activation_semantic_sha256=artifact.sha256,
        e3a_selection=selection,
        e3a_selection_sha256=selection.sha256,
        load_binding=load_binding,
        load_binding_sha256=load_binding.sha256,
        registered_load_plan_sha256=registered.paired_replay_sha256,
        registered_load_source_sha256=registered.scored.source_identity_sha256,
        registered_corpus_sha256=registered.scored.hashes.corpus_sha256,
        registered_request_ids_sha256=(registered.scored.hashes.request_ids_sha256),
        registered_sampling_parameters_sha256=(
            registered.scored.hashes.sampling_parameters_sha256
        ),
        registered_request_count=len(registered.scored.requests),
        adaptation_recipe=recipe,
        adaptation_recipe_sha256=None if recipe is None else recipe.sha256,
        expected_method=identity.method,
        expected_model=identity.model,
        expected_backend=identity.backend,
        expected_task=identity.task,
        expected_model_max_context_length=int(identity.context),
        expected_runtime_context_length=40960,
        expected_concurrency=int(identity.concurrency),
        expected_draft_width=expected_width,
        expected_draft_depth=None if expected_width is None else expected_width - 1,
        expected_speculation_enabled=identity.method != "target_only",
        expected_scope=identity.scope,
        expected_parameterization=identity.parameterization,
        expected_rank=identity.rank,
        expected_optimizer=identity.optimizer,
        expected_learning_rate=(
            None if recipe is None else recipe.optimizer.learning_rate
        ),
        expected_schedule=identity.schedule,
        expected_cohort=identity.cohort,
        expected_cohort_count=identity.cohort_count,
        expected_workload_seed=identity.seed,
        expected_runtime_random_seed=1,
        expected_sampling_profile_sha256=SamplingProfile().sha256,
        expected_regime=identity.regime,
        expected_arrival=identity.arrival,
        expected_slo=identity.slo,
        expected_topology=identity.topology,
    )


__all__ = [
    "EXECUTION_SEMANTICS_ACTIVATION_IDENTITY_MISMATCH_REASON",
    "EXECUTION_SEMANTICS_CELL_NOT_ACTIVATED_REASON",
    "EXECUTION_SEMANTICS_E3A_SELECTION_MISMATCH_REASON",
    "EXECUTION_SEMANTICS_E3A_SELECTION_UNAVAILABLE_REASON",
    "EXECUTION_SEMANTICS_FOREIGN_CELL_REASON",
    "EXECUTION_SEMANTICS_LOAD_MISMATCH_REASON",
    "EXECUTION_SEMANTICS_RAW_ACTIVATION_UNAVAILABLE_REASON",
    "EXECUTION_SEMANTICS_RECIPE_UNAVAILABLE_REASON",
    "EXECUTION_SEMANTICS_RUN_CONFIG_MISMATCH_REASON",
    "EXECUTION_SEMANTICS_UNSUPPORTED_EXPERIMENT_REASON",
    "CellExecutionSemantics",
    "CellExecutionSemanticsBlockedError",
    "resolve_cell_execution_semantics",
]
