"""Immutable run units (spec 13.6).

A minimal run unit is uniquely determined by exactly these keys; the
unit ID is the SHA-256 of the canonicalized manifest JSON, so any field
change produces a new unit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from lightcone_spec.locking.hashing import sha256_json

UNIT_KEYS = (
    "phase",
    "model_pair",
    "method",
    "dataset",
    "prompt_subset",
    "seed",
    "lifecycle",
    "sampling_profile",
    "trainable_scope",
    "stride",
    "logical_delay",
    "concurrency",
    "contention_condition",
    "adapter_rank",
    "transport_variant",
)


@dataclass(frozen=True)
class RunUnit:
    phase: str
    model_pair: str
    method: str
    dataset: str
    prompt_subset: str
    seed: int
    lifecycle: str
    sampling_profile: str  # main_t1_p1 | greedy_t0
    trainable_scope: str  # schema-v1 alias or canonical tail mode
    stride: int
    logical_delay: int
    concurrency: int
    contention_condition: str  # control | staleness_only | contention_only | realistic_async | none
    adapter_rank: int
    transport_variant: Optional[str] = None
    required: bool = True
    allow_resource_skip: bool = False
    parameter_scope: str = "tail"
    parameter_allowlist: tuple[str, ...] = ()

    def key_dict(self) -> dict:
        d = asdict(self)
        return {k: d[k] for k in UNIT_KEYS}

    def identity_dict(self) -> dict:
        """Return the effective experimental identity.

        Schema-v1 called the three tiers ``adapter``, ``lora`` and ``full``.
        Alias spelling cannot create a second experiment, and full-rank tail
        has no applicable adapter rank.
        """
        from lightcone_spec.config.schema import (
            canonical_parameter_scope,
            canonical_tail_layout_mode,
            canonical_weight_update_mode,
        )

        d = self.key_dict()
        mode = canonical_weight_update_mode(self.trainable_scope)
        scope = canonical_parameter_scope(self.parameter_scope)
        allowlist = tuple(str(name).strip() for name in self.parameter_allowlist)
        if scope == "allowlist" and not allowlist:
            raise ValueError(
                "parameter_scope=allowlist requires parameter_allowlist"
            )
        if len(set(allowlist)) != len(allowlist):
            raise ValueError("parameter_allowlist entries must be unique")
        if scope != "allowlist" and allowlist:
            raise ValueError(
                "parameter_allowlist is only valid with parameter_scope=allowlist"
            )
        if mode == "residual" and scope != "tail":
            raise ValueError(
                "weight_update_mode=residual requires parameter_scope=tail"
            )
        d["trainable_scope"] = canonical_tail_layout_mode(mode)
        if mode == "full":
            d["adapter_rank"] = None
        # Preserve every historical cache-safe-tail unit ID.  A non-tail
        # scope is a new experimental dimension and therefore extends the
        # identity explicitly.
        if scope != "tail":
            d["parameter_scope"] = scope
            d["parameter_allowlist"] = (
                sorted(allowlist)
                if scope == "allowlist"
                else []
            )
        return d

    @property
    def weight_update_mode(self) -> str:
        from lightcone_spec.config.schema import canonical_weight_update_mode

        return canonical_weight_update_mode(self.trainable_scope)

    @property
    def unit_id(self) -> str:
        return sha256_json(self.identity_dict())

    def to_manifest_dict(self) -> dict:
        # Manifests emitted after loading/overriding use the effective identity:
        # the schema-v1 ``full`` alias and its meaningless carried rank never
        # leak back into a resolved manifest.
        d = self.identity_dict()
        d["weight_update_mode"] = self.weight_update_mode
        from lightcone_spec.config.schema import canonical_parameter_scope

        parameter_scope = canonical_parameter_scope(self.parameter_scope)
        d["parameter_scope"] = parameter_scope
        if parameter_scope == "allowlist":
            d["parameter_allowlist"] = list(self.parameter_allowlist)
        d["unit_id"] = self.unit_id
        d["required"] = self.required
        d["allow_resource_skip"] = self.allow_resource_skip
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunUnit":
        missing = [
            key
            for key in UNIT_KEYS
            if key != "trainable_scope" and key not in d
        ]
        if missing:
            raise KeyError(missing[0])
        legacy_source_identity = (
            {k: d[k] for k in UNIT_KEYS}
            if (
                "trainable_scope" in d
                and d.get("parameter_scope", "tail") == "tail"
                and not d.get("parameter_allowlist")
            )
            else None
        )
        source_scope = d.get("trainable_scope", d.get("weight_update_mode"))
        if source_scope is None:
            raise KeyError("weight_update_mode")
        source_scope = str(source_scope)
        kwargs = {
            key: d[key]
            for key in UNIT_KEYS
            if key != "trainable_scope"
        }
        kwargs["trainable_scope"] = source_scope
        public_mode = d.get("weight_update_mode")
        from lightcone_spec.config.schema import (
            canonical_parameter_scope,
            canonical_tail_layout_mode,
            canonical_weight_update_mode,
        )

        canonical_mode = canonical_weight_update_mode(source_scope)
        if (
            public_mode is not None
            and canonical_weight_update_mode(str(public_mode)) != canonical_mode
        ):
            raise ValueError(
                "weight_update_mode conflicts with deprecated "
                f"trainable_scope: {public_mode!r} != {source_scope!r}"
            )
        kwargs["trainable_scope"] = canonical_tail_layout_mode(canonical_mode)
        if canonical_mode == "full":
            # The dataclass keeps an integer for code paths shared with the
            # ranked modes; it is never part of full-tail identity or output.
            kwargs["adapter_rank"] = (
                16 if kwargs["adapter_rank"] is None else kwargs["adapter_rank"]
            )
        kwargs["required"] = d.get("required", True)
        kwargs["allow_resource_skip"] = d.get("allow_resource_skip", False)
        kwargs["parameter_scope"] = canonical_parameter_scope(
            d.get("parameter_scope", "tail")
        )
        raw_allowlist = d.get("parameter_allowlist", ())
        if not isinstance(raw_allowlist, (list, tuple)):
            raise ValueError("parameter_allowlist must be a list of names")
        kwargs["parameter_allowlist"] = tuple(str(name) for name in raw_allowlist)
        if kwargs["parameter_scope"] == "allowlist":
            if not kwargs["parameter_allowlist"]:
                raise ValueError(
                    "parameter_scope=allowlist requires parameter_allowlist"
                )
            if len(set(kwargs["parameter_allowlist"])) != len(
                kwargs["parameter_allowlist"]
            ):
                raise ValueError("parameter_allowlist entries must be unique")
        elif kwargs["parameter_allowlist"]:
            raise ValueError(
                "parameter_allowlist is only valid with parameter_scope=allowlist"
            )
        unit = cls(**kwargs)
        declared = d.get("unit_id")
        if declared is not None and declared != unit.unit_id:
            legacy_alias_id = (
                sha256_json(legacy_source_identity)
                if legacy_source_identity is not None
                else None
            )
            if declared == legacy_alias_id:
                # Accept the immutable source provenance, but only expose the
                # canonical effective ID from this point onward.
                return unit
            raise ValueError(f"unit_id drift: declared {declared}, computed {unit.unit_id}")
        return unit
