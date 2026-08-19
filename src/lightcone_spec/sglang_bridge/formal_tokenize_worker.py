"""First-party offline tokenizer worker for sealed formal request schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lightcone_spec.runtime.proof_artifact import publish_canonical_json_no_replace


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_file_bytes(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _load_input(path: Path) -> dict[str, object]:
    body = path.read_bytes()
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("formal tokenizer input is not UTF-8 JSON") from error
    fields = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "schedule_source_sha256",
        "tokenizer_model_id",
        "tokenizer_revision",
        "tokenizer_snapshot_path",
        "tokenizer_content_authority_sha256",
        "requests",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema_version"] != 1
        or value["kind"] != "formal_serving_tokenization_input"
        or body != _canonical_file_bytes(value)
        or type(value["requests"]) is not list
        or not value["requests"]
    ):
        raise ValueError("formal tokenizer input schema differs")
    return value


def _tokenize(input_path: Path, output_path: Path) -> None:
    source = _load_input(input_path)
    from transformers import AutoTokenizer
    from transformers import __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(
        str(source["tokenizer_snapshot_path"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    rows: list[dict[str, object]] = []
    for item in source["requests"]:
        if (
            type(item) is not dict
            or set(item) != {"request_id", "ordinal", "prompt", "prompt_sha256"}
            or type(item["request_id"]) is not str
            or type(item["ordinal"]) is not int
            or isinstance(item["ordinal"], bool)
            or item["ordinal"] < 0
            or type(item["prompt"]) is not str
            or not item["prompt"]
            or _sha256(item["prompt"]) != item["prompt_sha256"]
        ):
            raise ValueError("formal tokenizer request row differs")
        encoded = tokenizer(
            item["prompt"],
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        token_ids = encoded.get("input_ids")
        if (
            type(token_ids) is not list
            or not token_ids
            or any(type(token) is not int or token < 0 for token in token_ids)
        ):
            raise ValueError("formal tokenizer emitted invalid token IDs")
        rows.append(
            {
                "request_id": item["request_id"],
                "ordinal": item["ordinal"],
                "prompt_sha256": item["prompt_sha256"],
                "input_token_ids": token_ids,
                "input_token_ids_sha256": _sha256(token_ids),
            }
        )
    if tuple(row["request_id"] for row in rows) != tuple(
        item["request_id"] for item in source["requests"]
    ):
        raise RuntimeError("formal tokenizer changed request order")
    output = {
        "schema_version": 1,
        "kind": "formal_serving_tokenization_output",
        "protocol_sha256": source["protocol_sha256"],
        "schedule_source_sha256": source["schedule_source_sha256"],
        "tokenizer_model_id": source["tokenizer_model_id"],
        "tokenizer_revision": source["tokenizer_revision"],
        "tokenizer_snapshot_path": source["tokenizer_snapshot_path"],
        "tokenizer_content_authority_sha256": source[
            "tokenizer_content_authority_sha256"
        ],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "transformers_version": transformers_version,
        "requests": rows,
    }
    publish_canonical_json_no_replace(output_path, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    _tokenize(args.input, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
