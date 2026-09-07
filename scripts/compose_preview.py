#!/usr/bin/env python3
"""Time-align six independently recorded originals at 1x; never interpolate tokens."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


def submission_alignment(inputs):
    """Require captured clock evidence; a recorder start is not a submission."""
    records = [json.loads(path.with_name("capture.json").read_text()) for path in inputs]
    configurations, labels, offsets, bounds = set(), set(), [], []
    for path, record in zip(inputs, records, strict=True):
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record.get("video_sha256"):
                raise ValueError("original video hash differs from capture evidence")
        measurement, alignment = record.get("measurement", {}), record.get("alignment", {})
        offset, bound = alignment.get("submission_offset_seconds"), alignment.get("clock_uncertainty_seconds")
        if (record.get("status") != "completed" or record.get("errors")
                or alignment.get("status") != "verified_clock_bounded"
                or not isinstance(offset, (float, int)) or not isinstance(bound, (float, int))
                or not math.isfinite(offset) or not math.isfinite(bound)
                or not 0 <= bound <= .1 or offset < bound):
            raise ValueError("all six takes require valid submission-clock alignment")
        configurations.add(tuple(measurement.get(key) for key in (
            "model", "tp", "dispatcher_concurrency", "input_tokens_per_request", "max_output_tokens",
        )))
        labels.add(measurement.get("label"))
        # Start at the earliest plausible submission: never discard prefill.
        offsets.append(offset - bound)
        bounds.append(bound)
    if len(configurations) != 1 or len(labels) != 6 or None in labels:
        raise ValueError("six distinct methods must share model, TP and workload")
    model, tp, concurrency, inputs_count, outputs_count = next(iter(configurations))
    if not model or tp not in (1, 2) or (concurrency, inputs_count, outputs_count) != (8, 16384, 1024):
        raise ValueError("video must use accepted common TP, actual c8 and 16K/1024 budget")
    return offsets, bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs=6, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if args.output.exists() or not all(path.is_file() for path in args.inputs):
        raise ValueError("output must be new and all six original recordings must exist")
    offsets, bounds = submission_alignment(args.inputs)
    # Only remove pre-submission recorder/UI lead-in; retain all actual prefill,
    # stream frames and originals. No speed multiplier or token interpolation.
    command = [args.ffmpeg, "-n"]
    for path in args.inputs:
        command.extend(("-i", str(path)))
    filters = ";".join(f"[{i}:v]trim=start={offsets[i]:.9f},setpts=PTS-STARTPTS,scale=800:526[v{i}]" for i in range(6))
    filters += ";" + "".join(f"[v{i}]" for i in range(6)) + (
        "xstack=inputs=6:layout=0_0|800_0|1600_0|0_526|800_526|1600_526:fill=black,"
        "drawtext=text='Independent real runs - submission aligned within 200ms plus frame resolution - 1x':"
        "x=20:y=h-34:fontsize=24:fontcolor=white:box=1:boxcolor=black[v]"
    )
    command.extend(("-filter_complex", filters, "-map", "[v]", "-an", "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)))
    subprocess.run(command, check=True)
    args.output.with_suffix(".json").write_text(json.dumps({
        "independent_runs": True, "playback_speed": 1,
        "alignment": "earliest clock-bounded submission; full prefill retained; native frame resolution applies",
        "removed_pre_submission_seconds": offsets,
        "clock_half_width_seconds": bounds,
        "inputs": [path.name for path in args.inputs], "excluded_from_benchmark": True,
    }, indent=2))


if __name__ == "__main__":
    main()
