#!/usr/bin/env python3
"""Time-align six independently recorded originals at 1x; never interpolate tokens."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs=6, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if args.output.exists() or not all(path.is_file() for path in args.inputs):
        raise ValueError("output must be new and all six original recordings must exist")
    # No setpts speed multiplier, trimming, inserted token animation, or frame interpolation.
    command = [args.ffmpeg, "-n"]
    for path in args.inputs:
        command.extend(("-i", str(path)))
    filters = ";".join(f"[{i}:v]setpts=PTS-STARTPTS,scale=800:526[v{i}]" for i in range(6))
    filters += ";" + "".join(f"[v{i}]" for i in range(6)) + (
        "xstack=inputs=6:layout=0_0|800_0|1600_0|0_526|800_526|1600_526:fill=black,"
        "drawtext=text='Independent real runs - time aligned - 1x playback':"
        "x=20:y=h-34:fontsize=24:fontcolor=white:box=1:boxcolor=black[v]"
    )
    command.extend(("-filter_complex", filters, "-map", "[v]", "-an", "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)))
    subprocess.run(command, check=True)
    args.output.with_suffix(".json").write_text(json.dumps({
        "independent_runs": True, "playback_speed": 1,
        "alignment": "original recording time zero; prefill and UI-start wait retained",
        "inputs": [path.name for path in args.inputs], "excluded_from_benchmark": True,
    }, indent=2))


if __name__ == "__main__":
    main()
