#!/usr/bin/env python3
"""Run the 44 highway and 43 city AevaScenes sequences in Table III."""

import argparse
import json
from pathlib import Path
import sys

from _script_utils import module_command, require_directory, require_file
from _paper_batch import run_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="AevaScenes dataset root")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/datasets/aevascenes_paper_sequences.json"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/datasets/reference.json")
    )
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = json.loads(args.manifest.read_text())
    membership = {sid: group for group, ids in manifest.items() for sid in ids}
    selected = list(membership) if args.sequences is None else args.sequences
    unknown = sorted(set(selected) - set(membership))
    if unknown:
        parser.error(f"Unknown sequences: {unknown}")
    jobs = []
    try:
        for sid in selected:
            sequence = require_directory(args.root / sid, "sequence")
            metadata = require_file(sequence / "sequence.json", "sequence metadata")
            pairs = len(json.loads(metadata.read_text())["frames"]) - 1
            if pairs < 1:
                raise ValueError(f"No frame pairs in {sid}")
            output = args.output / membership[sid] / sid
            jobs.append(
                {
                    "name": sid,
                    "group": membership[sid],
                    "output": output,
                    "command": module_command(
                        "dynamic_icp.aevascenes_paper_experiment",
                        (
                            "--sequence", sequence,
                            "--start", 0,
                            "--pairs", pairs,
                            "--config", args.config,
                            "--output", output,
                        ),
                    ),
                }
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return run_jobs(jobs, tuple(manifest), args.output, args.workers)


if __name__ == "__main__":
    sys.exit(main())
