#!/usr/bin/env python3
"""Run all six HeRCULES sequences used in Dynamic-ICP Table I."""

import argparse
import json
from pathlib import Path
import sys

from _script_utils import module_command, require_directory, require_file
from _paper_batch import run_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="HeRCULES dataset root")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/datasets/hercules_paper_sequences.json"),
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
    selected = list(manifest) if args.sequences is None else args.sequences
    unknown = sorted(set(selected) - set(manifest))
    if unknown:
        parser.error(f"Unknown sequences: {unknown}")
    jobs = []
    try:
        for name in selected:
            sequence = require_directory(args.root / name / manifest[name], "sequence")
            scans = require_directory(sequence / "LiDAR/Aeva", "scan")
            gt = require_file(sequence / "PR_GT/Aeva_gt.txt", "GT")
            pairs = len(list(scans.glob("*.bin"))) - 1
            output = args.output / name
            jobs.append(
                {
                    "name": name,
                    "group": "all",
                    "output": output,
                    "command": module_command(
                        "dynamic_icp.hercules_full_experiment",
                        (
                            "--scans", scans,
                            "--gt", gt,
                            "--config", args.config,
                            "--start", 0,
                            "--pairs", pairs,
                            "--output", output,
                        ),
                    ),
                }
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return run_jobs(jobs, ("all",), args.output, args.workers)


if __name__ == "__main__":
    sys.exit(main())
