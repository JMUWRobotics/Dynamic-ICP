"""Resumable subprocess orchestration and weighted RPE aggregation."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[1]


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def weighted_metrics(records):
    variants = sorted(
        set.intersection(*(set(record) for record in records)) if records else set()
    )
    output = {}
    for variant in variants:
        values = [record[variant] for record in records]
        pairs = sum(value["pairs"] for value in values)
        if not pairs:
            continue
        output[variant] = {
            "sequences": len(values),
            "pairs": pairs,
            "rte_mean_m": sum(v["pairs"] * v["rte_mean_m"] for v in values) / pairs,
            "rte_rmse_m": math.sqrt(
                sum(v["pairs"] * v["rte_rmse_m"] ** 2 for v in values) / pairs
            ),
            "rre_mean_deg": sum(v["pairs"] * v["rre_mean_deg"] for v in values) / pairs,
            "rre_rmse_deg": math.sqrt(
                sum(v["pairs"] * v["rre_rmse_deg"] ** 2 for v in values) / pairs
            ),
            "rte_rmse_sequence_mean_m": sum(v["rte_rmse_m"] for v in values)
            / len(values),
            "rre_rmse_sequence_mean_deg": sum(v["rre_rmse_deg"] for v in values)
            / len(values),
        }
    return output


def collect(jobs, groups):
    sequences = {}
    for job in jobs:
        result_path = job["output"] / "results.json"
        sequences[job["name"]] = {
            "group": job["group"],
            "status": "complete" if result_path.is_file() else "pending",
        }
        if result_path.is_file():
            sequences[job["name"]]["results"] = json.loads(result_path.read_text())
    aggregates = {}
    for group in groups:
        records = [
            value["results"]
            for value in sequences.values()
            if value["group"] == group and value["status"] == "complete"
        ]
        aggregates[group] = weighted_metrics(records)
    return {"sequences": sequences, "aggregates": aggregates}


def run_jobs(jobs, groups, output_root, workers):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "batch_summary.json"
    pending = [job for job in jobs if not (job["output"] / "results.json").is_file()]
    _atomic_json(summary_path, collect(jobs, groups))
    if not pending:
        return 0

    environment = os.environ.copy()
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"

    def execute(job):
        job["output"].mkdir(parents=True, exist_ok=True)
        resume = (job["output"] / "protocol.json").exists()
        command = list(job["command"]) + ["--resume"]
        mode = "a" if resume else "w"
        with (job["output"] / "runner.log").open(mode) as log:
            log.write("COMMAND " + " ".join(command) + "\n")
            log.flush()
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                cwd=REPOSITORY,
            )
        return job, result.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(execute, job) for job in pending]
        for future in as_completed(futures):
            job, code = future.result()
            state = "complete" if code == 0 else f"failed({code})"
            print(f"[{state}] {job['group']}/{job['name']}", flush=True)
            if code:
                failures.append(job["name"])
            _atomic_json(summary_path, collect(jobs, groups))
    return int(bool(failures))
