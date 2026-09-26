#!/usr/bin/env python3
"""Generate and run all six independent CAPPIBARAS fits locally.
"""

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

from config import CAPPIBARAS_PATH, OUTPUT_PATH


RUNS = [
    (0, "major"),
    (0, "minor"),
    (1, "major"),
    (1, "minor"),
    (2, "major"),
    (2, "minor"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--template",
        default="Wang2026",
        help="Base YAML name in CAPPIBARAS/yamls (without .yaml)",
    )
    p.add_argument("--job-prefix", default="Wang2026_gnfw")
    p.add_argument("--threads-per-fit", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_template(name):
    path = Path(CAPPIBARAS_PATH) / "yamls" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        doc = yaml.safe_load(f)
    if len(doc.get("likelihood", {})) != 1:
        raise ValueError("Template must contain exactly one likelihood")
    return path, doc


def make_run_yaml(template_doc, mass_bin, sector, out_path):
    doc = copy.deepcopy(template_doc)
    _, like = next(iter(doc["likelihood"].items()))
    new_name = f"Wang2026_bin{mass_bin}_{sector}"

    like["Measurement"]["mass_bin"] = int(mass_bin)
    like["Measurement"]["sector"] = str(sector)
    like["TargetData"]["mass_bin"] = int(mass_bin)
    doc["likelihood"] = {new_name: like}

    with open(out_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def main():
    args = parse_args()
    if args.threads_per_fit < 1:
        raise ValueError("--threads-per-fit must be >= 1")

    template_path, template_doc = load_template(args.template)

    papers_file = Path(CAPPIBARAS_PATH) / "Models" / "Papers" / "Wang2026.py"
    if not papers_file.exists():
        raise FileNotFoundError(
            f"{papers_file} is missing. Copy Wang2026.py there before running."
        )

    runchains = Path(CAPPIBARAS_PATH) / "runchains.py"
    if not runchains.exists():
        raise FileNotFoundError(runchains)

    # runchains.py uses OUTPUT_PATH/job_name and force=True, but refusing to
    # reuse an existing run directory avoids accidentally mixing old/new chains.
    collisions = []
    for mass_bin, sector in RUNS:
        job = f"{args.job_prefix}_b{mass_bin}_{sector}"
        if (Path(OUTPUT_PATH) / job).exists():
            collisions.append(str(Path(OUTPUT_PATH) / job))
    if collisions:
        raise FileExistsError(
            "Existing run directories found. Use a new --job-prefix or move/remove "
            "these first:\n  " + "\n  ".join(collisions)
        )

    generated = []
    for mass_bin, sector in RUNS:
        yaml_stem = f"{args.job_prefix}_b{mass_bin}_{sector}"
        yaml_path = Path(CAPPIBARAS_PATH) / "yamls" / f"{yaml_stem}.yaml"
        make_run_yaml(template_doc, mass_bin, sector, yaml_path)
        generated.append((mass_bin, sector, yaml_stem, yaml_path))

    print(f"Template: {template_path}")
    print("Generated six independent YAMLs:")
    for _, _, stem, path in generated:
        print(f"  {stem}: {path}")

    commands = []
    for mass_bin, sector, yaml_stem, _ in generated:
        job_name = f"{args.job_prefix}_b{mass_bin}_{sector}"
        cmd = [sys.executable, str(runchains), job_name, yaml_stem]
        commands.append((job_name, cmd))
        print("\nCommand:", " ".join(cmd))

    if args.dry_run:
        print("\nDry run only; no fits were started.")
        return

    env = os.environ.copy()
    nthread = str(args.threads_per_fit)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = nthread

    processes = []
    print("\nStarting all six fits concurrently...")
    for job_name, cmd in commands:
        run_dir = Path(OUTPUT_PATH) / job_name
        run_dir.mkdir(parents=True, exist_ok=False)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        stdout = open(stdout_path, "w")
        stderr = open(stderr_path, "w")
        proc = subprocess.Popen(
            cmd,
            cwd=str(CAPPIBARAS_PATH),
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        processes.append((job_name, proc, stdout, stderr, stdout_path, stderr_path))
        print(f"  started {job_name} (PID {proc.pid})")

    failures = []
    for job_name, proc, stdout, stderr, stdout_path, stderr_path in processes:
        returncode = proc.wait()
        stdout.close()
        stderr.close()
        if returncode == 0:
            print(f"  completed {job_name}")
        else:
            failures.append((job_name, returncode, stderr_path))
            print(f"  FAILED {job_name} (exit {returncode}); see {stderr_path}")

    if failures:
        details = "\n".join(
            f"  {job}: exit {code}, stderr={path}"
            for job, code, path in failures
        )
        raise RuntimeError(f"One or more fits failed:\n{details}")

    print(f"\nAll six fits completed. Outputs are under: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
