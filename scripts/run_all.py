from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(arguments):
    command = [sys.executable, *map(str, arguments)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete reproducible workflow.")
    parser.add_argument("--synthetic", action="store_true", help="Regenerate and use public synthetic data.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter(); args.output.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        run([ROOT / "synthetic" / "make_synthetic.py"])
    data = args.data or ROOT / "data" / "synthetic_development.csv"
    external = args.external or ROOT / "data" / "synthetic_external.csv"
    common = ["--output", args.output, "--device", args.device] + (["--fast"] if args.fast else [])
    run([ROOT / "analysis" / "05_baseline_table.py", "--data", data, "--output", args.output,
         *(["--fast"] if args.fast else [])])
    run([ROOT / "scripts" / "01_fit_models.py", "--data", data, *common])
    run([ROOT / "scripts" / "02_fit_group_subsets.py", "--data", data, *common])
    run([ROOT / "scripts" / "03_external.py", "--data", data, "--external", external, *common])
    fast_stats = ["--bootstrap", "50", "--fast"] if args.fast else []
    run([ROOT / "analysis" / "10_main_analysis.py", "--oof", args.output / "oof_raw_all.csv",
         "--external", args.output / "external_raw.csv", "--data", data, "--output", args.output, *fast_stats])
    run([ROOT / "analysis" / "11_complementarity.py", "--oof", args.output / "oof_raw_all.csv",
         "--data", data, "--external", args.output / "external_raw.csv", "--external-data", external,
         "--output", args.output, *fast_stats])
    shapley_stats = ["--bootstrap", "50", "--permutations", "100", "--fast"] if args.fast else []
    run([ROOT / "analysis" / "12_group_shapley.py", "--groups", args.output / "oof_groups.csv",
         "--oof", args.output / "oof_raw_all.csv", "--output", args.output, *shapley_stats])
    run([ROOT / "analysis" / "13_sensitivity_age.py", "--oof", args.output / "oof_raw_all.csv",
         "--external", args.output / "external_raw.csv", "--data", data, "--output", args.output, *fast_stats])
    run([ROOT / "analysis" / "14_strata_calibration.py", "--data", data, "--output", args.output,
         *(["--fast"] if args.fast else [])])
    run([ROOT / "analysis" / "20_figures.py", "--results", args.output, "--output", args.output / "figures", *(["--fast"] if args.fast else [])])
    elapsed = time.perf_counter() - started
    (args.output / "run_summary.txt").write_text(
        f"mode={'FAST synthetic smoke test; not paper estimates' if args.fast else 'full protocol'}\n"
        f"device={args.device}\nelapsed_seconds={elapsed:.3f}\n",
        encoding="utf-8",
    )
    print(f"Completed in {elapsed:.1f} seconds", flush=True)


if __name__ == "__main__": main()
