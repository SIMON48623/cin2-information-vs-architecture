from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from cin2.config import load_config
from cin2.cv import run_oof, write_metadata
from cin2.data import clean_features, load_table
from cin2.models import PRIMARY_FAMILIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic_development.csv")
    parser.add_argument("--sheet", type=lambda value: int(value) if str(value).isdigit() else value, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    frame, y = load_table(args.data, sheet_name=args.sheet)
    frame = clean_features(frame, config)
    groups = config["features"]["groups"]
    canonical = list(config["features"]["numeric"]) + list(config["features"]["binary"]) + [config["features"]["clinician"]] + list(config["features"]["categorical"])
    identity = None
    parts = []
    metadata_runs = {}
    for value in range(1, 32):
        code = format(value, "05b")
        selected = set()
        for bit, (_, members) in zip(code, groups.items()):
            if bit == "1":
                selected.update(members)
        columns = [column for column in canonical if column in selected]
        run = run_oof(frame, y, columns, PRIMARY_FAMILIES, config, args.device, args.fast)
        produced = [c for c in run.predictions.columns if c not in {"y", "fold"}]
        part = run.predictions.rename(columns={f: f"{f}__{code}" for f in produced})
        if identity is None:
            identity = part[["y", "fold"]].copy()
        else:
            if not identity.equals(part[["y", "fold"]]):
                raise RuntimeError("Fold assignments changed between subsets")
        parts.append(part.drop(columns=["y", "fold"]))
        metadata_runs[code] = run.metadata
    combined = pd.concat([identity, *parts], axis=1)
    combined.to_csv(args.output / "oof_groups.csv", index=False)
    available = sorted({column.rsplit("__", 1)[0] for column in combined.columns if column not in {"y", "fold"}})
    write_metadata(args.output / "group_subset_metadata.json", {
        "mode": "FAST synthetic smoke test; estimates are not paper results" if args.fast else "full protocol",
        "subsets": 31, "requested_families": PRIMARY_FAMILIES, "produced_families": available,
        "device": args.device, "runs": metadata_runs,
    })


if __name__ == "__main__":
    main()
