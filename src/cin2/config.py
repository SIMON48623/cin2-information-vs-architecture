from __future__ import annotations

from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "protocol.yaml"


def load_config(path: str | Path | None = None) -> dict:
    target = Path(path) if path else DEFAULT_CONFIG
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def feature_sets(config: dict) -> dict[str, list[str]]:
    f = config["features"]
    prefix = list(f["numeric"]) + list(f["binary"])
    categorical = list(f["categorical"])
    return {"full": prefix + [f["clinician"]] + categorical, "nopf": prefix + categorical}
