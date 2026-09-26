from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


DROP_RE = re.compile(r"(^|_)(id|name|identifier)(_|$)", re.IGNORECASE)


def load_table(
    path: str | Path,
    outcome: str = "pathology_group",
    sheet_name: str | int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a local table and immediately remove possible direct identifiers."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name=sheet_name)
    else:
        frame = pd.read_csv(path, comment="#")
    frame = frame.drop(columns=[c for c in frame if DROP_RE.search(str(c))], errors="ignore")
    if outcome not in frame:
        raise ValueError(f"Required outcome column is missing: {outcome}")
    y_raw = pd.to_numeric(frame.pop(outcome), errors="coerce")
    keep = y_raw.isin([0, 1])
    return frame.loc[keep].reset_index(drop=True), y_raw.loc[keep].astype(int).to_numpy()


def clean_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = frame.copy()
    f = config["features"]
    for col in f["numeric"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    missing_tokens = {"", "-", "--", "/", "\\", "nan", "none", "null", "unknown", "未知", "不详"}
    def clean_category(value):
        if value is None or pd.isna(value):
            return np.nan
        text = str(value).strip()
        return np.nan if text.lower() in missing_tokens else text
    for col in list(f["binary"]) + list(f["categorical"]) + [f["clinician"]]:
        out[col] = out[col].map(clean_category).astype(object)
    return out
