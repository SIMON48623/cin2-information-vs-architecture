from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fast_synthetic_workflow(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_all.py"), "--synthetic", "--fast", "--device", "cpu", "--output", str(tmp_path)], cwd=ROOT, check=True)
    expected = ["numbers.json", "numbers_A.json", "numbers_B.json", "perf_all.csv",
                "sens_age.json", "strata_calibration.json", "table2_baseline.csv"]
    assert all((tmp_path / name).is_file() for name in expected)
    figure_names = ["Figure1_cohort", "Figure2_effect_sizes", "Figure3_group_attribution",
                    "Figure4_complementarity", "Figure5_calibration_dca", "FigureS1_learning_curve"]
    assert all((tmp_path / "figures" / f"{name}.png").is_file() for name in figure_names)
    assert "FAST" in (tmp_path / "run_summary.txt").read_text(encoding="utf-8")
    metadata = json.loads((tmp_path / "fit_models_metadata.json").read_text(encoding="utf-8"))
    full = metadata["runs"]["full"]["families"]
    expected_classes = {
        "lr": "sklearn.linear_model._logistic.LogisticRegression",
        "rf": "sklearn.ensemble._forest.RandomForestClassifier",
        "xgb": "xgboost.sklearn.XGBClassifier",
        "lgbm": "lightgbm.sklearn.LGBMClassifier",
        "catboost": "catboost.core.CatBoostClassifier",
        "ftt": "rtdl_revisiting_models.FTTransformer",
        "tab": "cin2.tab_transformer.FTTransformer",
        "tabmfm_pre030_colid1": "cin2.tabmfm.TabTokTransformer",
        "tabmfm_nopre_colid1": "cin2.tabmfm.TabTokTransformer",
    }
    for family, class_name in expected_classes.items():
        assert full[family]["status"] == "SUCCESS"
        assert full[family]["model_class"] == class_name
    assert full["tabpfn"]["status"] == "SKIPPED (no local weights)"
    assert "tabpfn__full" not in __import__("pandas").read_csv(tmp_path / "oof_raw_all.csv").columns
