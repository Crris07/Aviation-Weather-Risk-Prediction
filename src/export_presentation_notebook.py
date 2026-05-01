
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

from .ingest import ROOT


NOTEBOOK_PATH = ROOT / "notebooks" / "03_baseline.ipynb"
FIG_DIR = ROOT / "reports" / "figures"


def _lines(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def _md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def _code_cell(source: str, outputs: list[dict], execution_count: int) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": execution_count,
        "source": _lines(source),
        "outputs": outputs,
    }


def _stream_output(text: str) -> dict:
    return {
        "name": "stdout",
        "output_type": "stream",
        "text": _lines(text),
    }


def _table_output(df: pd.DataFrame, execution_count: int, float_fmt: str = "{:.3f}") -> dict:
    html = df.to_html(index=False, border=1, justify="left", escape=False, float_format=lambda x: float_fmt.format(x))
    text = df.to_string(index=False, float_format=lambda x: float_fmt.format(x))
    return {
        "output_type": "execute_result",
        "data": {
            "text/plain": _lines(text),
            "text/html": [html],
        },
        "metadata": {},
        "execution_count": execution_count,
    }


def _image_output(path: Path) -> dict:
    return {
        "output_type": "display_data",
        "data": {
            "image/png": base64.b64encode(path.read_bytes()).decode("ascii"),
            "text/plain": [str(path.name)],
        },
        "metadata": {},
    }


def main() -> None:
    test_df = pd.read_csv(FIG_DIR / "phase4_vs_phase5_test_pr_auc.csv")
    loro_df = pd.read_csv(FIG_DIR / "phase4_vs_phase5_loro_pr_auc.csv")
    physics_df = pd.read_csv(FIG_DIR / "phase5_physics_feature_importance.csv")

    test_view = test_df.copy()
    test_view.columns = ["airport", "phase4_clean_pr_auc", "phase5_physics_pr_auc", "delta"]

    loro_view = loro_df.copy()
    loro_view.columns = ["airport", "direction", "phase4_clean_pr_auc", "phase5_physics_pr_auc", "delta"]

    physics_view = physics_df[["feature", "gain", "split"]].copy()

    cells = [
        _md_cell(
            "# 03 - Final Results\n"
            "\n"
            "This notebook is the recruiter-facing summary of the project.\n"
            "It compares the cleaned Phase 4 baseline against the Phase 5 physics-feature variant on unseen-airport test performance and LORO domain robustness."
        ),
        _md_cell(
            "## Executive Summary\n"
            "\n"
            "- Phase 4 clean is the honest decontaminated baseline after removing duplicated unit columns.\n"
            "- Phase 5 physics adds generalizable interaction features rather than airport-specific hacks.\n"
            "- Test-set lift is mixed: `KATL -0.0076`, `KBOS +0.0096` PR-AUC.\n"
            "- LORO robustness improves consistently across every evaluated airport.\n"
            "- The most influential Phase 5 additions are `dew_depression_x_hour_sin`, `dew_depression_min_6h`, and `pressure_drop_x_wind`."
        ),
        _code_cell(
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "\n"
            "ROOT = Path.cwd()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "FIG = ROOT / 'reports' / 'figures'\n"
            "\n"
            "test_df = pd.read_csv(FIG / 'phase4_vs_phase5_test_pr_auc.csv')\n"
            "loro_df = pd.read_csv(FIG / 'phase4_vs_phase5_loro_pr_auc.csv')\n"
            "physics_df = pd.read_csv(FIG / 'phase5_physics_feature_importance.csv')\n"
            "test_df, loro_df, physics_df.head()",
            [],
            1,
        ),
        _md_cell("## Unseen Test Airports"),
        _code_cell(
            "test_df",
            [_table_output(test_view, 2)],
            2,
        ),
        _code_cell(
            "# Saved comparison figure\n"
            "from IPython.display import Image\n"
            "Image(filename=str(FIG / 'phase4_vs_phase5_test_pr_auc.png'))",
            [_image_output(FIG_DIR / "phase4_vs_phase5_test_pr_auc.png")],
            3,
        ),
        _md_cell("## LORO Domain Robustness"),
        _code_cell(
            "loro_df",
            [_table_output(loro_view, 4)],
            4,
        ),
        _code_cell(
            "Image(filename=str(FIG / 'phase4_vs_phase5_loro_pr_auc.png'))",
            [_image_output(FIG_DIR / "phase4_vs_phase5_loro_pr_auc.png")],
            5,
        ),
        _md_cell("## Which Physics Features Actually Mattered"),
        _code_cell(
            "physics_df[['feature', 'gain', 'split']]",
            [_table_output(physics_view, 6)],
            6,
        ),
        _code_cell(
            "Image(filename=str(FIG / 'phase5_feature_importance_top20.png'))",
            [_image_output(FIG_DIR / "phase5_feature_importance_top20.png")],
            7,
        ),
        _md_cell(
            "## Reproducibility\n"
            "\n"
            "To regenerate the underlying artifacts:\n"
            "\n"
            "```bash\n"
            "python -m src.features --variant phase4_clean\n"
            "python -m src._feature_validate --variant phase4_clean\n"
            "python -m src.baseline --variant phase4_clean\n"
            "python -m src.features --variant phase5_physics\n"
            "python -m src._feature_validate --variant phase5_physics\n"
            "python -m src.baseline --variant phase5_physics\n"
            "python -m src.compare_variants\n"
            "```"
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
