from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PASS_FAIL_KEYWORDS = ("pass", "fail", "result", "status", "outcome")
SCORE_KEYWORDS = ("score", "mark", "grade")
DEFAULT_PASS_MARK = 60


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and normalize empty strings to missing values."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    return pd.read_csv(csv_path).replace(r"^\s*$", pd.NA, regex=True)


def get_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 3),
    }


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)

    return (
        pd.DataFrame(
            {
                "column": missing_count.index,
                "missing_count": missing_count.values,
                "missing_percent": missing_percent.values,
            }
        )
        .sort_values("missing_count", ascending=False)
        .reset_index(drop=True)
    )


def identify_target_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        normalized = column.lower().strip()
        if any(keyword in normalized for keyword in PASS_FAIL_KEYWORDS):
            unique_values = set(df[column].dropna().astype(str).str.lower().str.strip())
            if unique_values and unique_values.issubset({"pass", "fail", "passed", "failed", "0", "1"}):
                return column

    return None


def identify_score_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    score_columns = [
        column
        for column in numeric_columns
        if any(keyword in column.lower() for keyword in SCORE_KEYWORDS)
    ]

    return score_columns or numeric_columns


def create_pass_fail_target(
    df: pd.DataFrame,
    target_column: str = "pass_fail",
    pass_mark: int = DEFAULT_PASS_MARK,
) -> pd.DataFrame:
    """Create a pass/fail target from average score-like numeric columns."""
    working_df = df.copy()
    existing_target = identify_target_column(working_df)

    if existing_target:
        working_df[target_column] = (
            working_df[existing_target]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"passed": "pass", "failed": "fail", "1": "pass", "0": "fail"})
        )
        return working_df

    score_columns = identify_score_columns(working_df)
    if not score_columns:
        raise ValueError("No numeric score columns were found to create a pass/fail target.")

    average_score = working_df[score_columns].mean(axis=1)
    working_df[target_column] = average_score.apply(lambda score: "pass" if score >= pass_mark else "fail")
    return working_df


def get_correlation_data(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.corr(numeric_only=True).round(3)


def get_feature_options(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "numeric": df.select_dtypes(include="number").columns.tolist(),
        "categorical": df.select_dtypes(exclude="number").columns.tolist(),
        "all": df.columns.tolist(),
    }
