from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from utils.data_utils import create_pass_fail_target, identify_score_columns, identify_target_column


TARGET_COLUMN = "pass_fail"
MODEL_FILE = "best_student_model.joblib"
METADATA_FILE = "model_metadata.joblib"


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    working_df = df.copy()
    detected_target = identify_target_column(working_df)

    if detected_target and detected_target != TARGET_COLUMN:
        working_df = create_pass_fail_target(working_df, TARGET_COLUMN)
        working_df = working_df.drop(columns=[detected_target])
    elif TARGET_COLUMN not in working_df.columns:
        working_df = create_pass_fail_target(working_df, TARGET_COLUMN)

    working_df[TARGET_COLUMN] = (
        working_df[TARGET_COLUMN]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"passed": "pass", "failed": "fail", "1": "pass", "0": "fail"})
    )
    working_df = working_df[working_df[TARGET_COLUMN].isin(["pass", "fail"])].copy()

    return working_df, TARGET_COLUMN


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    exclude_score_features: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    excluded_columns: list[str] = []
    if exclude_score_features:
        excluded_columns = identify_score_columns(df.drop(columns=[target_column]))

    drop_columns = [target_column, *excluded_columns]
    X = df.drop(columns=drop_columns, errors="ignore")
    if X.empty:
        raise ValueError(
            "No non-score features are available for training. "
            "Add student profile columns such as attendance, study time, demographics, or preparation details."
        )

    y = df[target_column].map({"fail": 0, "pass": 1})

    if y.isna().any():
        raise ValueError("Target column contains labels other than pass/fail.")

    return X, y.astype(int), excluded_columns


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def get_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
    }


def train_models(df: pd.DataFrame, model_dir: str | Path) -> dict[str, Any]:
    prepared_df, target_column = prepare_training_frame(df)
    X, y, excluded_score_columns = split_features_target(prepared_df, target_column)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    results: list[dict[str, Any]] = []
    trained_models: dict[str, Pipeline] = {}
    best_name = ""
    best_score = -1.0
    best_pipeline: Pipeline | None = None
    best_predictions = None

    for name, model in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        results.append(
            {
                "model": name,
                "accuracy": accuracy,
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
            }
        )
        trained_models[name] = pipeline

        if accuracy > best_score:
            best_name = name
            best_score = accuracy
            best_pipeline = pipeline
            best_predictions = predictions

    if best_pipeline is None or best_predictions is None:
        raise RuntimeError("No model could be trained.")

    labels = ["fail", "pass"]
    matrix = confusion_matrix(y_test, best_predictions, labels=[0, 1]).tolist()
    feature_importance = get_feature_importance(best_pipeline)
    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)

    metadata = {
        "best_model_name": best_name,
        "best_accuracy": best_score,
        "target_column": target_column,
        "feature_columns": X.columns.tolist(),
        "excluded_score_columns": excluded_score_columns,
        "labels": labels,
        "confusion_matrix": matrix,
        "results": results_df,
        "feature_importance": feature_importance,
        "test_size": len(X_test),
        "train_size": len(X_train),
    }

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path / MODEL_FILE)
    joblib.dump(metadata, model_path / METADATA_FILE)

    return {
        "best_model": best_pipeline,
        "best_model_name": best_name,
        "model_path": model_path / MODEL_FILE,
        "metadata_path": model_path / METADATA_FILE,
        "metadata": metadata,
        "trained_models": trained_models,
    }


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    return pd.DataFrame(
        {
            "feature": [name.replace("numeric__", "").replace("categorical__", "") for name in feature_names],
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)


def load_saved_model(model_dir: str | Path) -> tuple[Pipeline | None, dict[str, Any] | None]:
    model_path = Path(model_dir) / MODEL_FILE
    metadata_path = Path(model_dir) / METADATA_FILE

    if not model_path.exists() or not metadata_path.exists():
        return None, None

    return joblib.load(model_path), joblib.load(metadata_path)


def predict_student(model: Pipeline, student_data: dict[str, Any]) -> dict[str, Any]:
    input_df = pd.DataFrame([student_data])
    prediction = int(model.predict(input_df)[0])
    label = "pass" if prediction == 1 else "fail"

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(max(probabilities))

    return {
        "prediction": label,
        "confidence": confidence,
    }
