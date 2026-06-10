from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_dark"
COLOR_SCALE = ["#06b6d4", "#7c3aed", "#22c55e", "#f59e0b", "#ef4444"]


def apply_chart_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb", "family": "Inter, sans-serif"},
        margin={"l": 20, "r": 20, "t": 55, "b": 25},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    return fig


def distribution_chart(df: pd.DataFrame, column: str) -> go.Figure:
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(
            df,
            x=column,
            nbins=30,
            color_discrete_sequence=["#06b6d4"],
            opacity=0.86,
        )
    else:
        counts = df[column].fillna("Missing").astype(str).value_counts().reset_index()
        counts.columns = [column, "count"]
        fig = px.bar(
            counts,
            x=column,
            y="count",
            color=column,
            color_discrete_sequence=COLOR_SCALE,
        )

    return apply_chart_layout(fig, f"Distribution of {column}")


def missing_values_chart(missing_df: pd.DataFrame) -> go.Figure:
    chart_df = missing_df[missing_df["missing_count"] > 0].copy()

    if chart_df.empty:
        chart_df = pd.DataFrame({"column": ["No missing values"], "missing_count": [0]})

    fig = px.bar(
        chart_df,
        x="missing_count",
        y="column",
        orientation="h",
        color="missing_count",
        color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_chart_layout(fig, "Missing Values by Column")


def correlation_heatmap(correlation_df: pd.DataFrame) -> go.Figure:
    if correlation_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No numeric columns available for correlation analysis.",
            showarrow=False,
            font={"size": 16, "color": "#cbd5e1"},
        )
        return apply_chart_layout(fig, "Correlation Heatmap")

    fig = px.imshow(
        correlation_df,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    return apply_chart_layout(fig, "Correlation Heatmap")


def target_pie_chart(df: pd.DataFrame, target_column: str) -> go.Figure:
    counts = df[target_column].fillna("Missing").astype(str).value_counts().reset_index()
    counts.columns = [target_column, "count"]

    fig = px.pie(
        counts,
        names=target_column,
        values="count",
        hole=0.48,
        color_discrete_sequence=COLOR_SCALE,
    )
    return apply_chart_layout(fig, "Pass vs Fail Distribution")


def model_comparison_chart(results_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        results_df.sort_values("accuracy", ascending=True),
        x="accuracy",
        y="model",
        orientation="h",
        color="accuracy",
        text="accuracy",
        color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%")
    return apply_chart_layout(fig, "Model Accuracy Comparison")


def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    if importance_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Feature importance is not available for the selected model.",
            showarrow=False,
            font={"size": 16, "color": "#cbd5e1"},
        )
        return apply_chart_layout(fig, "Feature Importance")

    top_features = importance_df.sort_values("importance", ascending=False).head(15)
    fig = px.bar(
        top_features.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#06b6d4", "#7c3aed"],
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_chart_layout(fig, "Top Feature Importances")


def confusion_matrix_chart(matrix: list[list[int]], labels: list[str]) -> go.Figure:
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale=["#0f172a", "#06b6d4", "#22c55e"],
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
    )
    return apply_chart_layout(fig, "Confusion Matrix")
