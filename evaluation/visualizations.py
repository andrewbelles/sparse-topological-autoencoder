#!/usr/bin/env python3
#
# visualizations.py  Andrew Belles  April 13th, 2026
#
# Plotting helpers for linear probe evaluation.
#

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fma-topology-evaluation-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


GENRE_ORDER = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]


def ratio_axis_max(frame: pd.DataFrame) -> float:
    if frame.empty or "ratio_percent" not in frame.columns:
        return 100.0
    ratios = pd.to_numeric(frame["ratio_percent"], errors="coerce").dropna()
    if ratios.empty:
        return 100.0
    return max(1.0, float(ratios.max()))


def score_axis_bounds(values: pd.Series | list[float], extra_values: list[float] | None = None) -> tuple[float, float]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if extra_values:
        series = pd.concat([series, pd.to_numeric(pd.Series(extra_values), errors="coerce").dropna()], ignore_index=True)
    if series.empty:
        return 0.0, 1.0
    lower = float(series.min())
    upper = float(series.max())
    span = max(upper - lower, 0.02)
    padding = max(0.015, span * 0.2)
    return max(0.0, lower - padding), min(1.0, upper + padding)


def set_tight_score_axis(axis: plt.Axes, values: pd.Series | list[float], extra_values: list[float] | None = None) -> None:
    lower, upper = score_axis_bounds(values, extra_values)
    axis.set_ylim(lower, upper)


def anchor_baseline(metric_column: str) -> float | None:
    path = Path(__file__).resolve().parent / "data" / "anchor_r100_s00_logistic_summary.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if frame.empty or metric_column not in frame.columns:
        return None
    values = pd.to_numeric(frame[metric_column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def add_anchor_baseline(axis: plt.Axes, metric_column: str | list[str]) -> list[float]:
    columns = [metric_column] if isinstance(metric_column, str) else metric_column
    values = [value for column in columns if (value := anchor_baseline(column)) is not None]
    if not values:
        return []
    label = "Anchor upper bound"
    for index, value in enumerate(values):
        axis.axhline(
            value,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            label=label if index == 0 else None,
        )
    return values


def clean_label(label: str) -> str:
    return label.replace("_", " ").strip().title()


def order_labels(labels: list[str]) -> list[str]:
    cleaned = [clean_label(label) for label in labels]
    order = [label for label in GENRE_ORDER if label in cleaned]
    order.extend(label for label in cleaned if label not in order)
    return order


def save_subset_accuracy_plot(
    subset_frame: pd.DataFrame,
    output_path: Path,
    title: str,
) -> Path:
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)

    plot_frame = subset_frame.copy()
    plot_frame["subset_percent"] = plot_frame["subset_fraction"] * 100.0
    metric_columns = {
        "test_accuracy": ("Accuracy", "#1f77b4"),
        "test_f1_macro": ("F1 Macro", "#ff7f0e"),
        "test_pr_auc_macro": ("PR-AUC Macro", "#2ca02c"),
    }

    for column, (label, color) in metric_columns.items():
        sns.lineplot(
            data=plot_frame,
            x="subset_percent",
            y=column,
            marker="o",
            linewidth=2.0,
            markersize=7,
            ax=axis,
            color=color,
            label=label,
        )

    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_xlabel("Training Subset (%)")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(title="Metric")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
) -> Path:
    sns.set_theme(style="white")
    ordered_labels = order_labels(labels)
    label_to_index = {clean_label(label): index for index, label in enumerate(labels)}
    reorder_indices = [label_to_index[label] for label in ordered_labels]
    ordered_matrix = matrix[np.ix_(reorder_indices, reorder_indices)]

    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    sns.heatmap(
        ordered_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        square=True,
        xticklabels=ordered_labels,
        yticklabels=ordered_labels,
        cbar_kws={"label": "Row-Normalized Accuracy"},
        ax=axis,
    )
    axis.set_xlabel("Predicted Genre")
    axis.set_ylabel("True Genre")
    axis.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def save_ratio_metric_plot(
    summary_frame: pd.DataFrame,
    metric_column: str,
    output_path: Path,
    title: str,
    method_filter: str | None = None,
) -> Path:
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)

    compressed = summary_frame[summary_frame["ratio_percent"].notna()].copy()
    if method_filter is not None:
        compressed = compressed[compressed["method"] == method_filter].copy()

    if not compressed.empty:
        compressed["ratio_percent"] = compressed["ratio_percent"].astype(float)
        if method_filter is None:
            sns.lineplot(
                data=compressed,
                x="ratio_percent",
                y=metric_column,
                hue="method",
                marker="o",
                linewidth=2.0,
                markersize=6,
                errorbar="se",
                err_style="bars",
                err_kws={"elinewidth": 0.9, "capsize": 3, "capthick": 0.9},
                ax=axis,
            )
        else:
            sns.lineplot(
                data=compressed,
                x="ratio_percent",
                y=metric_column,
                marker="o",
                linewidth=2.0,
                markersize=6,
                errorbar="se",
                err_style="bars",
                err_kws={"elinewidth": 0.9, "capsize": 3, "capthick": 0.9},
                ax=axis,
                color="#1f77b4",
                label=clean_label(method_filter),
            )

    max_ratio = ratio_axis_max(compressed)
    anchor_values = add_anchor_baseline(axis, metric_column)

    axis.set_xlabel("Compression Ratio m/N (%)")
    axis.set_ylabel(clean_label(metric_column.replace("test_", "")))
    set_tight_score_axis(axis, compressed[metric_column] if metric_column in compressed else [], anchor_values)
    axis.set_xlim(left=0.0, right=max_ratio)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    if axis.get_legend() is not None:
        axis.legend(title="Method")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def save_dual_metric_method_plot(
    summary_frame: pd.DataFrame,
    output_path: Path,
    title: str,
    method_filter: str | None = None,
) -> Path:
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    plot_frame = summary_frame[summary_frame["ratio_percent"].notna()].copy()
    if method_filter is not None:
        plot_frame = plot_frame[plot_frame["method"] == method_filter].copy()
    if not plot_frame.empty:
        plot_frame["ratio_percent"] = plot_frame["ratio_percent"].astype(float)
        melted = plot_frame.melt(
            id_vars=["method", "ratio_percent", "seed"],
            value_vars=["test_f1_macro", "test_pr_auc_macro"],
            var_name="metric",
            value_name="score",
        )
        melted["metric"] = melted["metric"].map(
            {"test_f1_macro": "Macro-F1", "test_pr_auc_macro": "Macro PR-AUC"}
        )
        sns.lineplot(
            data=melted,
            x="ratio_percent",
            y="score",
            hue="method" if method_filter is None else "metric",
            style="metric" if method_filter is None else None,
            marker="o",
            linewidth=2.0,
            markersize=6,
            errorbar="se",
            err_style="bars",
            err_kws={"elinewidth": 0.9, "capsize": 3, "capthick": 0.9},
            ax=axis,
        )

    anchor_values = add_anchor_baseline(axis, ["test_f1_macro", "test_pr_auc_macro"])
    axis.set_xlabel("Compression Ratio m/d (%)")
    axis.set_ylabel("Score")
    set_tight_score_axis(axis, plot_frame[["test_f1_macro", "test_pr_auc_macro"]].to_numpy().reshape(-1) if not plot_frame.empty else [], anchor_values)
    axis.set_xlim(left=0.0, right=ratio_axis_max(plot_frame))
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    if axis.get_legend() is not None:
        axis.legend(title="")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def save_topology_preservation_plot(
    frame: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    method_filter: str | None = None,
) -> Path:
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    plot_frame = frame[frame["ratio_percent"].notna()].copy()
    if method_filter is not None:
        plot_frame = plot_frame[plot_frame["method"] == method_filter].copy()
    plot_frame["ratio_percent"] = plot_frame["ratio_percent"].astype(float)

    plot_kwargs = {
        "data": plot_frame,
        "x": "ratio_percent",
        "y": value_column,
        "hue": "genre_top",
        "marker": "o",
        "linewidth": 1.8,
        "markersize": 5,
        "ax": axis,
    }
    if method_filter is None:
        plot_kwargs["style"] = "method"
    sns.lineplot(**plot_kwargs)

    axis.set_xlabel("Compression Ratio m/N (%)")
    axis.set_ylabel(clean_label(value_column))
    axis.set_xlim(left=0.0, right=ratio_axis_max(plot_frame))
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    if axis.get_legend() is not None:
        sns.move_legend(axis, "upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def save_topology_metric_panel(
    frame: pd.DataFrame,
    method: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    plot_frame = frame[frame["method"] == method].copy()
    if {"persistence_image_h0", "persistence_image_h1"}.issubset(plot_frame.columns):
        metrics = [
            ("wasserstein_h0", "Wasserstein H0"),
            ("wasserstein_h1", "Wasserstein H1"),
            ("persistence_image_h0", "Persistence Image H0"),
            ("persistence_image_h1", "Persistence Image H1"),
        ]
    else:
        metrics = [
            ("wasserstein_h0", "Wasserstein H0"),
            ("wasserstein_h1", "Wasserstein H1"),
            ("betti_dist_h0", "BettiDist H0"),
            ("betti_dist_h1", "BettiDist H1"),
        ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, (column, title) in zip(axes.flatten(), metrics, strict=True):
        matrix = plot_frame.pivot_table(
            index="genre_top",
            columns="ratio_percent",
            values=column,
            aggfunc="mean",
        )
        genres = [genre for genre in GENRE_ORDER if genre in set(matrix.index)]
        genres.extend(sorted(genre for genre in matrix.index if genre not in set(genres)))
        matrix = matrix.reindex(index=genres, columns=sorted(matrix.columns))
        values = matrix.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        vmax = float(finite.max()) if finite.size else 1.0
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="mako",
            vmin=0.0,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 8},
            linewidths=0.4,
            linecolor="#ffffff",
            cbar_kws={"label": title},
        )
        ax.set_title(title)
        ax.set_xlabel("m/d (%)")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(f"Topology Preservation - {clean_label(method)} vs Anchor", fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_topology_performance_scatter(
    frame: pd.DataFrame,
    topology_metric: str,
    performance_metric: str,
    output_path: Path,
    title: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_frame = frame.dropna(subset=[topology_metric, performance_metric]).copy()
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    if not plot_frame.empty:
        sns.scatterplot(
            data=plot_frame,
            x=topology_metric,
            y=performance_metric,
            hue="method",
            style="genre_top" if "genre_top" in plot_frame.columns else None,
            size="ratio_percent",
            sizes=(35, 130),
            alpha=0.8,
            ax=ax,
        )
    ax.set_title(title)
    ax.set_xlabel(clean_label(topology_metric))
    ax.set_ylabel(clean_label(performance_metric.replace("test_", "")))
    anchor_values = add_anchor_baseline(ax, performance_metric)
    set_tight_score_axis(ax, plot_frame[performance_metric] if performance_metric in plot_frame else [], anchor_values)
    ax.grid(True, alpha=0.25)
    if ax.get_legend() is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
