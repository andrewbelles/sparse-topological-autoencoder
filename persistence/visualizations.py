#!/usr/bin/env python3
#
# visualizations.py  Andrew Belles  May 6th, 2026
#
# Plotting helpers for persistence diagrams.
#

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fma-topology-persistence-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from persim import plot_diagrams


def finite_diagram_points(diagram):
    if len(diagram) == 0:
        return diagram
    return diagram[np.isfinite(diagram).all(axis=1)]


def save_persistence_diagram(
    diagrams: list,
    title: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    plot_diagrams(
        diagrams,
        ax=ax,
        show=False,
        legend=True,
        labels=[f"H{index}" for index in range(len(diagrams))],
    )

    ax.set_title(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _axis_limits_for_diagrams(diagram_sets: list) -> tuple[float, float]:
    finite_sets = [finite_diagram_points(values) for values in diagram_sets if len(values) > 0]
    finite_sets = [values for values in finite_sets if len(values) > 0]

    if not finite_sets:
        return 0.0, 1.0

    stacked = np.concatenate(finite_sets, axis=0)
    axis_min = float(min(stacked[:, 0].min(), stacked[:, 1].min()))
    axis_max = float(max(stacked[:, 0].max(), stacked[:, 1].max()))
    margin = max(1e-6, 0.05 * (axis_max - axis_min))
    return axis_min - margin, axis_max + margin


def _plot_residual_axis(
    ax,
    genre_diagram,
    residual_diagram,
    null_diagram,
    threshold: float,
    axis_limits: tuple[float, float],
    title: str,
    show_ylabel: bool,
) -> None:
    genre = finite_diagram_points(genre_diagram)
    residual = finite_diagram_points(residual_diagram)
    background = finite_diagram_points(null_diagram)
    axis_min, axis_max = axis_limits

    ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", color="#222222", linewidth=1.5)

    if len(background) > 0:
        ax.scatter(
            background[:, 0],
            background[:, 1],
            s=18,
            color="#7c8da5",
            alpha=0.32,
            label="Null background",
        )

    if len(genre) > 0:
        ax.scatter(
            genre[:, 0],
            genre[:, 1],
            s=24,
            color="#d8d8d8",
            edgecolor="#9a9a9a",
            linewidth=0.3,
            alpha=0.75,
            label="Genre features",
        )

    if len(residual) > 0:
        ax.scatter(
            residual[:, 0],
            residual[:, 1],
            s=52,
            color="#d1495b",
            edgecolor="#711f2a",
            linewidth=0.6,
            alpha=0.95,
            label="Residual salient",
        )

    if threshold > 0.0:
        threshold_x_end = axis_max - threshold
        if threshold_x_end > axis_min:
            ax.plot(
                [axis_min, threshold_x_end],
                [axis_min + threshold, axis_max],
                linestyle=":",
                color="#d1495b",
                linewidth=1.8,
                label=f"Null lifetime threshold={threshold:.2f}",
            )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_title(title)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death" if show_ylabel else "")
    if len(genre) == 0 and len(residual) == 0 and len(background) == 0:
        ax.text(0.5, 0.5, "No finite features", transform=ax.transAxes, ha="center", va="center")


def save_residual_persistence_diagram(
    genre_diagrams: list,
    residual_diagrams: list,
    null_diagrams: list,
    thresholds: dict[int, float],
    title: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    homology_dims = list(range(len(genre_diagrams)))
    fig, axes = plt.subplots(
        1,
        len(homology_dims),
        figsize=(7.5 * len(homology_dims), 6),
        sharey=False,
        constrained_layout=True,
    )
    if len(homology_dims) == 1:
        axes = [axes]

    for ax, homology_dim in zip(axes, homology_dims):
        axis_limits = _axis_limits_for_diagrams(
            [genre_diagrams[homology_dim], residual_diagrams[homology_dim], null_diagrams[homology_dim]]
        )
        _plot_residual_axis(
            ax,
            genre_diagrams[homology_dim],
            residual_diagrams[homology_dim],
            null_diagrams[homology_dim],
            thresholds.get(homology_dim, 0.0),
            axis_limits,
            title=f"H{homology_dim}",
            show_ylabel=homology_dim == homology_dims[0],
        )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", fontsize=9, frameon=True)

    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_residual_source_comparison(
    source_records: dict[str, dict[str, object]],
    genre: str,
    title: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    preferred_order = ["anchor"]
    sources = [source for source in preferred_order if source in source_records]
    sources.extend(sorted(source for source in source_records if source not in set(preferred_order)))
    if not sources:
        raise ValueError("source_records must contain at least one source")

    homology_dims = list(range(len(next(iter(source_records.values()))["diagrams"])))
    fig, axes = plt.subplots(
        len(sources),
        len(homology_dims),
        figsize=(7.5 * len(homology_dims), 5.4 * len(sources)),
        sharex="col",
        sharey="col",
        constrained_layout=True,
    )
    if len(sources) == 1 and len(homology_dims) == 1:
        axes = np.array([[axes]])
    elif len(sources) == 1:
        axes = np.array([axes])
    elif len(homology_dims) == 1:
        axes = np.array([[axis] for axis in axes])

    axis_limits_by_dim: dict[int, tuple[float, float]] = {}
    for homology_dim in homology_dims:
        diagram_sets = []
        for source in sources:
            record = source_records[source]
            diagram_sets.extend(
                [
                    record["diagrams"][homology_dim],
                    record["residual_diagrams"][homology_dim],
                    record["null_diagrams"][homology_dim],
                ]
            )
        axis_limits_by_dim[homology_dim] = _axis_limits_for_diagrams(diagram_sets)

    for row_index, source in enumerate(sources):
        record = source_records[source]
        for column_index, homology_dim in enumerate(homology_dims):
            _plot_residual_axis(
                axes[row_index, column_index],
                record["diagrams"][homology_dim],
                record["residual_diagrams"][homology_dim],
                record["null_diagrams"][homology_dim],
                record["thresholds"].get(homology_dim, 0.0),
                axis_limits_by_dim[homology_dim],
                title=f"{source.replace('_', ' ').title()} H{homology_dim}",
                show_ylabel=column_index == 0,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(loc="upper left", fontsize=9, frameon=True)
    fig.suptitle(f"{title} - {genre}", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_within_genre_variation_plot(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_frame = frame.copy()
    plot_frame["homology"] = plot_frame["homology_dim"].apply(lambda value: f"H{int(value)}")

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    sns.barplot(
        data=plot_frame,
        x="genre_top",
        y="mean_pairwise_wasserstein",
        hue="homology",
        errorbar=None,
        palette=["#4c78a8", "#f58518"],
        ax=ax,
    )
    sns.stripplot(
        data=plot_frame,
        x="genre_top",
        y="mean_pairwise_wasserstein",
        hue="homology",
        dodge=True,
        palette=["#1f1f1f"] * int(plot_frame["homology"].nunique()),
        alpha=0.45,
        size=4,
        legend=False,
        ax=ax,
    )
    ax.set_title("Within-Genre Topology Variation")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Pairwise Diagram Wasserstein")
    ax.tick_params(axis="x", rotation=30)
    if ax.get_legend() is not None:
        ax.legend(title="Homology")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _source_label(source: str) -> str:
    return source.replace("_", " ").title()


def save_within_genre_variation_source_panel(
    frame: pd.DataFrame,
    output_path: Path,
    sources: list[str],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    homology_dims = sorted(frame["homology_dim"].unique())
    fig, axes = plt.subplots(
        len(sources),
        len(homology_dims),
        figsize=(7.2 * len(homology_dims), 4.8 * len(sources)),
        constrained_layout=True,
        squeeze=False,
    )

    for row_index, source in enumerate(sources):
        for column_index, homology_dim in enumerate(homology_dims):
            ax = axes[row_index, column_index]
            subset = frame[(frame["source"] == source) & (frame["homology_dim"] == homology_dim)].copy()
            sns.barplot(
                data=subset,
                x="genre_top",
                y="mean_pairwise_wasserstein",
                errorbar=None,
                color="#4c78a8" if homology_dim == 0 else "#f58518",
                ax=ax,
            )
            ax.set_title(f"{_source_label(source)} H{int(homology_dim)}")
            ax.set_xlabel("")
            ax.set_ylabel("Mean Pairwise Wasserstein" if column_index == 0 else "")
            if not subset.empty:
                ymax = float(subset["mean_pairwise_wasserstein"].max())
                ax.set_ylim(0.0, max(ymax * 1.12, 1e-6))
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Within-Genre Topology Variation by Basis", fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_genre_covariance_heatmaps(frame: pd.DataFrame, value_column: str, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    homology_dims = sorted(frame["homology_dim"].unique())
    fig, axes = plt.subplots(
        1,
        len(homology_dims),
        figsize=(7.2 * len(homology_dims), 6.2),
        constrained_layout=True,
    )
    if len(homology_dims) == 1:
        axes = [axes]

    for ax, homology_dim in zip(axes, homology_dims):
        subset = frame[frame["homology_dim"] == homology_dim]
        matrix = subset.pivot(index="genre_a", columns="genre_b", values=value_column)
        genres = sorted(matrix.index)
        matrix = matrix.reindex(index=genres, columns=genres)

        finite_values = np.abs(matrix.to_numpy(dtype=float))
        finite_values = finite_values[np.isfinite(finite_values)]
        vmax = float(finite_values.max()) if finite_values.size else 1.0
        if value_column == "correlation":
            vmin, vmax, center = -1.0, 1.0, 0.0
            cmap = "vlag"
        else:
            vmin, center = -vmax, 0.0
            cmap = "vlag"

        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            annot=True,
            fmt=".2f" if value_column == "correlation" else ".1f",
            annot_kws={"fontsize": 8},
            square=True,
            linewidths=0.4,
            linecolor="#ffffff",
            cbar_kws={"label": value_column.replace("_", " ").title()},
        )
        ax.set_title(f"H{int(homology_dim)}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_genre_covariance_source_panel(
    frame: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    sources: list[str],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    homology_dims = sorted(frame["homology_dim"].unique())
    fig, axes = plt.subplots(
        len(sources),
        len(homology_dims),
        figsize=(7.2 * len(homology_dims), 6.0 * len(sources)),
        constrained_layout=True,
        squeeze=False,
    )

    for row_index, source in enumerate(sources):
        for column_index, homology_dim in enumerate(homology_dims):
            ax = axes[row_index, column_index]
            subset = frame[(frame["source"] == source) & (frame["homology_dim"] == homology_dim)]
            matrix = subset.pivot(index="genre_a", columns="genre_b", values=value_column)
            genres = sorted(matrix.index)
            matrix = matrix.reindex(index=genres, columns=genres)

            if value_column == "correlation":
                vmin, vmax, center = -1.0, 1.0, 0.0
                cmap = "vlag"
            else:
                values = matrix.to_numpy(dtype=float)
                values = np.abs(values[np.isfinite(values)])
                vmax = float(values.max()) if values.size else 1.0
                vmin, center = -vmax, 0.0
                cmap = "vlag"

            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                annot=True,
                fmt=".2f" if value_column == "correlation" else ".1f",
                annot_kws={"fontsize": 8},
                square=True,
                linewidths=0.4,
                linecolor="#ffffff",
                cbar_kws={"label": value_column.replace("_", " ").title()},
            )
            ax.set_title(f"{_source_label(source)} H{int(homology_dim)}")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=35)
            ax.tick_params(axis="y", rotation=0)

    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_between_genre_distance_heatmaps(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    homology_dims = sorted(frame["homology_dim"].unique())
    fig, axes = plt.subplots(
        1,
        len(homology_dims),
        figsize=(7.2 * len(homology_dims), 6.2),
        constrained_layout=True,
    )
    if len(homology_dims) == 1:
        axes = [axes]

    for ax, homology_dim in zip(axes, homology_dims):
        subset = frame[frame["homology_dim"] == homology_dim]
        matrix = subset.pivot(index="genre_a", columns="genre_b", values="mean_wasserstein")
        genres = sorted(matrix.index)
        matrix = matrix.reindex(index=genres, columns=genres)
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="mako",
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 8},
            square=True,
            linewidths=0.4,
            linecolor="#ffffff",
            cbar_kws={"label": "Mean Wasserstein"},
        )
        ax.set_title(f"H{int(homology_dim)}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Between-Genre Topology Distance", fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_between_genre_distance_source_panel(
    frame: pd.DataFrame,
    output_path: Path,
    sources: list[str],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    homology_dims = sorted(frame["homology_dim"].unique())
    fig, axes = plt.subplots(
        len(sources),
        len(homology_dims),
        figsize=(7.2 * len(homology_dims), 6.0 * len(sources)),
        constrained_layout=True,
        squeeze=False,
    )

    for row_index, source in enumerate(sources):
        for column_index, homology_dim in enumerate(homology_dims):
            ax = axes[row_index, column_index]
            subset = frame[(frame["source"] == source) & (frame["homology_dim"] == homology_dim)]
            matrix = subset.pivot(index="genre_a", columns="genre_b", values="mean_wasserstein")
            genres = sorted(matrix.index)
            matrix = matrix.reindex(index=genres, columns=genres)
            values = matrix.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            vmax = float(values.max()) if values.size else 1.0
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="mako",
                vmin=0.0,
                vmax=vmax,
                annot=True,
                fmt=".1f",
                annot_kws={"fontsize": 8},
                square=True,
                linewidths=0.4,
                linecolor="#ffffff",
                cbar_kws={"label": "Mean Wasserstein"},
            )
            ax.set_title(f"{_source_label(source)} H{int(homology_dim)}")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=35)
            ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Between-Genre Topology Distance by Basis", fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_topology_meaningfulness_plot(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_frame = frame.copy()
    plot_frame["homology"] = plot_frame["homology_dim"].apply(lambda value: f"H{int(value)}")
    plot_frame["source_label"] = plot_frame["source"].apply(_source_label)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    sns.barplot(
        data=plot_frame,
        x="source_label",
        y="between_within_ratio",
        hue="homology",
        errorbar=None,
        palette=["#4c78a8", "#f58518"],
        ax=axes[0],
    )
    axes[0].axhline(1.0, linestyle="--", color="#222222", linewidth=1.2)
    axes[0].set_title("Between / Within Topology Distance")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Ratio")
    axes[0].legend(title="Homology")

    melted = plot_frame.melt(
        id_vars=["source_label", "homology"],
        value_vars=["mean_within_wasserstein", "mean_between_wasserstein"],
        var_name="distance_type",
        value_name="mean_wasserstein",
    )
    melted["distance_type"] = melted["distance_type"].map(
        {
            "mean_within_wasserstein": "Within Genre",
            "mean_between_wasserstein": "Between Genres",
        }
    )
    sns.barplot(
        data=melted,
        x="source_label",
        y="mean_wasserstein",
        hue="distance_type",
        errorbar=None,
        palette=["#72b7b2", "#e45756"],
        ax=axes[1],
    )
    axes[1].set_title("Mean Topology Distance")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean Wasserstein")
    axes[1].legend(title="")

    fig.suptitle("Topology Meaningfulness by Basis", fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
