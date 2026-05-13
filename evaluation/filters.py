#!/usr/bin/env python3
#
# filters.py  Andrew Belles  May 8th, 2026
#
# Shared config-driven run filtering for evaluation hooks.
#

from fnmatch import fnmatchcase
from pathlib import Path


FILTER_DEFAULTS = {
    "include_methods": [],
    "exclude_methods": [],
    "include_runs": [],
    "exclude_runs": [],
    "include_ratios": [],
    "exclude_ratios": [],
}


def _patterns(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, list | tuple):
        return [str(value) for value in values]
    raise ValueError("filter values must be a string or list")


def _ratio_values(values: object) -> set[int | None]:
    ratios: set[int | None] = set()
    for value in _patterns(values):
        if value.lower() in {"base", "baseline", "none", "null"}:
            ratios.add(None)
        else:
            ratios.add(int(value))
    return ratios


def matches_any(value: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(value, pattern) for pattern in patterns)


def passes_run_filters(run_name: str, method: str, ratio: int | None, config: dict) -> bool:
    include_methods = _patterns(config.get("include_methods", []))
    exclude_methods = _patterns(config.get("exclude_methods", []))
    include_runs = _patterns(config.get("include_runs", []))
    exclude_runs = _patterns(config.get("exclude_runs", []))
    include_ratios = _ratio_values(config.get("include_ratios", []))
    exclude_ratios = _ratio_values(config.get("exclude_ratios", []))

    if include_methods and not matches_any(method, include_methods):
        return False
    if exclude_methods and matches_any(method, exclude_methods):
        return False
    if include_runs and not matches_any(run_name, include_runs):
        return False
    if exclude_runs and matches_any(run_name, exclude_runs):
        return False
    if include_ratios and ratio not in include_ratios:
        return False
    if exclude_ratios and ratio in exclude_ratios:
        return False
    return True


def path_run_name(path: Path, split: str) -> str:
    suffix = f"_{split}"
    if not path.stem.endswith(suffix):
        raise ValueError(f"expected path ending in _{split}: {path}")
    return path.stem[: -len(suffix)]
