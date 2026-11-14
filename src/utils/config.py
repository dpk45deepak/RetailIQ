"""Central configuration: paths, constants and logging.

Keeping every path and magic number in one place means notebooks, the
pipeline, the API and the dashboard can never disagree about where files live.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

RANDOM_STATE = 42
HORIZON_DAYS = 90          # how far ahead the value model looks
HIGH_VALUE_TOP_PCT = 0.20  # top 20% of training-period future revenue = "high value"

# Segment colours: the validated categorical order (blue, orange, aqua, yellow, magenta, violet).
# Verified with the dataviz palette validator (adjacent-pair colour-blind separation) in light AND dark mode.
# Colour is never the only cue: every chart also carries a legend and/or direct labels.
SEGMENT_ORDER = ["Champions", "At-Risk Customers", "Loyal Customers", "New Customers",
                 "Potential Loyalists", "Hibernating Customers"]
SEGMENT_COLORS = dict(zip(SEGMENT_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
SEGMENT_COLORS_DARK = dict(zip(SEGMENT_ORDER, ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]))


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def raw(self) -> Path: return self.root / "data" / "raw"
    @property
    def processed(self) -> Path: return self.root / "data" / "processed"
    @property
    def models(self) -> Path: return self.root / "models"
    @property
    def docs(self) -> Path: return self.root / "docs"

    def ensure(self) -> "ProjectPaths":
        for p in (self.raw, self.processed, self.models, self.docs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: str | Path | None = None) -> ProjectPaths:
    """Resolve the project root: explicit arg > RETAILIQ_ROOT env var > repo root."""
    if root is None:
        root = os.environ.get("RETAILIQ_ROOT") or Path(__file__).resolve().parents[2]
    return ProjectPaths(Path(root).resolve())


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=os.environ.get("RETAILIQ_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
