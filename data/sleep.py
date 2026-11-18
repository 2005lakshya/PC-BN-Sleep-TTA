"""Sleep-EDF nights from the cache, grouped into the source pool and the shifted targets.

A night is one recording of one person: `x` (n, 3000) EEG epochs and `y` (n,) stages in
time order. Test-time adaptation works one night at a time, because a new patient's night
is exactly the unit a deployed sleep scorer receives.

Domains
    source   SC, age < 50        young healthy sleepers, home recorder
    SC-mid   SC, 50 <= age < 65  mild shift   (used to pick hyper-parameters)
    SC-old   SC, age >= 65       strong age shift: far less N3, more wake
    ST       ST placebo night    insomnia patients, hospital telemetry hardware
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

STAGES = ["W", "N1", "N2", "N3", "REM"]
N_STAGES = len(STAGES)
TARGETS = ["SC-mid", "SC-old", "ST"]


@dataclass
class Night:
    name: str
    cohort: str
    subject: int
    age: int
    x: np.ndarray            # (n, 3000) float16, microvolts
    y: np.ndarray            # (n,) int64
    idx: np.ndarray          # (n,) epoch position in the recording; gaps = dropped epochs

    @property
    def domain(self):
        if self.cohort == "ST":
            return "ST"
        return "source" if self.age < 50 else "SC-mid" if self.age < 65 else "SC-old"

    def segments(self):
        """Index ranges of runs of consecutive epochs (no dropped epoch in between)."""
        cut = np.flatnonzero(np.diff(self.idx) != 1) + 1
        return np.split(np.arange(len(self.y)), cut)


def load_nights(cache_dir="data/sleep_cache"):
    nights = []
    for f in sorted(Path(cache_dir).glob("*.npz")):
        d = np.load(f)
        nights.append(Night(f.stem, str(d["cohort"]), int(d["subject"]), int(d["age"]),
                            d["x"], d["y"].astype(np.int64), d["idx"]))
    if not nights:
        raise FileNotFoundError(f"no nights in {cache_dir}; run `python -m data.sleepedf` first")
    return nights


def split_source(nights, seed, n_val=3, n_test=3):
    """Source nights split by subject (both nights of a person stay together)."""
    src = [n for n in nights if n.domain == "source"]
    subjects = sorted({n.subject for n in src})
    order = np.random.default_rng(seed).permutation(subjects)
    val, test = set(order[:n_val]), set(order[n_val:n_val + n_test])
    return {"train": [n for n in src if n.subject not in val | test],
            "val": [n for n in src if n.subject in val],
            "test": [n for n in src if n.subject in test]}


def targets(nights):
    return {d: [n for n in nights if n.domain == d] for d in TARGETS}


class Scale:
    """One global scale for the raw EEG, fit on source-train only (no per-night z-scoring:
    that would already be a form of test-time adaptation and is tested separately)."""

    def __init__(self, train_nights):
        self.s = float(np.concatenate([n.x[:, ::10].ravel() for n in train_nights])
                       .astype(np.float32).std())

    @classmethod
    def from_value(cls, s):
        obj = cls.__new__(cls)
        obj.s = s
        return obj

    def __call__(self, night, device):
        return torch.from_numpy(night.x.astype(np.float32) / self.s).to(device)[:, None, :]


def transitions(y_list, n=N_STAGES, smooth=1.0, segments_list=None):
    """Counts of stage i -> stage j between consecutive epochs, plus `smooth` per cell."""
    C = np.full((n, n), smooth)
    for k, y in enumerate(y_list):
        segs = segments_list[k] if segments_list else [np.arange(len(y))]
        for s in segs:
            np.add.at(C, (y[s[:-1]], y[s[1:]]), 1)
    return C
