"""Figure for the write-up: pooled macro-F1 per target for the main methods (mean +- std
over source seeds), from outputs/sleep_summary.json.

    python plot_sleep.py        # -> figures/sleep_results.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.sleep import TARGETS

SHOW = [("source|argmax", "No adaptation", "#9e9e9e"),
        ("bn|argmax", "BN (AdaBN)", "#c9a227"),
        ("shot_im|hmm_src", "SHOT-IM + HMM", "#7b8fd4"),
        ("alpha_bn|hmm_src", "α-BN + HMM", "#3d5bb5"),
        ("pcbn|hmm_src", "PC-BN + HMM (ours)", "#c4452b"),
        ("pcbn_iid|hmm_src", "PC-BN(iid) + HMM (ours)", "#e58a6b")]
LABEL = {"SC-mid": "SC 50–64\n(tuning only)", "SC-old": "SC 65+\n(age shift)",
         "ST": "ST insomnia\n(hardware shift)"}


def main():
    s = json.load(open("outputs/sleep_summary.json"))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.8 / len(SHOW)
    for i, (k, name, color) in enumerate(SHOW):
        mean = [s[d][k]["macro_f1"][0] for d in TARGETS]
        std = [s[d][k]["macro_f1"][1] for d in TARGETS]
        ax.bar(np.arange(len(TARGETS)) + (i - (len(SHOW) - 1) / 2) * w, mean, w, yerr=std,
               color=color, label=name, capsize=2, linewidth=0)
    ax.set_xticks(range(len(TARGETS)), [LABEL[d] for d in TARGETS])
    ax.set_ylabel("macro-F1 (5 seeds)")
    ax.set_ylim(0.45, 0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/sleep_results.png", dpi=160)


if __name__ == "__main__":
    main()
