"""Collect the per-seed result files into one table, plus paired tests per night.

    python summarize_sleep.py   # merges outputs/sleep_{tta,bal,alpha}_s*.json

For every target: mean +- std over source seeds of pooled macro-F1, per-night macro-F1,
kappa and the rare-stage F1s. The paired test compares two settings night by night
(per-night macro-F1 averaged over seeds first), with a Wilcoxon signed-rank test.
"""
import glob
import json

import numpy as np
from scipy.stats import wilcoxon

from common import save_json
from data.sleep import TARGETS

ROWS = [("source", "argmax"), ("bn", "argmax"), ("tent", "argmax"), ("shot_im", "argmax"),
        ("ent_prior", "argmax"), ("jump", "argmax"), ("hmm_teach", "argmax"),
        ("bn", "em_prior"), ("bn", "hmm_src"), ("bn", "hmm_full"), ("bn", "hmm_dwell"),
        ("tent", "hmm_src"), ("shot_im", "hmm_src"), ("jump", "hmm_src"),
        ("jump", "hmm_dwell"), ("hmm_teach", "hmm_src"), ("hmm_teach", "hmm_dwell"),
        ("source", "hmm_src"), ("pcbn", "argmax"), ("pcbn_iid", "argmax"), ("ubbn", "argmax"),
        ("pcbn", "hmm_src"), ("pcbn_iid", "hmm_src"), ("ubbn", "hmm_src"),
        ("alpha_bn", "argmax"), ("alpha_bn", "hmm_src")]
PAIRS = [(("jump", "argmax"), ("bn", "argmax")),
         (("jump", "argmax"), ("shot_im", "argmax")),
         (("hmm_teach", "argmax"), ("bn", "argmax")),
         (("hmm_teach", "argmax"), ("bn", "hmm_src")),
         (("hmm_teach", "hmm_src"), ("bn", "hmm_src")),
         (("bn", "hmm_dwell"), ("bn", "hmm_src")),
         (("bn", "hmm_src"), ("bn", "argmax")),
         (("pcbn", "argmax"), ("bn", "argmax")),
         (("pcbn", "argmax"), ("source", "argmax")),
         (("pcbn", "argmax"), ("ubbn", "argmax")),
         (("pcbn", "argmax"), ("pcbn_iid", "argmax")),
         (("pcbn", "hmm_src"), ("bn", "hmm_src")),
         (("pcbn", "hmm_src"), ("source", "argmax")),
         (("pcbn", "argmax"), ("alpha_bn", "argmax")),
         (("pcbn", "hmm_src"), ("alpha_bn", "hmm_src")),
         (("pcbn_iid", "hmm_src"), ("alpha_bn", "hmm_src"))]
COLS = ["macro_f1", "night_macro_f1", "kappa", "f1_N1", "f1_N3"]


def main():
    per_seed = {}
    # in this order, so a re-run of some methods (bal, alpha) overrides the main run (tta)
    for f in sum((sorted(glob.glob(f"outputs/sleep_{k}_s*.json")) for k in ("tta", "bal", "alpha")), []):
        for seed, doms in json.load(open(f))["per_seed"].items():
            for dom, res in doms.items():
                # source/bn appear in both runs and are identical; later files fill in the rest
                per_seed.setdefault(seed, {}).setdefault(dom, {}).update(res)
    per_seed = list(per_seed.values())
    summary, lines = {}, [f"{len(per_seed)} source seeds\n"]
    for dom in TARGETS:
        res = [ps[dom] for ps in per_seed]
        lines += [f"\n### {dom} ({len(res[0]['source|argmax']['night_f1s'])} nights)\n",
                  "| method | decoder | " + " | ".join(COLS) + " |",
                  "|---|---|" + "---|" * len(COLS)]
        summary[dom] = {}
        for m, d in ROWS:
            k = f"{m}|{d}"
            vals = {c: [r[k][c] for r in res] for c in COLS}
            summary[dom][k] = {c: [float(np.mean(v)), float(np.std(v))] for c, v in vals.items()}
            lines.append(f"| {m} | {d} | " + " | ".join(
                f"{np.mean(v):.3f} \u00b1 {np.std(v):.3f}" for v in vals.values()) + " |")
        lines += ["", "| comparison (per-night macro-F1) | mean diff | wins | p (Wilcoxon) |",
                  "|---|---|---|---|"]
        for a, b in PAIRS:
            fa = np.mean([r["|".join(a)]["night_f1s"] for r in res], 0)
            fb = np.mean([r["|".join(b)]["night_f1s"] for r in res], 0)
            diff = fa - fb
            p = wilcoxon(fa, fb).pvalue if np.any(diff) else 1.0
            name = f"{'+'.join(a)} vs {'+'.join(b)}"
            summary[dom][name] = {"diff": float(diff.mean()), "wins": int((diff > 0).sum()),
                                  "n": len(diff), "p": float(p)}
            lines.append(f"| {name} | {diff.mean():+.4f} | {(diff > 0).sum()}/{len(diff)} | "
                         f"{p:.2g} |")
    text = "\n".join(lines)
    open("outputs/sleep_summary.md", "w", encoding="utf-8").write(text)
    print(text.replace("\u00b1", "+-"))
    save_json(summary, "outputs/sleep_summary.json")


if __name__ == "__main__":
    main()
