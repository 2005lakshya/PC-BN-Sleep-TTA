"""Premise check, labels only, no model: where does the stage dynamics shift live?

Write each domain's stage-transition matrix as A = diag(s) + diag(1 - s) J:
    s  dwell  = probability of staying in a stage for one more epoch
    J  jumps  = where a stage goes when it ends
Score every target night's true stage sequence (log-likelihood per transition) under
    src     the source matrix, unchanged
    dwell   source J, the target's own s           (5 free numbers)
    jump    source s, the target's own J           (15 free numbers)
    full    the target's own matrix                (20 free numbers, upper bound)
and report what fraction of the src -> full gain each partial refit recovers. If `dwell`
recovers most of it, the shift is mainly in how long stages last, which is what the
`jump` loss and the `hmm_dwell` decoder rely on.

    python study_dwell.py
"""
import numpy as np

from common import save_json
from data.sleep import STAGES, load_nights, targets, transitions
from tta.hmm import join_dwell, split_dwell


def norm(C):
    return C / C.sum(1, keepdims=True)


def loglik(C_counts, A):
    """Mean log-likelihood per transition of observed counts under matrix A."""
    return float((C_counts * np.log(A)).sum() / C_counts.sum())


def main():
    nights = load_nights()
    src = [n for n in nights if n.domain == "source"]
    A_src = norm(transitions([n.y for n in src], segments_list=[n.segments() for n in src]))
    s_src, J_src = split_dwell(A_src)
    out = {"source": {"prior": np.bincount(np.concatenate([n.y for n in src]), minlength=5)
                      .tolist(), "dwell_epochs": (1 / (1 - s_src)).tolist(),
                      "jump": J_src.tolist()}}
    print("source  mean dwell (epochs):", np.round(1 / (1 - s_src), 1).tolist())
    for d, ns in targets(nights).items():
        # domain level: pooled counts, fitted and scored on the same pooled counts
        C = transitions([n.y for n in ns], smooth=0.0, segments_list=[n.segments() for n in ns])
        A_t = norm(C + 1.0)
        s_t, J_t = split_dwell(A_t)
        ll = {"src": loglik(C, A_src), "dwell": loglik(C, join_dwell(s_t, J_src)),
              "jump": loglik(C, join_dwell(s_src, J_t)), "full": loglik(C, A_t)}
        gap = ll["full"] - ll["src"]
        rec = {k: (ll[k] - ll["src"]) / gap for k in ("dwell", "jump")}

        # night level: each night refit on its own labels (what a per-night method could do)
        per = []
        for n in ns:
            c = transitions([n.y], smooth=0.0, segments_list=[n.segments()])
            a = norm(c + 1.0)
            s_n, J_n = split_dwell(a)
            l = [loglik(c, A_src), loglik(c, join_dwell(s_n, J_src)),
                 loglik(c, join_dwell(s_src, J_n)), loglik(c, a)]
            per.append([(l[1] - l[0]) / (l[3] - l[0]), (l[2] - l[0]) / (l[3] - l[0])])
        per = np.array(per)
        prior = np.bincount(np.concatenate([n.y for n in ns]), minlength=5)
        out[d] = {"nights": len(ns), "prior": prior.tolist(), "loglik": ll, "recovered": rec,
                  "night_recovered_median": {"dwell": float(np.median(per[:, 0])),
                                             "jump": float(np.median(per[:, 1]))},
                  "dwell_epochs": (1 / (1 - s_t)).tolist(), "jump": J_t.tolist(),
                  "jump_row_tv": (0.5 * np.abs(J_t - J_src).sum(1)).tolist()}
        print(f"\n{d}: {len(ns)} nights, stage mix "
              f"{dict(zip(STAGES, np.round(prior / prior.sum(), 3).tolist()))}")
        print(f"  mean dwell (epochs): {np.round(1 / (1 - s_t), 1).tolist()}")
        print(f"  log-lik/transition: " + "  ".join(f"{k} {v:.4f}" for k, v in ll.items()))
        print(f"  share of shift recovered  domain: dwell {rec['dwell']:.0%}  jump {rec['jump']:.0%}"
              f"   per night (median): dwell {np.median(per[:, 0]):.0%}  "
              f"jump {np.median(per[:, 1]):.0%}")
    save_json(out, "outputs/sleep_dwell_premise.json")


if __name__ == "__main__":
    main()
