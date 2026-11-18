"""Sleep staging metrics. Macro-F1 is the headline: it weighs the rare stages (N1, N3)
as much as the common ones, so collapsing onto N2 cannot hide behind high accuracy."""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from data.sleep import N_STAGES, STAGES


def score(y, pred):
    per = f1_score(y, pred, labels=range(N_STAGES), average=None, zero_division=0)
    return {"macro_f1": float(per.mean()), "acc": float(accuracy_score(y, pred)),
            "kappa": float(cohen_kappa_score(y, pred)),
            **{f"f1_{s}": float(v) for s, v in zip(STAGES, per)}}


def score_nights(ys, preds):
    """Pooled metrics over all epochs, plus the mean of per-night macro-F1 (each patient
    counts once, however long the night)."""
    out = score(np.concatenate(ys), np.concatenate(preds))
    out["night_macro_f1"] = float(np.mean([night_macro_f1(y, p) for y, p in zip(ys, preds)]))
    return out


def night_macro_f1(y, pred):
    """Macro-F1 over the stages that occur in this night or its prediction (sklearn's
    default). Some elderly nights have no N3 at all; a stage absent from both says nothing."""
    return float(f1_score(y, pred, average="macro", zero_division=0))
