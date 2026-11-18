"""Main experiment: every adaptation method x every decoder, one night at a time.

    python study_sleep_tta.py --tune --seeds 0           # pick hyper-parameters on SC-mid
    python study_sleep_tta.py --seeds 0 1 2 3 4          # final run on all targets

Hyper-parameters are chosen on SC-mid only (the mild-shift target) and then frozen for
SC-old and ST, so the headline targets never influence any choice. Gradient methods share
one grid (lr x steps); `jump` additionally gets its loss weight from the grid.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import save_json, seed_everything
from data.sleep import TARGETS, Scale, load_nights, targets
from eval.sleep_metrics import night_macro_f1, score_nights
from models.sleepnet import SleepCNN
from tta.hmm import DECODERS, decode
from tta.methods import BALANCED, METHODS, adapt

GRAD = ["tent", "shot_im", "ent_prior", "jump", "hmm_teach"]
GRID = {"lr": [3e-4, 1e-3, 3e-3], "steps": [20, 60]}
LAMBDAS = [0.3, 1.0, 3.0]
KAPPAS = [5.0, 20.0, 80.0]
BALANCED_GRID = {"tau": [5.0, 20.0, 80.0], "rounds": [1, 2, 4]}
DEFAULT = {"lr": 1e-3, "steps": 20, "lam": 1.0, "kappa": 20.0}


def load(seed, device):
    ck = torch.load(f"outputs/sleep_s{seed}/model.pt", weights_only=False, map_location=device)
    model = SleepCNN().to(device)
    model.load_state_dict(ck["state"])
    A = ck["trans"] / ck["trans"].sum(1, keepdims=True)
    return model, Scale.from_value(ck["scale"]), ck["prior"], A


def run(model, scale, prior, A, nights, methods, hp, decoders, device, seed):
    """-> {(method, decoder): [pred per night]}"""
    preds = {}
    for night in nights:
        x = scale(night, device)
        segs = night.segments()
        for m in methods:
            h = hp.get(m, DEFAULT)
            p = adapt(model, x, m, prior, A, steps=h["steps"], lr=h["lr"], lam=h.get("lam", 1.0),
                      seed=seed, segments=segs, rounds=h.get("rounds", 2),
                      tau=h.get("tau", 20.0), alpha=h.get("alpha", 0.5)).cpu().numpy().astype(np.float64)
            for d in decoders:
                q = decode(p, prior, A, segs, d, kappa=h.get("kappa", DEFAULT["kappa"]))
                preds.setdefault((m, d), []).append(q.argmax(1))
    return preds


def tune(nights, device):
    """Grid search on SC-mid, seed 0, scored by mean per-night macro-F1."""
    model, scale, prior, A = load(0, device)
    mid = targets(nights)["SC-mid"]
    ys = [n.y for n in mid]
    hp, log = {}, []
    for m in GRAD:
        best = None
        lams = LAMBDAS if m == "jump" else [1.0]
        for lr, steps, lam in itertools.product(GRID["lr"], GRID["steps"], lams):
            h = {"lr": lr, "steps": steps, "lam": lam}
            pr = run(model, scale, prior, A, mid, [m], {m: h}, ["argmax"], device, 0)
            f = score_nights(ys, pr[(m, "argmax")])["night_macro_f1"]
            log.append({"method": m, **h, "night_macro_f1": f})
            print(f"  {m:9s} lr {lr:g} steps {steps} lam {lam:g}: {f:.4f}")
            if best is None or f > best[0]:
                best = (f, h)
        hp[m] = best[1]
    # decoder pseudo-count, tuned once on the plain BN-adapted model
    best = None
    for k in KAPPAS:
        pr = run(model, scale, prior, A, mid, ["bn"], {"bn": {**DEFAULT, "kappa": k}},
                 ["hmm_dwell"], device, 0)
        f = score_nights(ys, pr[("bn", "hmm_dwell")])["night_macro_f1"]
        log.append({"decoder": "hmm_dwell", "kappa": k, "night_macro_f1": f})
        print(f"  hmm_dwell kappa {k:g}: {f:.4f}")
        if best is None or f > best[0]:
            best = (f, k)
    for m in METHODS:
        hp.setdefault(m, dict(DEFAULT))
        hp[m]["kappa"] = best[1]
    return hp, log


def tune_balanced(nights, device, hp):
    """Same protocol for the balanced-BN variants (added after the first round): SC-mid,
    seed 0, per-night macro-F1 of the plain argmax prediction."""
    model, scale, prior, A = load(0, device)
    mid = targets(nights)["SC-mid"]
    ys = [n.y for n in mid]
    log = []
    for m in [b for b in BALANCED if "tau" not in hp.get(b, {})]:   # tune() fills defaults only
        best = None
        for tau, rounds in itertools.product(BALANCED_GRID["tau"], BALANCED_GRID["rounds"]):
            h = {**DEFAULT, "kappa": hp["bn"]["kappa"], "tau": tau, "rounds": rounds}
            pr = run(model, scale, prior, A, mid, [m], {m: h}, ["argmax"], device, 0)
            f = score_nights(ys, pr[(m, "argmax")])["night_macro_f1"]
            log.append({"method": m, "tau": tau, "rounds": rounds, "night_macro_f1": f})
            print(f"  {m:9s} tau {tau:g} rounds {rounds}: {f:.4f}")
            if best is None or f > best[0]:
                best = (f, h)
        hp[m] = best[1]
    best = None
    for alpha in [0.25, 0.5, 0.75]:
        h = {**DEFAULT, "kappa": hp["bn"]["kappa"], "alpha": alpha}
        pr = run(model, scale, prior, A, mid, ["alpha_bn"], {"alpha_bn": h}, ["argmax"], device, 0)
        f = score_nights(ys, pr[("alpha_bn", "argmax")])["night_macro_f1"]
        log.append({"method": "alpha_bn", "alpha": alpha, "night_macro_f1": f})
        print(f"  alpha_bn  alpha {alpha:g}: {f:.4f}")
        if best is None or f > best[0]:
            best = (f, h)
    hp["alpha_bn"] = best[1]
    return hp, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune-balanced", action="store_true")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--hp", default="outputs/sleep_tta_hp.json")
    ap.add_argument("--out", default="outputs/sleep_tta_results.json")
    args = ap.parse_args()
    device = "cuda"
    nights = load_nights()

    if args.tune:
        hp, log = tune(nights, device)
        save_json({"hp": hp, "log": log}, args.hp)
        print(json.dumps(hp, indent=1))
        return

    saved = json.loads(Path(args.hp).read_text())
    hp = saved["hp"]
    if args.tune_balanced:
        hp, log = tune_balanced(nights, device, hp)
        saved["log"] += log
        save_json(saved, args.hp)
        print(json.dumps({m: hp[m] for m in list(BALANCED) + ["alpha_bn"]}, indent=1))
        return
    tgt = targets(nights)
    results = {"hp": hp, "per_seed": {}}
    for seed in args.seeds:
        seed_everything(seed)
        model, scale, prior, A = load(seed, device)
        results["per_seed"][seed] = {}
        for dom in TARGETS:
            t0 = time.time()
            pr = run(model, scale, prior, A, tgt[dom], args.methods, hp, DECODERS, device, seed)
            ys = [n.y for n in tgt[dom]]
            res = {}
            for (m, d), p in pr.items():
                res[f"{m}|{d}"] = score_nights(ys, p)
                res[f"{m}|{d}"]["night_f1s"] = [night_macro_f1(y, q)
                                                for y, q in zip(ys, p)]
            results["per_seed"][seed][dom] = res
            print(f"seed {seed} {dom:7s} ({time.time() - t0:.0f}s)  " + "  ".join(
                f"{m} {res[m + '|argmax']['macro_f1']:.3f}/{res[m + '|hmm_src']['macro_f1']:.3f}"
                for m in args.methods))
            save_json(results, args.out)


if __name__ == "__main__":
    main()
