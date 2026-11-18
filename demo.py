"""Quick live demo (~20 s on a GPU, ~10 min on a laptop CPU): load an already-trained model and adapt it
to a few unseen nights, comparing no adaptation, plain BN and our PC-BN, with and without
the HMM decoder.

    python demo.py                          # all 35 elderly (SC-old) nights
    python demo.py --target ST              # insomnia / telemetry nights
    python demo.py --nights 5               # quicker, but 5 nights are too few to trust

Needs the preprocessed data in data/sleep_cache and outputs/sleep_s0/model.pt.
The full study is run_sleep_all.sh (about 2 h on a GPU); this is just a small slice of it.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from common import seed_everything
from data.sleep import TARGETS, load_nights, targets
from eval.sleep_metrics import score_nights
from study_sleep_tta import load, run

SHOW = [("source", "argmax", "No adaptation"),
        ("bn", "argmax", "BN (AdaBN)"),
        ("pcbn", "argmax", "PC-BN (ours)"),
        ("bn", "hmm_src", "BN + HMM"),
        ("pcbn", "hmm_src", "PC-BN + HMM (ours)"),
        ("pcbn_iid", "hmm_src", "PC-BN(iid) + HMM")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="SC-old", choices=TARGETS)
    ap.add_argument("--nights", type=int, default=0, help="first N nights only (0 = all; few nights are noisy)")
    ap.add_argument("--seed", type=int, default=0, choices=range(5))
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU exists")
    args = ap.parse_args()

    gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = "cuda" if gpu and not args.cpu else "cpu"
    seed_everything(args.seed)
    t0 = time.time()
    nights = targets(load_nights())[args.target]
    if args.nights:
        nights = nights[:args.nights]
    model, scale, prior, A = load(args.seed, device)
    hp = json.loads(Path("outputs/sleep_tta_hp.json").read_text())["hp"]

    print(f"\nTarget {args.target}: {len(nights)} night(s), model seed {args.seed}, device {device}")
    methods = sorted({m for m, _, _ in SHOW})
    decoders = sorted({d for _, d, _ in SHOW})
    preds = run(model, scale, prior, A, nights, methods, hp, decoders, device, args.seed)
    ys = [n.y for n in nights]

    print(f"\n{'method':22s} {'macro-F1':>9s} {'kappa':>7s} {'N3 F1':>7s}")
    print("-" * 48)
    for m, d, name in SHOW:
        r = score_nights(ys, preds[(m, d)])
        print(f"{name:22s} {r['macro_f1']:9.3f} {r['kappa']:7.3f} {r['f1_N3']:7.3f}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
