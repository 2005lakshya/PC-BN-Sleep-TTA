"""Train the source sleep scorer on young SC subjects and save what test time may use.

    python train_sleep.py --seed 0

Writes outputs/sleep_s<seed>/model.pt with the weights, the input scale, the source class
prior and the source stage-transition counts. Nothing about the target nights is used.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import save_json, seed_everything
from data.sleep import N_STAGES, Scale, load_nights, split_source, targets, transitions
from eval.sleep_metrics import score_nights
from models.sleepnet import SleepCNN


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/sleep_cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    return ap.parse_args()


@torch.no_grad()
def predict_probs(model, x, batch=1024):
    model.eval()
    return torch.cat([torch.softmax(model(x[i:i + batch]), -1) for i in range(0, len(x), batch)])


def evaluate(model, nights, scale, device):
    preds = [predict_probs(model, scale(n, device)).argmax(-1).cpu().numpy() for n in nights]
    return score_nights([n.y for n in nights], preds)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = "cuda"
    out = Path(f"outputs/sleep_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    nights = load_nights(args.cache)
    split = split_source(nights, args.seed)
    scale = Scale(split["train"])
    x = torch.cat([scale(n, device) for n in split["train"]])
    y = torch.from_numpy(np.concatenate([n.y for n in split["train"]])).to(device)
    counts = np.bincount(y.cpu().numpy(), minlength=N_STAGES)
    print(f"source train {len(split['train'])} nights, {len(y)} epochs, stages {counts}")

    weight = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight / weight.mean())
    model = SleepCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_state, bad, history = -1, None, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(y), device=device)
        total = 0.0
        for i in range(0, len(y), args.batch):
            b = perm[i:i + args.batch]
            xb = x[b] * (1 + 0.1 * torch.randn(len(b), 1, 1, device=device))   # gain jitter
            loss = loss_fn(model(xb), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        val = evaluate(model, split["val"], scale, device)
        history.append({"epoch": epoch, "loss": total / len(y), **val})
        print(f"epoch {epoch:2d} loss {total / len(y):.3f} val macro-F1 {val['macro_f1']:.3f} "
              f"acc {val['acc']:.3f} ({time.time() - t0:.0f}s)")
        if val["macro_f1"] > best:
            best, bad = val["macro_f1"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)

    results = {"args": vars(args), "history": history,
               "source_val": evaluate(model, split["val"], scale, device),
               "source_test": evaluate(model, split["test"], scale, device)}
    torch.save({"state": best_state, "scale": scale.s, "prior": counts / counts.sum(),
                "trans": transitions([n.y for n in split["train"]],
                                     segments_list=[n.segments() for n in split["train"]]),
                "split": {k: [n.name for n in v] for k, v in split.items()}}, out / "model.pt")
    for d, ns in targets(nights).items():
        if ns:
            results[d] = evaluate(model, ns, scale, device)
    for k in ["source_test", "SC-mid", "SC-old", "ST"]:
        if k in results:
            r = results[k]
            print(f"{k:12s} macro-F1 {r['macro_f1']:.3f} acc {r['acc']:.3f} "
                  f"kappa {r['kappa']:.3f} N1 {r['f1_N1']:.3f} N3 {r['f1_N3']:.3f}")
    save_json(results, out / "results.json")


if __name__ == "__main__":
    main()
