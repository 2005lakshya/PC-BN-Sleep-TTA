# How to run the project live

There are three ways to run it, from fastest to slowest. **Use option 1 if you can.**

---

## Option 1 — On your own laptop (best: about 30 seconds)

Your laptop already has everything: the data, the trained models and a GPU (RTX 3060).

1. Open a terminal (PowerShell or Command Prompt).
2. Go to the project folder:

   ```
   cd C:\Users\Admin\Downloads\deep_learning
   ```

3. Run the demo on the 35 elderly nights (≈ 20 s):

   ```
   python demo.py
   ```

4. Run it on the 22 insomnia nights (≈ 15 s):

   ```
   python demo.py --target ST
   ```

### What you will see (elderly nights, seed 0)

```
Target SC-old: 35 night(s), model seed 0, device cuda

method                  macro-F1   kappa   N3 F1
------------------------------------------------
No adaptation              0.566   0.513   0.655
BN (AdaBN)                 0.558   0.518   0.500
PC-BN (ours)               0.619   0.609   0.570
BN + HMM                   0.617   0.566   0.527
PC-BN + HMM (ours)         0.634   0.610   0.566
PC-BN(iid) + HMM           0.644   0.615   0.547
```

### What to say while it runs

* "The model was trained on young adults (25–34). These are 35 nights from people over 65 that it has
  never seen, and we have **no labels** for them."
* "**BN (AdaBN)** is the standard trick. It is *worse* than doing nothing (0.558 vs 0.566), and deep
  sleep (N3) collapses from 0.655 to 0.500. That's the problem we found."
* "**PC-BN** is our fix. It jumps to 0.619, and with the HMM decoder to 0.634."
* "This is one seed. The slides show the average over 5 seeds with significance tests."

### Other things you can show

| Command | What it does |
|---|---|
| `python demo.py --seed 2` | same thing with another trained model (seeds 0–4) |
| `python demo.py --target SC-mid` | the 50–64 group (only used for tuning) |
| `python demo.py --nights 5` | only 5 nights — fast but noisy, **don't use it to make a point** |
| open `outputs/sleep_summary.md` | the full results table (all methods, 5 seeds) |
| open `figures/sleep_results.png` | the results figure |

**Note about ST:** on seed 0, `PC-BN + HMM` is lower on ST (0.600) while `PC-BN(iid) + HMM`
is 0.649. That is expected and already in the report (PC-BN + HMM has high seed variance on ST;
the iid variant is stable). If she asks, say exactly that.

---

## Option 2 — On ma'am's computer (copy the folder)

Only if she wants it on *her* machine.

1. Copy the whole `deep_learning` folder to a pen drive (about **1 GB**, mostly `data/sleep_cache`).
   The files the demo really needs are:
   * the `.py` files and the folders `data/`, `models/`, `tta/`, `eval/`
   * `data/sleep_cache/` (the pre-processed nights)
   * `outputs/sleep_s0/model.pt` … `sleep_s4/model.pt` and `outputs/sleep_tta_hp.json`
2. On her computer, install Python 3.10+ and then:

   ```
   pip install -r requirements.txt
   ```

3. Run:

   ```
   python demo.py
   ```

   It picks the GPU automatically if there is one. **Without a GPU it is slow** (about 10 minutes for the 35 elderly nights on a laptop CPU, same numbers),
   so on a CPU-only machine either start it before you begin talking, or run the smaller insomnia target (~6 min):

   ```
   python demo.py --target ST
   ```

---

## Option 3 — Re-run the whole study from scratch (not for a live demo)

This downloads the raw data (~6 GB), trains 5 models and runs every method. It needs a GPU
and takes several hours, so only do it if she asks for full reproduction:

```
pip install -r requirements.txt
python -m data.sleepedf --cache data/sleep_cache
for s in 0 1 2 3 4; do python train_sleep.py --seed $s; done
python study_dwell.py
bash run_sleep_all.sh
```

(The `for` line and `bash` need Git Bash on Windows.) Results land in `outputs/sleep_summary.md`
and `figures/sleep_results.png`.

---

## If something goes wrong

| Error | Fix |
|---|---|
| `No module named torch` (or numpy, mne …) | `pip install -r requirements.txt` |
| `No such file ... model.pt` | you are not inside the `deep_learning` folder, `cd` into it |
| `CUDA out of memory` or a CUDA error | add `--cpu` (slower) |
| it seems frozen on CPU | it's working; use `--target ST` for a shorter run |
