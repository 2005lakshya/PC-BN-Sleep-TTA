"""Download Sleep-EDF Expanded and turn each night into a small cache file.

Only one EEG channel (Fpz-Cz, 100 Hz) is kept, cut into 30 s epochs and labelled with the
five AASM stages W, N1, N2, N3, REM (R&K stages 3 and 4 are merged into N3). Each raw EDF is
deleted as soon as it is converted, so the whole thing needs well under 1 GB of disk.

Which nights are fetched:
    SC (healthy, home recordings), age < 65 : both nights   (source pool + mild-shift target)
    SC, age >= 65                           : first night   (elderly target)
    ST (insomnia study, hospital telemetry) : placebo night (cohort / hardware target)

    python -m data.sleepedf --cache data/sleep_cache
"""
import argparse
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

MIRRORS = ["https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/",
           "https://physionet.org/files/sleep-edfx/1.0.0/"]
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_OF = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
            "Sleep stage 4": 3, "Sleep stage R": 4}           # '?' and movement are dropped
EPOCH_S, FS = 30, 100
WAKE_EDGE = 60                                                 # keep 30 min of wake around sleep


def listing(session, folder):
    html = session.get(MIRRORS[1] + folder + "/", timeout=60).text
    return sorted(set(re.findall(r'>((?:SC|ST)\w+-(?:PSG|Hypnogram)\.edf)<', html)))


def plan(session):
    """Returns a list of dicts: psg/hyp paths plus cohort, subject, night and age."""
    sc_meta = pd.read_excel(io.BytesIO(session.get(MIRRORS[1] + "SC-subjects.xls").content))
    st_meta = pd.read_excel(io.BytesIO(session.get(MIRRORS[1] + "ST-subjects.xls").content),
                            header=[0, 1])
    jobs = []
    files = listing(session, "sleep-cassette")
    for row in sc_meta.itertuples():
        if row.age >= 65 and row.night != 1:
            continue
        key = f"SC4{row.subject:02d}{row.night}"
        psg = [f for f in files if f.startswith(key) and f.endswith("PSG.edf")]
        hyp = [f for f in files if f.startswith(key) and f.endswith("Hypnogram.edf")]
        if psg and hyp:                      # a few subject/night pairs were never recorded
            jobs.append(dict(cohort="SC", subject=int(row.subject), night=int(row.night),
                             age=int(row.age), psg="sleep-cassette/" + psg[0],
                             hyp="sleep-cassette/" + hyp[0]))
    files = listing(session, "sleep-telemetry")
    for _, row in st_meta.iterrows():
        subj, night = int(row.iloc[0]), int(row.iloc[3])
        key = f"ST7{subj:02d}{night}"
        psg = [f for f in files if f.startswith(key) and f.endswith("PSG.edf")]
        hyp = [f for f in files if f.startswith(key) and f.endswith("Hypnogram.edf")]
        if psg and hyp:
            jobs.append(dict(cohort="ST", subject=subj, night=night, age=int(row.iloc[1]),
                             psg="sleep-telemetry/" + psg[0], hyp="sleep-telemetry/" + hyp[0]))
    return jobs


def fetch(session, rel, dest, mirror):
    for base in MIRRORS[mirror:] + MIRRORS[:mirror]:
        try:
            r = session.get(base + rel, timeout=300)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return
        except requests.RequestException:
            continue
    raise RuntimeError(f"could not download {rel}")


def epochs_from_edf(psg_path, hyp_path):
    """-> x (n, 3000) float16 in microvolts, y (n,) int8."""
    import mne
    raw = mne.io.read_raw_edf(psg_path, include=["EEG Fpz-Cz"], preload=True, verbose=False)
    assert raw.info["sfreq"] == FS
    sig = raw.get_data()[0] * 1e6
    labels = np.full(len(sig) // (EPOCH_S * FS), -1, dtype=np.int8)
    for onset, dur, desc in zip(*[getattr(mne.read_annotations(hyp_path), k)
                                  for k in ("onset", "duration", "description")]):
        if desc in STAGE_OF:
            a = int(round(onset / EPOCH_S))
            labels[a:a + int(round(dur / EPOCH_S))] = STAGE_OF[desc]
    sleep = np.flatnonzero(labels > 0)
    lo, hi = max(sleep[0] - WAKE_EDGE, 0), min(sleep[-1] + WAKE_EDGE + 1, len(labels))
    idx = np.arange(lo, hi)
    idx = idx[labels[idx] >= 0]              # drop unscored / movement epochs
    x = sig[: len(labels) * EPOCH_S * FS].reshape(len(labels), -1)[idx]
    return x.astype(np.float16), labels[idx], idx.astype(np.int32)


def convert(job, cache, tmp, session, mirror):
    out = cache / f"{job['cohort']}{job['subject']:02d}n{job['night']}.npz"
    if out.exists():
        return out
    psg, hyp = tmp / Path(job["psg"]).name, tmp / Path(job["hyp"]).name
    fetch(session, job["psg"], psg, mirror)
    fetch(session, job["hyp"], hyp, mirror)
    x, y, idx = epochs_from_edf(psg, hyp)
    # `idx` is each epoch's position in the night, so gaps from dropped epochs stay visible
    np.savez(out, x=x, y=y, idx=idx, **{k: job[k] for k in ("cohort", "subject", "night", "age")})
    psg.unlink()
    hyp.unlink()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/sleep_cache")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cache, tmp = Path(args.cache), Path(args.cache) / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    jobs = plan(session)
    print(f"{len(jobs)} nights:", pd.DataFrame(jobs).groupby("cohort").size().to_dict())
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(convert, j, cache, tmp, requests.Session(), i % 2)
                for i, j in enumerate(jobs)]
        for f in tqdm(as_completed(futs), total=len(futs)):
            f.result()


if __name__ == "__main__":
    main()
