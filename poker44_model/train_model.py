"""Reproducible training for poker-puretree-wide -> writes model.joblib.

A PURE-TREE diversity bag over C2's 180 sanitization-invariant features
(features.py FEATURE_NAMES), soft-voted:

    ExtraTrees(300, msl4) + HistGradientBoosting + CatBoost + tuned LightGBM

NO linear head (the live new-eval signal penalises linear heads ~0.15). All
learners are single-threaded to avoid the axon batched-predict deadlock
(ExtraTrees n_jobs=-1 hangs): n_jobs=1 / thread_count=1 / num_threads=1.

Every raw benchmark hand is passed through the validator's
`prepare_hand_for_miner` BEFORE feature extraction so the training distribution
matches what the validator serves (train==serve). Live chunks are already
sanitized validator-side, so inference does NOT re-sanitize.

Two ways to run:

    # (a) self-contained from raw benchmark + payload_view (like C2's trainer)
    python3 poker44_model/train_model.py --data /root/ares/Poker/train/raw \
        --payload-view /root/ares/Poker/main/poker44/validator/payload_view.py

    # (b) from the shared sweep_lib cached sanitized 724-group matrix (fast)
    python3 poker44_model/train_model.py --from-sweeplib \
        --sweeplib-dir /tmp/.../scratchpad
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import typing

import numpy as np
import joblib
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              VotingClassifier)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from poker44_model.features import chunk_features, FEATURE_NAMES


def build_ensemble(seed=0):
    """Pure-tree soft-vote bag. All learners single-threaded (axon-safe)."""
    et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4,
                              random_state=seed, n_jobs=1)
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03,
                                         max_iter=300, l2_regularization=1.0,
                                         random_state=seed)
    cb = CatBoostClassifier(depth=6, iterations=400, learning_rate=0.05,
                            thread_count=1, random_seed=seed, verbose=False,
                            allow_writing_files=False)
    # tuned LightGBM: shallow, regularised, distinct from HGB/CatBoost geometry
    lgb = LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31,
                         max_depth=5, min_child_samples=20, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.7,
                         reg_lambda=1.0, reg_alpha=0.0, random_state=seed,
                         n_jobs=1, num_threads=1, verbosity=-1)
    return VotingClassifier(
        estimators=[("et", et), ("hgb", hgb), ("cb", cb), ("lgb", lgb)],
        voting="soft", n_jobs=1)


def _load_sanitizer(pv_path):
    spec = importlib.util.spec_from_file_location("_p44_payload_view", pv_path)
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    pv.Optional = typing.Optional  # payload_view uses Optional but never imports it
    fn = pv.prepare_hand_for_miner

    def sanitize_chunk(chunk):
        out = []
        for h in (chunk or []):
            try:
                out.append(fn(h))
            except Exception:
                out.append(h)
        return out

    return sanitize_chunk


def load(raw):
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "chunks_*.json"))):
        for rc in json.load(open(f)).get("chunks", []):
            for g, l in zip(rc.get("chunks") or [], rc.get("groundTruth") or []):
                out.append((g, int(l)))
    return out


def _build_from_raw(args):
    sanitize_chunk = _load_sanitizer(args.payload_view)
    data = load(args.data)
    rows, y = [], []
    for g, l in data:
        feats = chunk_features(sanitize_chunk(g))   # TRAIN == SERVE
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
        y.append(l)
    return np.array(rows, dtype=float), np.array(y)


def _build_from_sweeplib(args):
    sys.path.insert(0, args.sweeplib_dir)
    import sweep_lib as S  # noqa: E402
    Xb, y, _ = S.c2_bench()
    return np.asarray(Xb, dtype=float), np.asarray(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to train/raw chunk JSON dir")
    ap.add_argument("--payload-view", help="path to payload_view.py (the sanitizer)")
    ap.add_argument("--from-sweeplib", action="store_true",
                    help="use sweep_lib's cached sanitized 724-group matrix")
    ap.add_argument("--sweeplib-dir", help="dir containing sweep_lib.py")
    args = ap.parse_args()

    if args.from_sweeplib:
        X, y = _build_from_sweeplib(args)
    else:
        if not (args.data and args.payload_view):
            ap.error("need --data and --payload-view, or --from-sweeplib")
        X, y = _build_from_raw(args)

    model = build_ensemble(seed=0).fit(X, y)

    out = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, out)
    print(f"wrote {out} ({X.shape[0]} examples, {X.shape[1]} features)")


if __name__ == "__main__":
    main()
