"""Participant-owned model package for the Poker44 miner — poker-puretree-wide.

Bot detector = a PURE-TREE diversity bag (NO linear head): ExtraTrees +
HistGradientBoosting + CatBoost + a tuned LightGBM, soft-vote, over C2's 180
sanitization-invariant features (cross-hand duplication sig_* + entropies +
structural / aggression aggregates). Trained on the full 724-group benchmark
passed through the validator's prepare_hand_for_miner (train==serve); scored by
within-batch ranking. Inference does NOT re-sanitize (live hands arrive already
sanitized validator-side).

Distinct from C2 (2-bag ET+HGB), the uid174 4-bag (ET+ET+RF+HGB), and the
varrobust ens_catboost bag (ET+HGB+CatBoost+RF): this bag drops RF and adds a
tuned LightGBM alongside CatBoost, giving a different boosting/tree mixture.
The live new-eval signal favours pure trees (a linear head cost ~0.15), so this
build carries none.

All learners are single-threaded (n_jobs=1 / thread_count=1 / num_threads=1) —
ExtraTrees n_jobs=-1 deadlocks the axon's batched predict. See detector.py
(inference), features.py (extraction + FEATURE_NAMES), train_model.py (training),
model.joblib (trained model).
"""

from poker44_model.detector import score_batch, score_chunk

MODEL_NAME = "poker-puretree-wide"

__all__ = ["score_batch", "score_chunk", "MODEL_NAME"]
