"""Participant-owned model package for the Poker44 miner — variance-robust
portfolio entry `poker-stack-lxe` (stack_lgbxgbet).

Bot detector = STACKED ensemble over C2's 180 sanitization-invariant behavioral
features: LightGBM(depth-6, strong-reg) + XGBoost(depth-6, hist) + ExtraTrees(300)
base learners, 5-fold out-of-fold predict_proba stacked into a
LogisticRegression(C=0.8) meta-learner (sklearn StackingClassifier, cv=5). This
is the stack architecture the risen-under-new-eval top miners use; the OOF
meta-blend floors higher under noisy few-validator (chunk-draw-lottery) scoring
than any single learner.

Reuses C2's feature extraction (features.py), inference path (detector.py:
within-batch rank output, NO re-sanitization — live hands are already sanitized
validator-side), and capture. All base learners are single-thread
(n_jobs/thread_count=1) so the threaded axon does not deadlock on batched
predict. Trained on the full 724-group sanitized benchmark. See
model.joblib (StackingClassifier), detector.py, features.py.
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
