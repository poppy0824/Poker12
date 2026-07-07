"""Build stack_lgbxgbet (poker-stack-lxe): LGB+XGB+ET base -> LogReg(C=0.8) meta,
5-fold OOF (sklearn StackingClassifier cv=5). All learners single-thread."""
import numpy as np, joblib, sys, os
import sweep_lib as S
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def make_stack(seed=0):
    lgb = LGBMClassifier(
        n_estimators=400, max_depth=6, num_leaves=31, learning_rate=0.03,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0, min_child_samples=20,
        random_state=seed, n_jobs=1, verbosity=-1)
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.03, tree_method="hist",
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
        min_child_weight=5, random_state=seed, n_jobs=1,
        eval_metric="logloss", verbosity=0)
    et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4,
                              random_state=seed, n_jobs=1)
    meta = LogisticRegression(C=0.8, max_iter=2000)
    return StackingClassifier(
        estimators=[("lgb", lgb), ("xgb", xgb), ("et", et)],
        final_estimator=meta, cv=5, stack_method="predict_proba", n_jobs=1,
        passthrough=False)


if __name__ == "__main__":
    Xb, y, dates = S.c2_bench()
    print(f"[data] bench {Xb.shape} pos={int(y.sum())}", flush=True)

    if "--gateA" in sys.argv:
        rA = S.bench_cv_reward(Xb, y, dates, factory=make_stack, folds=5)
        print(f"[GATE A] stack_lgbxgbet bench_cv_reward = {rA:.4f}", flush=True)
        rC2 = S.bench_cv_reward(*S.c2_bench(), factory=S.ens_c2, folds=5)
        print(f"[GATE A] C2 ref bench_cv_reward         = {rC2:.4f}", flush=True)

    if "--fit" in sys.argv:
        m = make_stack(0).fit(Xb, y)
        out = "/root/ares/Poker/train/dev/varrobust/stack_lgbxgbet/poker44_model/model.joblib"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        joblib.dump(m, out)
        print(f"[fit] wrote {out}", flush=True)
