import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'scripts' in __file__ else os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'data') if os.path.exists(os.path.join(BASE_DIR, 'data')) else BASE_DIR

def get_data_path(fname):
    p = os.path.join(DATA_DIR, fname)
    return p if os.path.exists(p) else fname
"""
tune_hyperparameters.py
Executes systematic hyperparameter optimization for all 5 tabular models
using 5-Fold GroupKFold strictly by seismic event ID (evid) to prevent data leakage.
Optimizes for PR-AUC (Average Precision) and Balanced Accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

def evaluate_param_set(clf_class, param_candidates, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    best_score = -1.0
    best_params = None
    best_metrics = None

    for params in param_candidates:
        accs, b_accs, precs, recs, roc_aucs, pr_aucs = [], [], [], [], [], []
        for tr_idx, te_idx in gkf.split(X, y, groups=groups):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            clf = clf_class(**params)
            clf.fit(X_tr_s, y_tr)

            pred = clf.predict(X_te_s)
            prob = clf.predict_proba(X_te_s)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_te_s)

            accs.append(accuracy_score(y_te, pred))
            b_accs.append(balanced_accuracy_score(y_te, pred))
            precs.append(precision_score(y_te, pred, zero_division=0))
            recs.append(recall_score(y_te, pred, zero_division=0))
            roc_aucs.append(roc_auc_score(y_te, prob))
            pr_aucs.append(average_precision_score(y_te, prob))

        mean_pr = np.mean(pr_aucs)
        mean_acc = np.mean(accs)
        # Composite score prioritizing PR-AUC and Accuracy
        composite = 0.5 * mean_pr + 0.5 * mean_acc

        if composite > best_score:
            best_score = composite
            best_params = params
            best_metrics = {
                "Accuracy": mean_acc,
                "Balanced_Acc": np.mean(b_accs),
                "Precision": np.mean(precs),
                "Recall": np.mean(recs),
                "ROC-AUC": np.mean(roc_aucs),
                "PR-AUC": mean_pr
            }

    return best_params, best_metrics

def main():
    print("Loading 3C features dataset...")
    df = pd.read_csv(get_data_path("features.csv"))
    feat_cols = [c for c in df.columns if c not in ["evid", "station", "event_type", "label", "station_dist"]]
    X = df[feat_cols]
    y = df["label"]
    groups = df["evid"]

    print(f"Dataset: {len(df)} samples across {groups.nunique()} events with {len(feat_cols)} pure wave features.\n")

    # 1. Lasso L1 Tuning
    print("--- 1. Tuning Lasso (L1 Regularization) ---")
    lasso_grid = [
        {"solver": "saga", "l1_ratio": 1.0, "C": c, "max_iter": 3000, "random_state": 42}
        for c in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
    ]
    best_lasso_p, best_lasso_m = evaluate_param_set(LogisticRegression, lasso_grid, X, y, groups)
    print(f"Optimal Lasso Params: C={best_lasso_p['C']}")
    print(f"Metrics: Acc={best_lasso_m['Accuracy']*100:.2f}%, PR-AUC={best_lasso_m['PR-AUC']:.4f}\n")

    # 2. Elastic Net Tuning
    print("--- 2. Tuning Elastic Net ---")
    enet_grid = [
        {"solver": "saga", "l1_ratio": l1, "C": c, "max_iter": 3000, "random_state": 42}
        for c in [0.1, 0.3, 0.5, 1.0, 2.0]
        for l1 in [0.2, 0.4, 0.5, 0.6, 0.8]
    ]
    best_enet_p, best_enet_m = evaluate_param_set(LogisticRegression, enet_grid, X, y, groups)
    print(f"Optimal Elastic Net Params: C={best_enet_p['C']}, l1_ratio={best_enet_p['l1_ratio']}")
    print(f"Metrics: Acc={best_enet_m['Accuracy']*100:.2f}%, PR-AUC={best_enet_m['PR-AUC']:.4f}\n")

    # 3. Random Forest Tuning
    print("--- 3. Tuning Random Forest ---")
    rf_grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_split": s, "random_state": 42}
        for n in [100, 150, 200]
        for d in [4, 6, 8, None]
        for s in [2, 4]
    ]
    best_rf_p, best_rf_m = evaluate_param_set(RandomForestClassifier, rf_grid, X, y, groups)
    print(f"Optimal RF Params: n_est={best_rf_p['n_estimators']}, max_depth={best_rf_p['max_depth']}, min_split={best_rf_p['min_samples_split']}")
    print(f"Metrics: Acc={best_rf_m['Accuracy']*100:.2f}%, PR-AUC={best_rf_m['PR-AUC']:.4f}\n")

    # 4. SVM (RBF Kernel) Tuning
    print("--- 4. Tuning SVM (RBF Kernel) ---")
    svm_grid = [
        {"kernel": "rbf", "C": c, "gamma": g, "random_state": 42}
        for c in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        for g in ["scale", "auto", 0.01, 0.05, 0.1]
    ]
    best_svm_p, best_svm_m = evaluate_param_set(SVC, svm_grid, X, y, groups)
    print(f"Optimal SVM Params: C={best_svm_p['C']}, gamma={best_svm_p['gamma']}")
    print(f"Metrics: Acc={best_svm_m['Accuracy']*100:.2f}%, PR-AUC={best_svm_m['PR-AUC']:.4f}\n")

    # 5. XGBoost Tuning
    print("--- 5. Tuning XGBoost ---")
    xgb_grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42}
        for n in [100, 150, 200]
        for d in [3, 4, 5]
        for lr in [0.02, 0.05, 0.1]
    ]
    best_xgb_p, best_xgb_m = evaluate_param_set(XGBClassifier, xgb_grid, X, y, groups)
    print(f"Optimal XGBoost Params: n_est={best_xgb_p['n_estimators']}, max_depth={best_xgb_p['max_depth']}, lr={best_xgb_p['learning_rate']}")
    print(f"Metrics: Acc={best_xgb_m['Accuracy']*100:.2f}%, PR-AUC={best_xgb_m['PR-AUC']:.4f}\n")

if __name__ == "__main__":
    main()

