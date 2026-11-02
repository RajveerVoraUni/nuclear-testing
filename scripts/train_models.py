import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'scripts' in __file__ else os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'data') if os.path.exists(os.path.join(BASE_DIR, 'data')) else BASE_DIR

def get_data_path(fname):
    p = os.path.join(DATA_DIR, fname)
    return p if os.path.exists(p) else fname
"""
train_models.py
Trains and evaluates all 5 statistical and ML classifiers on 3-Component physics features:
1. Logistic Regression (Lasso L1)
2. Logistic Regression (Elastic Net)
3. Random Forest
4. Support Vector Machine (SVM with RBF Kernel)
5. XGBoost Classifier
Evaluates via 5-Fold GroupKFold strictly by seismic event ID (evid).
Saves results to get_data_path("classification_results.csv").
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score
)

def evaluate_models(models, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    results = []

    for name, clf in models.items():
        accs, b_accs, precs, recs, f1s, roc_aucs, pr_aucs = [], [], [], [], [], [], []

        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            clf.fit(X_tr_s, y_tr)

            pred = clf.predict(X_te_s)
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X_te_s)[:, 1]
            elif hasattr(clf, "decision_function"):
                prob = clf.decision_function(X_te_s)
            else:
                prob = pred

            accs.append(accuracy_score(y_te, pred))
            b_accs.append(balanced_accuracy_score(y_te, pred))
            precs.append(precision_score(y_te, pred, zero_division=0))
            recs.append(recall_score(y_te, pred, zero_division=0))
            f1s.append(f1_score(y_te, pred, zero_division=0))
            roc_aucs.append(roc_auc_score(y_te, prob))
            pr_aucs.append(average_precision_score(y_te, prob))

        results.append({
            "Model": name,
            "Accuracy": np.mean(accs),
            "Balanced_Acc": np.mean(b_accs),
            "Precision": np.mean(precs),
            "Recall": np.mean(recs),
            "F1-Score": np.mean(f1s),
            "ROC-AUC": np.mean(roc_aucs),
            "PR-AUC": np.mean(pr_aucs)
        })

    return pd.DataFrame(results)

def main():
    print("Loading 3-Component physical feature dataset...")
    df = pd.read_csv(get_data_path("features.csv"))

    feat_cols = [c for c in df.columns if c not in ["evid", "station", "event_type", "label", "station_dist"]]
    X = df[feat_cols]
    y = df["label"]
    groups = df["evid"]

    print(f"Dataset: {len(df)} samples across {groups.nunique()} unique events.")
    print(f"Features ({len(feat_cols)}): {feat_cols}\n")

    models = {
        "Logistic Regression (Lasso L1)": LogisticRegression(
            solver="saga", l1_ratio=1.0, C=0.3, max_iter=2500, random_state=42
        ),
        "Logistic Regression (Elastic Net)": LogisticRegression(
            solver="saga", l1_ratio=0.5, C=0.5, max_iter=2500, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_split=4, random_state=42
        ),
        "SVM (RBF Kernel)": SVC(
            kernel="rbf", C=5.0, gamma="scale", random_state=42
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42
        )
    }

    print("Running 5-Fold Event-Grouped Cross-Validation on 3C Data...")
    res_df = evaluate_models(models, X, y, groups)

    display_df = res_df.copy()
    display_df["Accuracy"] = display_df["Accuracy"].apply(lambda v: f"{v*100:.2f}%")
    display_df["Balanced_Acc"] = display_df["Balanced_Acc"].apply(lambda v: f"{v*100:.2f}%")
    display_df["Precision"] = display_df["Precision"].apply(lambda v: f"{v*100:.2f}%")
    display_df["Recall"] = display_df["Recall"].apply(lambda v: f"{v*100:.2f}%")
    display_df["F1-Score"] = display_df["F1-Score"].apply(lambda v: f"{v:.4f}")
    display_df["ROC-AUC"] = display_df["ROC-AUC"].apply(lambda v: f"{v:.4f}")
    display_df["PR-AUC"] = display_df["PR-AUC"].apply(lambda v: f"{v:.4f}")

    print("\n" + "="*85)
    print("3-COMPONENT CLASSIFICATION RESULTS (5-Fold Event-Grouped CV)")
    print("="*85)
    print(display_df.to_string(index=False))
    print("="*85)

    # Feature Importance from XGBoost
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    xgb_full = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    )
    xgb_full.fit(X_s, y)
    fi = pd.Series(xgb_full.feature_importances_, index=feat_cols).sort_values(ascending=False)

    print("\nTop 3-Component Physical Discriminators (XGBoost Importance):")
    for feat, imp in fi.items():
        print(f"  {feat:<28}: {imp*100:>6.2f}%")

    res_df.to_csv(get_data_path("classification_results.csv"), index=False)
    print("\nSaved 3-component classification results to classification_results.csv")

if __name__ == "__main__":
    main()

