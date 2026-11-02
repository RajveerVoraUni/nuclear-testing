import os
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'scripts' in __file__ else os.path.abspath('.')
FIG_DIR = os.path.join(BASE_DIR, 'figures') if os.path.exists(os.path.join(BASE_DIR, 'figures')) else BASE_DIR

"""
plot_results.py
Generates 30% larger publication-quality benchmark and feature importance figures:
1. model_benchmark_bars.png: 5-Fold Group-CV metrics across all 6 models with 20% wider bars,
   scaled up 30% larger in dimensions and typography, legend outside, best overall CNN outside.
2. feature_importance.png: Top 10 XGBoost physical discriminators with 20% longer/wider bars,
   scaled up 30% larger in dimensions and typography.
3. classification_results.png: Stacked composite visualization (one below the other, 30% larger).
"""

# Global font and style configuration (scaled 30% larger)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13.0,
    "axes.labelsize": 13.5,
    "axes.titlesize": 14.5,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 11.2,
})

models = [
    "Lasso (L1)",
    "Elastic Net",
    "Random Forest",
    "SVM (RBF)",
    "XGBoost",
    "3-Channel CNN"
]

accuracy  = [94.82, 95.98, 95.97, 96.84, 96.26, 97.98]
precision = [98.36, 98.40, 97.39, 97.45, 97.89, 99.01]
recall    = [92.58, 94.53, 95.53, 97.05, 95.53, 97.54]
pr_auc    = [99.33, 99.50, 99.33, 99.61, 99.51, 99.76]

features = [
    "spec_spread_h\n(Horizontal Spectral Spread)",
    "cross_phase_spectral_ratio\n(Z vs H Spectral Ratio)",
    "spec_spread_z\n(Vertical Spectral Spread)",
    "spec_centroid_z\n(Vertical Spectral Centroid)",
    "spec_ratio_h\n(Horizontal High/Low Freq Ratio)",
    "spec_centroid_h\n(Horizontal Spectral Centroid)",
    "spec_ratio_z\n(Vertical High/Low Freq Ratio)",
    "ps_ratio_cross\n(Cross-Component P/S Ratio)",
    "h_energy_frac\n(Transverse Energy Fraction)",
    "rectilinearity\n(3D Covariance Rectilinearity)"
]
importances = [48.06, 13.12, 9.18, 4.45, 4.01, 3.01, 2.72, 1.76, 1.76, 1.50]


def generate_benchmark_bars():
    # 30% larger figure size
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(models)) * 1.20
    width = 0.23  # 20% wider bars

    ax.bar(x - 1.5 * width, accuracy,  width, label="Accuracy",  color="#3470a3", alpha=0.92, edgecolor="#1c3d5a", lw=0.7)
    ax.bar(x - 0.5 * width, precision, width, label="Precision", color="#e68438", alpha=0.92, edgecolor="#9c4d10", lw=0.7)
    ax.bar(x + 0.5 * width, recall,    width, label="Recall",    color="#d65252", alpha=0.92, edgecolor="#8c2525", lw=0.7)
    ax.bar(x + 1.5 * width, pr_auc,    width, label="PR-AUC",    color="#48a05c", alpha=0.92, edgecolor="#256633", lw=0.7)

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("3-Component Seismic Classification (5-Fold Event-Grouped CV)", fontweight="bold", pad=48)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(88, 101.5)
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    # Legend OUTSIDE the graph (top-left)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, frameon=True, framealpha=0.95, edgecolor="#cbd5e1", columnspacing=0.8, handletextpad=0.35)

    # Best Overall OUTSIDE the graph (top-right)
    ax.text(1.0, 1.02, "Best Overall: CNN (97.98% Acc, 99.01% Prec)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11.0, fontweight="bold",
            color="#1c3d5a", bbox=dict(boxstyle="round,pad=0.38", fc="#f0f6ff", ec="#3470a3", lw=1.2))

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "model_benchmark_bars.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_feature_importance():
    # 30% larger figure size
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    y_pos = np.arange(len(features))

    bars = ax.barh(y_pos, importances, align="center", height=0.72, color="#3b6978", alpha=0.90, edgecolor="#1b3742", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Feature Importance Gain (%)", fontweight="bold")
    ax.set_title("Top 3-Component Physical Discriminators (XGBoost)", fontweight="bold", pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    ax.set_xlim(0, 56)

    for bar, v in zip(bars, importances):
        ax.text(v + 0.7, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                va="center", fontweight="bold", fontsize=11.5, color="#1b3742")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "feature_importance.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_composite_results():
    # 30% larger composite figure size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.8, 12.2))

    # --- Subplot 1: Model Benchmarks ---
    x = np.arange(len(models)) * 1.20
    width = 0.23

    ax1.bar(x - 1.5 * width, accuracy,  width, label="Accuracy",  color="#3470a3", alpha=0.92, edgecolor="#1c3d5a", lw=0.7)
    ax1.bar(x - 0.5 * width, precision, width, label="Precision", color="#e68438", alpha=0.92, edgecolor="#9c4d10", lw=0.7)
    ax1.bar(x + 0.5 * width, recall,    width, label="Recall",    color="#d65252", alpha=0.92, edgecolor="#8c2525", lw=0.7)
    ax1.bar(x + 1.5 * width, pr_auc,    width, label="PR-AUC",    color="#48a05c", alpha=0.92, edgecolor="#256633", lw=0.7)

    ax1.set_ylabel("Score (%)", fontweight="bold")
    ax1.set_title("3-Component Seismic Classification (5-Fold Event-Grouped CV)", fontweight="bold", pad=48)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(88, 101.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)

    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, frameon=True, framealpha=0.95, edgecolor="#cbd5e1", columnspacing=0.8, handletextpad=0.35)
    ax1.text(1.0, 1.02, "Best Overall: CNN (97.98% Acc, 99.01% Prec)",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=11.0, fontweight="bold",
             color="#1c3d5a", bbox=dict(boxstyle="round,pad=0.38", fc="#f0f6ff", ec="#3470a3", lw=1.2))

    # --- Subplot 2: Feature Importance ---
    y_pos = np.arange(len(features))
    bars = ax2.barh(y_pos, importances, align="center", height=0.72, color="#3b6978", alpha=0.90, edgecolor="#1b3742", lw=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features, fontsize=11.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Relative Feature Importance Gain (%)", fontweight="bold")
    ax2.set_title("Top 3-Component Physical Discriminators (XGBoost)", fontweight="bold", pad=14)
    ax2.grid(axis="x", linestyle="--", alpha=0.45)
    ax2.set_xlim(0, 56)

    for bar, v in zip(bars, importances):
        ax2.text(v + 0.7, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                 va="center", fontweight="bold", fontsize=11.5, color="#1b3742")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "classification_results.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_benchmark_bars()
    generate_feature_importance()
    generate_composite_results()
    print("All 30% larger figures successfully regenerated!")
