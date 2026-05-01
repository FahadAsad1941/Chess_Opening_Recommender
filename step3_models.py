"""
STEP 3: Train & Compare ML Models
===================================
Trains Decision Tree, Random Forest, SVM, and Naive Bayes.
Compares them with accuracy, confusion matrices, and plots.
Covers Labs 3, 4, 5, and 7 all in one project.

HOW TO RUN:
    python step3_models.py

INPUT:  train_test_data.pkl
OUTPUT: model_comparison.png
        confusion_matrices.png
        feature_importance.png
        best_model.pkl  (saved for the recommender)
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, f1_score)

print("=" * 55)
print("  STEP 3: Training & Comparing Models")
print("=" * 55)

# ─────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────
with open("train_test_data.pkl", "rb") as f:
    data = pickle.load(f)

X_train       = data["X_train"]
X_test        = data["X_test"]
X_train_scaled = data["X_train_scaled"]
X_test_scaled  = data["X_test_scaled"]
y_train       = data["y_train"]
y_test        = data["y_test"]
FEATURES      = data["features"]
scaler        = data["scaler"]

classes = sorted(y_train.unique())
print(f"\n Classes: {classes}")
print(f" Training samples: {len(y_train)}")
print(f" Test samples: {len(y_test)}")

# ─────────────────────────────────────────
#  DEFINE MODELS
#  (unscaled for tree models, scaled for SVM/NB)
# ─────────────────────────────────────────
models = {
    # Lab 3 — Naive Bayes
    "Naive Bayes": {
        "model": GaussianNB(),
        "scaled": True,
        "color": "#D4537E",
        "lab": "Lab 3"
    },
    # Lab 4 — Decision Tree
    "Decision Tree": {
        "model": DecisionTreeClassifier(
            criterion="entropy", max_depth=6, random_state=42
        ),
        "scaled": False,
        "color": "#378ADD",
        "lab": "Lab 4"
    },
    # Lab 5 — Random Forest (Ensemble)
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        ),
        "scaled": False,
        "color": "#1D9E75",
        "lab": "Lab 5"
    },
    # Lab 5 — AdaBoost (Ensemble)
    "AdaBoost": {
        "model": AdaBoostClassifier(n_estimators=100, random_state=42),
        "scaled": False,
        "color": "#EF9F27",
        "lab": "Lab 5"
    },
    # Lab 7 — SVM Linear
    "SVM Linear": {
        "model": SVC(kernel="linear", probability=True, random_state=42),
        "scaled": True,
        "color": "#7F77DD",
        "lab": "Lab 7"
    },
    # Lab 7 — SVM RBF
    "SVM RBF": {
        "model": SVC(kernel="rbf", probability=True, random_state=42),
        "scaled": True,
        "color": "#534AB7",
        "lab": "Lab 7"
    },
}

# ─────────────────────────────────────────
#  TRAIN ALL MODELS
# ─────────────────────────────────────────
results = {}

print("\n Training models...\n")
for name, cfg in models.items():
    model = cfg["model"]
    Xtr = X_train_scaled if cfg["scaled"] else X_train
    Xte = X_test_scaled  if cfg["scaled"] else X_test

    print(f"  [{cfg['lab']}] Training {name}...", end=" ")
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cm  = confusion_matrix(y_test, y_pred, labels=classes)

    results[name] = {
        "model":  model,
        "y_pred": y_pred,
        "acc":    acc,
        "f1":     f1,
        "cm":     cm,
        "color":  cfg["color"],
        "lab":    cfg["lab"],
    }
    print(f"Accuracy: {acc:.3f}  |  F1: {f1:.3f}")

# ─────────────────────────────────────────
#  PLOT 1: MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────
print("\n Generating model comparison plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Comparison — Chess Opening Recommender", 
             fontsize=13, fontweight="bold")

names  = list(results.keys())
accs   = [results[n]["acc"]   for n in names]
f1s    = [results[n]["f1"]    for n in names]
colors = [results[n]["color"] for n in names]
labs   = [results[n]["lab"]   for n in names]

# Accuracy bars
bars = ax1.barh(names, accs, color=colors, height=0.6)
ax1.set_xlim(0, 1.1)
ax1.set_xlabel("Accuracy")
ax1.set_title("Accuracy by Model")
ax1.axvline(max(accs), color="gray", linestyle="--", linewidth=1, alpha=0.5)
for bar, acc, lab in zip(bars, accs, labs):
    ax1.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
             f"{acc:.3f}  ({lab})", va="center", fontsize=9)

# F1 bars
bars2 = ax2.barh(names, f1s, color=colors, height=0.6)
ax2.set_xlim(0, 1.1)
ax2.set_xlabel("Weighted F1 Score")
ax2.set_title("F1 Score by Model")
for bar, f1 in zip(bars2, f1s):
    ax2.text(f1 + 0.01, bar.get_y() + bar.get_height()/2,
             f"{f1:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("   Saved: model_comparison.png")

# ─────────────────────────────────────────
#  PLOT 2: CONFUSION MATRICES (all 6)
# ─────────────────────────────────────────
print(" Generating confusion matrices...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")

for ax, (name, res) in zip(axes.flat, results.items()):
    cm = res["cm"]
    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=cm, fmt="d", ax=ax,
                cmap="Blues", xticklabels=classes, yticklabels=classes,
                linewidths=0.5, cbar=False)
    ax.set_title(f"{name}\nAcc: {res['acc']:.3f}", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("   Saved: confusion_matrices.png")

# ─────────────────────────────────────────
#  PLOT 3: FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────
print(" Generating feature importance plot...")

rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(FEATURES)), importances[indices],
               color=[plt.cm.viridis(i / len(FEATURES)) for i in range(len(FEATURES))])
plt.xticks(range(len(FEATURES)),
           [FEATURES[i] for i in indices], rotation=40, ha="right", fontsize=9)
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title("Feature Importance — Random Forest\n(Which player stats best predict opening choice?)")
for bar, imp in zip(bars, importances[indices]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{imp:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("   Saved: feature_importance.png")

# ─────────────────────────────────────────
#  PRINT CLASSIFICATION REPORT (Best Model)
# ─────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["acc"])
best_res  = results[best_name]

print(f"\n {'='*50}")
print(f"  Best Model: {best_name}  (Accuracy: {best_res['acc']:.3f})")
print(f" {'='*50}")
print(classification_report(y_test, best_res["y_pred"], target_names=classes))

# ─────────────────────────────────────────
#  SAVE BEST MODEL
# ─────────────────────────────────────────
with open("best_model.pkl", "wb") as f:
    pickle.dump({
        "model_name": best_name,
        "model": best_res["model"],
        "scaler": scaler,
        "features": FEATURES,
        "classes": classes,
        "all_results": {n: {"acc": r["acc"], "f1": r["f1"]} for n, r in results.items()},
    }, f)

print(f"\n Saved best model ({best_name}) to best_model.pkl")
print("\n STEP 3 COMPLETE — Run step4_recommender.py next!")
