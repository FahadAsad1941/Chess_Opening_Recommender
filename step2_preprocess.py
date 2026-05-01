"""
STEP 2: Preprocessing & Exploratory Data Analysis
===================================================
Cleans the dataset, encodes features, scales them,
and visualizes the data — exactly like your Labs 2, 3, 6.

HOW TO RUN:
    python step2_preprocess.py

INPUT:  chess_dataset.csv
OUTPUT: chess_processed.csv  (ready for ML models)
        Several plots saved as PNG files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("  STEP 2: Preprocessing & EDA")
print("=" * 55)

# ─────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("chess_dataset.csv")
print(f"\n Raw dataset shape: {df.shape}")
print(f"\n Columns:\n{df.dtypes}")
print(f"\n Missing values:\n{df.isnull().sum()}")
print(f"\n Sample rows:")
print(df.head(3))

# ─────────────────────────────────────────
#  HANDLE MISSING VALUES
# ─────────────────────────────────────────
# Fill numeric nulls with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"   Filled {col} nulls with median")

# Fill categorical nulls with mode
cat_cols = ["time_class", "played_as", "outcome"]
for col in cat_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"   Filled {col} nulls with mode")

print(f"\n After cleaning: {df.shape}")

# ─────────────────────────────────────────
#  EDA — VISUALIZATIONS
# ─────────────────────────────────────────
print("\n Generating EDA plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Chess Opening Recommender — Exploratory Data Analysis", 
             fontsize=14, fontweight='bold')

# Plot 1: Opening family distribution (our target variable)
ax = axes[0, 0]
counts = df["opening_family"].value_counts()
bars = ax.bar(counts.index, counts.values, 
              color=["#7F77DD", "#1D9E75", "#D85A30", "#D4537E", "#378ADD"])
ax.set_title("Opening Family Distribution (Target)")
ax.set_xlabel("Opening Family")
ax.set_ylabel("Number of Games")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha='center', fontsize=9)

# Plot 2: ELO distribution
ax = axes[0, 1]
ax.hist(df["player_elo"].dropna(), bins=30, color="#7F77DD", alpha=0.7, label="Player")
ax.hist(df["opponent_elo"].dropna(), bins=30, color="#1D9E75", alpha=0.7, label="Opponent")
ax.set_title("ELO Rating Distribution")
ax.set_xlabel("ELO Rating")
ax.set_ylabel("Count")
ax.legend()

# Plot 3: Time class breakdown
ax = axes[0, 2]
tc_counts = df["time_class"].value_counts()
ax.pie(tc_counts.values, labels=tc_counts.index, autopct='%1.1f%%',
       colors=["#7F77DD", "#1D9E75", "#D85A30", "#378ADD"])
ax.set_title("Time Control Breakdown")

# Plot 4: Win rate by opening family
ax = axes[1, 0]
win_rates = df.groupby("opening_family").apply(
    lambda x: (x["outcome"] == "win").mean() * 100
).sort_values(ascending=False)
bars = ax.bar(win_rates.index, win_rates.values,
              color=["#1D9E75" if v > 50 else "#D85A30" for v in win_rates.values])
ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% line")
ax.set_title("Win Rate by Opening Family")
ax.set_xlabel("Opening Family")
ax.set_ylabel("Win Rate (%)")
ax.legend()
for bar, val in zip(bars, win_rates.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha='center', fontsize=9)

# Plot 5: Game length by opening family
ax = axes[1, 1]
df.boxplot(column="num_moves", by="opening_family", ax=ax,
           boxprops=dict(color="#7F77DD"),
           medianprops=dict(color="#D85A30", linewidth=2))
ax.set_title("Game Length by Opening Family")
ax.set_xlabel("Opening Family")
ax.set_ylabel("Number of Moves")
plt.sca(ax)
plt.xticks(rotation=15)

# Plot 6: Outcome distribution by color played
ax = axes[1, 2]
outcome_color = df.groupby(["played_as", "outcome"]).size().unstack(fill_value=0)
outcome_color.plot(kind="bar", ax=ax, 
                   color=["#1D9E75", "#D85A30", "#378ADD"])
ax.set_title("Outcome by Color Played")
ax.set_xlabel("Played As")
ax.set_ylabel("Number of Games")
ax.legend(title="Outcome")
plt.sca(ax)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("   Saved: eda_plots.png")

# ─────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────
print("\n Engineering features...")

# Encode: played_as (white=1, black=0)
df["played_as_enc"] = (df["played_as"] == "white").astype(int)

# Encode: time_class (Label Encoding — Lab 2 style)
le_tc = LabelEncoder()
df["time_class_enc"] = le_tc.fit_transform(df["time_class"].fillna("unknown"))
print(f"   Time class mapping: {dict(zip(le_tc.classes_, le_tc.transform(le_tc.classes_)))}")

# Encode: outcome (win=2, draw=1, loss=0)
outcome_map = {"win": 2, "draw": 1, "loss": 0}
df["outcome_enc"] = df["outcome"].map(outcome_map).fillna(1)

# Per-player aggregate features
print("   Computing per-player style features...")
player_stats = df.groupby("username").agg(
    avg_elo=("player_elo", "mean"),
    avg_game_length=("num_moves", "mean"),
    win_rate=("outcome", lambda x: (x == "win").mean()),
    draw_rate=("outcome", lambda x: (x == "draw").mean()),
    decisive_rate=("decisive", "mean"),
    resign_rate=("resigned", "mean"),
    aggression_score=("num_moves", lambda x: 1 / (x.mean() + 1))  # shorter games = more aggressive
).reset_index()

# Merge back
df = df.merge(player_stats, on="username", suffixes=("", "_agg"))
print(f"   Added {len(player_stats.columns)-1} per-player aggregate features")

# Correlation heatmap
print("\n Generating correlation heatmap...")
feature_cols = ["player_elo", "opponent_elo", "elo_diff", "num_moves",
                "decisive", "resigned", "played_as_enc", "time_class_enc",
                "win_rate", "draw_rate", "decisive_rate", "resign_rate"]
feature_cols = [c for c in feature_cols if c in df.columns]

plt.figure(figsize=(10, 8))
corr = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("   Saved: correlation_heatmap.png")

# ─────────────────────────────────────────
#  PREPARE FINAL FEATURES FOR ML
# ─────────────────────────────────────────
print("\n Preparing final feature matrix...")

FEATURES = [
    "player_elo",
    "opponent_elo",
    "elo_diff",
    "played_as_enc",
    "time_class_enc",
    "num_moves",
    "decisive",
    "resigned",
    "win_rate",
    "draw_rate",
    "decisive_rate",
    "resign_rate",
    "aggression_score",
]

TARGET = "opening_family"

# Keep only available features
FEATURES = [f for f in FEATURES if f in df.columns]
df_ml = df[FEATURES + [TARGET]].dropna()

print(f"   Features used: {FEATURES}")
print(f"   Dataset size: {len(df_ml)} rows")
print(f"\n Target class distribution:")
print(df_ml[TARGET].value_counts())

# ─────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────
X = df_ml[FEATURES]
y = df_ml[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n Train size: {X_train.shape}, Test size: {X_test.shape}")

# ─────────────────────────────────────────
#  FEATURE SCALING (StandardScaler — Lab 7)
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────────
#  SAVE PROCESSED DATA
# ─────────────────────────────────────────
# Save processed ml-ready dataframe
df_ml.to_csv("chess_processed.csv", index=False)

# Save train/test splits as numpy arrays for next step
import pickle
with open("train_test_data.pkl", "wb") as f:
    pickle.dump({
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "features": FEATURES,
        "scaler": scaler,
    }, f)

print("\n Saved: chess_processed.csv")
print(" Saved: train_test_data.pkl")
print("\n STEP 2 COMPLETE — Run step3_models.py next!")
