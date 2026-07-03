# phase6_ml.py
# Machine learning anomaly detection using:
#   Model A — Isolation Forest (scikit-learn)
#   Model B — Autoencoder (Keras / TensorFlow)
# Both trained on engineered features, evaluated against ground truth.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving files
import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TensorFlow info messages

from sklearn.ensemble       import IsolationForest
from sklearn.preprocessing  import StandardScaler, RobustScaler
from sklearn.metrics        import (precision_score, recall_score, f1_score,
                                    roc_auc_score, confusion_matrix,
                                    RocCurveDisplay)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

DATA_DIR   = "data/raw"
OUTPUT_DIR = "outputs/ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD AND MERGE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
inv = pd.read_csv(f"{DATA_DIR}/invoices.csv",
                  parse_dates=["invoice_date","due_date","payment_date"])
po  = pd.read_csv(f"{DATA_DIR}/purchase_orders.csv", parse_dates=["po_date"])
ven = pd.read_csv(f"{DATA_DIR}/vendors.csv",
                  parse_dates=["vendor_creation_date"])
bud = pd.read_csv(f"{DATA_DIR}/budget.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
# Each feature captures a specific fraud signal.
# Explanation of what each one measures is in comments.
print("Engineering features...")

# Join vendor and PO data onto invoices
df = inv.merge(
    ven[["vendor_id","vendor_creation_date","vendor_category","is_ghost_vendor"]],
    on="vendor_id", how="left"
)
df = df.merge(
    po[["po_number","total_po_value"]], on="po_number", how="left"
)
df = df.merge(
    bud[["cost_centre","approved_budget","actual_spend"]].rename(
        columns={"approved_budget":"cc_budget","actual_spend":"cc_actual"}
    ).groupby("cost_centre").sum().reset_index(),
    on="cost_centre", how="left"
)

# Feature 1: Amount deviation from vendor's historical average (z-score)
# High z-score = this invoice is unusually large/small for this vendor
vendor_avg = df.groupby("vendor_id")["invoice_amount"].transform("mean")
vendor_std = df.groupby("vendor_id")["invoice_amount"].transform("std").replace(0, 1)
df["feat_amount_zscore"] = (df["invoice_amount"] - vendor_avg) / vendor_std

# Feature 2: Invoice amount as % of PO value
# Value far above 1.0 = invoice exceeds PO (match failure signal)
df["feat_invoice_to_po_ratio"] = np.where(
    df["total_po_value"].notna() & (df["total_po_value"] > 0),
    df["invoice_amount"] / df["total_po_value"],
    1.5   # no PO = assign ratio of 1.5 (above threshold)
)

# Feature 3: Days between invoice date and vendor creation date
# Very small number = new vendor transacting immediately (ghost vendor signal)
df["feat_vendor_age_days"] = (
    df["invoice_date"] - df["vendor_creation_date"]
).dt.days.clip(lower=0).fillna(365)

# Feature 4: Days to payment (how fast was the invoice paid)
# Unusually fast payment can indicate collusion
df["feat_days_to_payment"] = (
    df["payment_date"] - df["invoice_date"]
).dt.days.fillna(90).clip(0, 200)

# Feature 5: Invoice frequency — how many invoices did this vendor
# submit in the same month? High frequency = potential billing abuse
inv_month_freq = df.groupby(
    ["vendor_id", df["invoice_date"].dt.to_period("M")]
)["invoice_number"].transform("count")
df["feat_vendor_monthly_freq"] = inv_month_freq

# Feature 6: Invoice amount as % of cost centre budget
# Close to or above 1.0 = this one invoice nearly uses the whole budget
df["feat_pct_of_cc_budget"] = np.where(
    df["cc_budget"].notna() & (df["cc_budget"] > 0),
    df["invoice_amount"] / df["cc_budget"],
    0.5
).clip(0, 5)

# Feature 7: Is the posting hour outside business hours?
# Binary flag — after-hours postings are suspicious
df["invoice_hour"] = df["invoice_date"].dt.hour
df["invoice_dow"]  = df["invoice_date"].dt.dayofweek
df["feat_after_hours"] = (
    (df["invoice_dow"] >= 5) |
    (df["invoice_hour"] < 8) |
    (df["invoice_hour"] >= 19)
).astype(int)

# Feature 8: Does the invoice have a PO? Binary flag.
# No PO = maverick buying
df["feat_no_po"] = df["po_number"].isna().astype(int)

# Feature 9: Round number indicator
# Is the invoice amount divisible by a suspicious round modulo?
def round_number_score(amount):
    for mod in [100000, 50000, 10000, 5000, 1000]:
        if amount % mod == 0:
            return 1
    return 0

df["feat_round_number"] = df["invoice_amount"].apply(round_number_score)

# Feature 10: Benford's Law first digit deviation
# Expected Benford probability for each digit
benford_p = {d: np.log10(1 + 1/d) for d in range(1, 10)}

def benford_deviation(amount):
    """How far does this amount's first digit deviate from Benford expectation?"""
    if amount <= 0:
        return 0
    first_digit = int(str(int(amount))[0])
    if first_digit not in benford_p:
        return 0
    # Deviation = absolute difference from expected probability
    return abs(1/9 - benford_p[first_digit])   # 1/9 = uniform distribution

df["feat_benford_dev"] = df["invoice_amount"].apply(benford_deviation)

# ── Collect all feature columns ───────────────────────────────────────────────
FEATURE_COLS = [
    "feat_amount_zscore",
    "feat_invoice_to_po_ratio",
    "feat_vendor_age_days",
    "feat_days_to_payment",
    "feat_vendor_monthly_freq",
    "feat_pct_of_cc_budget",
    "feat_after_hours",
    "feat_no_po",
    "feat_round_number",
    "feat_benford_dev",
]

X = df[FEATURE_COLS].copy()
y = df["anomaly_flag"].values   # ground truth for evaluation

# Handle any remaining NaN or inf values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"  Feature matrix shape: {X.shape}")
print(f"  Anomaly rate in dataset: {y.mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SCALE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
# RobustScaler is better than StandardScaler for fraud data
# because it uses median and IQR — outliers (the anomalies) don't
# distort the scaling of normal data.
scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)

print("Features scaled with RobustScaler.")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL A — ISOLATION FOREST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL A — Isolation Forest")
print("="*50)

# contamination: our estimate of what fraction of data is anomalous
# n_estimators: number of trees (more = more stable, slower)
# max_samples: how many samples each tree uses — 'auto' = min(256, n_samples)
iso_forest = IsolationForest(
    contamination = 0.05,    # ~5% anomaly rate
    n_estimators  = 200,
    max_samples   = "auto",
    random_state  = SEED,
    n_jobs        = -1,      # use all CPU cores
)

print("Training Isolation Forest...")
iso_forest.fit(X_scaled)

# Scores: negative = more anomalous, positive = more normal
# We flip the sign so higher score = more anomalous (intuitive)
iso_scores = -iso_forest.score_samples(X_scaled)

# Predictions: IsolationForest returns -1 for anomaly, 1 for normal
iso_raw_pred = iso_forest.predict(X_scaled)
iso_pred     = (iso_raw_pred == -1).astype(int)   # convert to 0/1

# Evaluate
iso_prec  = precision_score(y, iso_pred, zero_division=0)
iso_rec   = recall_score(y, iso_pred, zero_division=0)
iso_f1    = f1_score(y, iso_pred, zero_division=0)
iso_auc   = roc_auc_score(y, iso_scores)
iso_cm    = confusion_matrix(y, iso_pred)

print(f"\n  Precision:  {iso_prec:.3f}")
print(f"  Recall:     {iso_rec:.3f}")
print(f"  F1 Score:   {iso_f1:.3f}")
print(f"  ROC-AUC:    {iso_auc:.3f}")
print(f"\n  Confusion Matrix:")
print(f"  TN={iso_cm[0][0]:,}  FP={iso_cm[0][1]:,}")
print(f"  FN={iso_cm[1][0]:,}  TP={iso_cm[1][1]:,}")

# Feature importance proxy for Isolation Forest:
# run the model with one feature permuted — if score drops, that feature mattered
print("\n  Computing feature importance (permutation-based)...")
baseline_auc = iso_auc
importances  = {}
for feat in FEATURE_COLS:
    X_permuted          = X_scaled.copy()
    col_idx             = FEATURE_COLS.index(feat)
    X_permuted[:, col_idx] = np.random.permutation(X_permuted[:, col_idx])
    permuted_scores     = -iso_forest.score_samples(X_permuted)
    permuted_auc        = roc_auc_score(y, permuted_scores)
    importances[feat]   = baseline_auc - permuted_auc

imp_series = pd.Series(importances).sort_values(ascending=False)
print("\n  Feature Importances (AUC drop when permuted):")
for feat, val in imp_series.items():
    bar = "█" * int(val * 500)
    print(f"  {feat:<35} {val:.4f}  {bar}")

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 5))
imp_series.sort_values().plot(kind="barh", ax=ax, color="#2C7BB6")
ax.set_title("Isolation Forest — Feature Importance (Permutation)",
             fontweight="bold")
ax.set_xlabel("AUC Drop When Feature Permuted (higher = more important)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/iso_feature_importance.png", dpi=150)
plt.close()

# Store scores on the main dataframe
df["iso_score"]       = iso_scores
df["iso_prediction"]  = iso_pred

# ─────────────────────────────────────────────────────────────────────────────
# MODEL B — AUTOENCODER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL B — Autoencoder (Neural Network)")
print("="*50)

# The Autoencoder learns to compress and reconstruct NORMAL transactions.
# Anomalous transactions reconstruct poorly = high reconstruction error.

# Train the Autoencoder ONLY on normal transactions.
# This is the key insight: the model learns the "shape" of normal data.
# When it sees an anomaly, it does not know how to reconstruct it well.
X_normal = X_scaled[y == 0]   # only normal transactions for training
X_all    = X_scaled            # full dataset for scoring

print(f"  Training on {len(X_normal):,} normal transactions only.")
print(f"  Scoring all {len(X_all):,} transactions.")

input_dim = X_scaled.shape[1]   # number of features = 10

# ── Build the Autoencoder architecture ───────────────────────────────────────
# Encoder: compress input (10 features) → bottleneck (3 features)
# Decoder: reconstruct from bottleneck (3) back to original (10)
# The bottleneck forces the model to learn the essential "normal" pattern

inputs  = keras.Input(shape=(input_dim,), name="input")

# Encoder layers
encoded = keras.layers.Dense(16, activation="relu",
                              name="encoder_1")(inputs)
encoded = keras.layers.Dropout(0.1)(encoded)       # regularisation
encoded = keras.layers.Dense(8,  activation="relu",
                              name="encoder_2")(encoded)
encoded = keras.layers.Dense(3,  activation="relu",
                              name="bottleneck")(encoded)   # compressed representation

# Decoder layers — mirror of encoder
decoded = keras.layers.Dense(8,  activation="relu",
                              name="decoder_1")(encoded)
decoded = keras.layers.Dense(16, activation="relu",
                              name="decoder_2")(decoded)
decoded = keras.layers.Dense(input_dim, activation="linear",
                              name="output")(decoded)       # linear = regression output

autoencoder = keras.Model(inputs=inputs, outputs=decoded, name="Autoencoder")
autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"   # mean squared error — measures reconstruction quality
)

print(f"\n  Autoencoder architecture:")
autoencoder.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,          # stop if validation loss doesn't improve for 5 epochs
    restore_best_weights=True
)

history = autoencoder.fit(
    X_normal, X_normal,   # input and target are the same (reconstruction task)
    epochs          = 50,
    batch_size      = 64,
    validation_split= 0.15,
    callbacks       = [early_stop],
    verbose         = 0,   # suppress per-epoch output
)

print(f"\n  Training complete. Stopped at epoch "
      f"{len(history.history['loss'])}.")
print(f"  Final training loss:   {history.history['loss'][-1]:.6f}")
print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")

# ── Plot training loss ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history["loss"],     label="Training Loss",   color="#2C7BB6")
ax.plot(history.history["val_loss"], label="Validation Loss", color="#D7191C")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Autoencoder Training Loss", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/autoencoder_training_loss.png", dpi=150)
plt.close()

# ── Compute reconstruction error ──────────────────────────────────────────────
X_reconstructed = autoencoder.predict(X_all, verbose=0)
reconstruction_errors = np.mean((X_all - X_reconstructed) ** 2, axis=1)

# ── Choose threshold: 95th percentile of normal transaction errors ────────────
# Only look at normal transactions to set the threshold.
# Transactions with error above this threshold = predicted anomaly.
normal_errors = reconstruction_errors[y == 0]
threshold_95  = np.percentile(normal_errors, 95)
print(f"\n  Reconstruction error threshold (95th pct of normal): {threshold_95:.6f}")

ae_pred   = (reconstruction_errors > threshold_95).astype(int)
ae_scores = reconstruction_errors   # higher = more anomalous

# Evaluate
ae_prec = precision_score(y, ae_pred, zero_division=0)
ae_rec  = recall_score(y, ae_pred, zero_division=0)
ae_f1   = f1_score(y, ae_pred, zero_division=0)
ae_auc  = roc_auc_score(y, ae_scores)
ae_cm   = confusion_matrix(y, ae_pred)

print(f"\n  Precision:  {ae_prec:.3f}")
print(f"  Recall:     {ae_rec:.3f}")
print(f"  F1 Score:   {ae_f1:.3f}")
print(f"  ROC-AUC:    {ae_auc:.3f}")
print(f"\n  Confusion Matrix:")
print(f"  TN={ae_cm[0][0]:,}  FP={ae_cm[0][1]:,}")
print(f"  FN={ae_cm[1][0]:,}  TP={ae_cm[1][1]:,}")

df["ae_score"]      = ae_scores
df["ae_prediction"] = ae_pred

# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE — COMBINE BOTH MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("ENSEMBLE — Combining Both Models")
print("="*50)

# Normalise both scores to [0, 1] so they are on the same scale before combining
def min_max_normalise(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-10)

iso_norm = min_max_normalise(df["iso_score"])
ae_norm  = min_max_normalise(df["ae_score"])

# Weighted average: give slightly more weight to Isolation Forest
# because it is trained on all features; Autoencoder is trained only on normals
WEIGHT_ISO = 0.55
WEIGHT_AE  = 0.45
df["ensemble_score"] = (WEIGHT_ISO * iso_norm) + (WEIGHT_AE * ae_norm)

# Threshold: flag top ~5% as anomalies
ens_threshold = np.percentile(df["ensemble_score"], 95)
df["ensemble_prediction"] = (df["ensemble_score"] > ens_threshold).astype(int)

ens_prec = precision_score(y, df["ensemble_prediction"], zero_division=0)
ens_rec  = recall_score(y, df["ensemble_prediction"], zero_division=0)
ens_f1   = f1_score(y, df["ensemble_prediction"], zero_division=0)
ens_auc  = roc_auc_score(y, df["ensemble_score"])
ens_cm   = confusion_matrix(y, df["ensemble_prediction"])

print(f"\n  Precision:  {ens_prec:.3f}")
print(f"  Recall:     {ens_rec:.3f}")
print(f"  F1 Score:   {ens_f1:.3f}")
print(f"  ROC-AUC:    {ens_auc:.3f}")
print(f"\n  Confusion Matrix:")
print(f"  TN={ens_cm[0][0]:,}  FP={ens_cm[0][1]:,}")
print(f"  FN={ens_cm[1][0]:,}  TP={ens_cm[1][1]:,}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON CHART
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curves
for model_name, scores in [("Isolation Forest", iso_norm),
                             ("Autoencoder",      ae_norm),
                             ("Ensemble",         df["ensemble_score"])]:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, scores)
    auc_val = roc_auc_score(y, scores)
    axes[0].plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})", linewidth=2)

axes[0].plot([0,1],[0,1],"k--", alpha=0.3, label="Random")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves — Model Comparison", fontweight="bold")
axes[0].legend()

# Bar chart: precision / recall / F1
models   = ["Isolation Forest", "Autoencoder", "Ensemble"]
metrics  = {
    "Precision": [iso_prec, ae_prec, ens_prec],
    "Recall":    [iso_rec,  ae_rec,  ens_rec],
    "F1 Score":  [iso_f1,   ae_f1,   ens_f1],
}
x     = np.arange(len(models))
width = 0.25

for i, (metric, vals) in enumerate(metrics.items()):
    axes[1].bar(x + i*width, vals, width, label=metric, alpha=0.85)

axes[1].set_xticks(x + width)
axes[1].set_xticklabels(models, fontsize=9)
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1)
axes[1].set_title("Model Performance Comparison", fontweight="bold")
axes[1].legend()

plt.suptitle("ML Model Comparison — Isolation Forest vs Autoencoder vs Ensemble",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150)
plt.close()
print(f"\n  Saved model comparison chart.")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE SCORED DATASET FOR PHASES 7–9
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv(f"{OUTPUT_DIR}/invoices_scored.csv", index=False)
print(f"\n  Saved scored dataset → {OUTPUT_DIR}/invoices_scored.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 6 RESULTS SUMMARY")
print("="*60)
print(f"{'Model':<20} {'Precision':>10} {'Recall':>10} "
      f"{'F1':>10} {'ROC-AUC':>10}")
print("-"*60)
print(f"{'Isolation Forest':<20} {iso_prec:>10.3f} {iso_rec:>10.3f} "
      f"{iso_f1:>10.3f} {iso_auc:>10.3f}")
print(f"{'Autoencoder':<20} {ae_prec:>10.3f} {ae_rec:>10.3f} "
      f"{ae_f1:>10.3f} {ae_auc:>10.3f}")
print(f"{'Ensemble':<20} {ens_prec:>10.3f} {ens_rec:>10.3f} "
      f"{ens_f1:>10.3f} {ens_auc:>10.3f}")
print("="*60)
print("\n✓ Phase 6 complete.")