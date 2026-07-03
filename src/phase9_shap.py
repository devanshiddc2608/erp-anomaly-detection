# phase9_shap.py
# Applies SHAP to the Isolation Forest model to explain
# why each transaction was flagged as anomalous.

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

ML_DIR     = "outputs/ml"
RISK_DIR   = "outputs/risk_scoring"
OUTPUT_DIR = "outputs/shap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "feat_amount_zscore", "feat_invoice_to_po_ratio", "feat_vendor_age_days",
    "feat_days_to_payment", "feat_vendor_monthly_freq", "feat_pct_of_cc_budget",
    "feat_after_hours", "feat_no_po", "feat_round_number", "feat_benford_dev",
]

FEATURE_LABELS = {
    "feat_amount_zscore":        "Amount vs Vendor Average",
    "feat_invoice_to_po_ratio":  "Invoice-to-PO Ratio",
    "feat_vendor_age_days":      "Vendor Age (days)",
    "feat_days_to_payment":      "Days to Payment",
    "feat_vendor_monthly_freq":  "Vendor Invoice Frequency",
    "feat_pct_of_cc_budget":     "% of Cost Centre Budget",
    "feat_after_hours":          "After-Hours Posting",
    "feat_no_po":                "No PO Reference",
    "feat_round_number":         "Round Number Amount",
    "feat_benford_dev":          "Benford's Law Deviation",
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — RECREATE THE TRAINED MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Note: phase6_ml.py didn't save the trained model object, only scores.
# We retrain identically here (same seed = same model) so SHAP has
# a live model object to work with. This is a one-time re-run, not
# duplicated effort — production pipelines would just load a pickled model.

print("Reloading scored data and retraining Isolation Forest for SHAP...")
df = pd.read_csv(f"{ML_DIR}/invoices_scored.csv",
                 parse_dates=["invoice_date","due_date","payment_date"])

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

SEED = 42
iso_forest = IsolationForest(
    contamination=0.05, n_estimators=200, max_samples="auto",
    random_state=SEED, n_jobs=-1,
)
iso_forest.fit(X_scaled)
print("  Model retrained (identical to Phase 6 — same seed, same params).")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD SHAP EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding SHAP TreeExplainer...")
explainer = shap.TreeExplainer(iso_forest)

# For performance on large datasets, compute SHAP values on a representative
# sample rather than all 13,000+ rows — 2,000 is plenty for the summary plot
# and we compute full values only for flagged transactions separately.
sample_size = min(2000, len(X_scaled_df))
sample_idx  = np.random.RandomState(SEED).choice(
    len(X_scaled_df), sample_size, replace=False
)
X_sample = X_scaled_df.iloc[sample_idx]

print(f"  Computing SHAP values on {sample_size:,} sampled transactions...")
shap_values_sample = explainer.shap_values(X_sample)
print("  Done.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — GLOBAL SHAP SUMMARY PLOT
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating SHAP summary plot (global feature importance)...")

X_sample_labelled = X_sample.rename(columns=FEATURE_LABELS)

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_sample, X_sample_labelled,
    show=False, plot_size=None
)
plt.title("SHAP Summary — What Drives Anomaly Detection Globally",
          fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/shap_summary_plot.png")

# Also save a clean bar version (mean |SHAP value|) — easier to read for
# a non-technical audience like an audit committee
plt.figure(figsize=(9, 5))
shap.summary_plot(
    shap_values_sample, X_sample_labelled,
    plot_type="bar", show=False
)
plt.title("Average Impact of Each Feature on Anomaly Score",
          fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_bar_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/shap_bar_plot.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — WATERFALL PLOT FOR A SINGLE FLAGGED TRANSACTION
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating SHAP waterfall plot for top flagged transaction...")

risk_df = pd.read_csv(f"{RISK_DIR}/powerbi_master_export.csv")
top_case = risk_df.sort_values("composite_risk_score", ascending=False).iloc[0]
top_invoice_number = top_case["invoice_number"]

# Find this transaction's row in the original feature matrix
case_idx = df[df["invoice_number"] == top_invoice_number].index[0]
case_features = X_scaled_df.iloc[[case_idx]]
case_features_labelled = case_features.rename(columns=FEATURE_LABELS)

case_shap_values = explainer.shap_values(case_features)
case_base_value = explainer.expected_value

# Fix: ensure scalar
if isinstance(case_base_value, (list, np.ndarray)):
    case_base_value = case_base_value[0]

explanation_obj = shap.Explanation(
    values        = case_shap_values[0],
    base_values   = case_base_value,
    data          = case_features_labelled.iloc[0].values,
    feature_names = case_features_labelled.columns.tolist(),
)

plt.figure(figsize=(10, 6))
shap.plots.waterfall(explanation_obj, show=False, max_display=10)
plt.title(f"Why Invoice {top_invoice_number} Was Flagged",
          fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_waterfall_top_case.png", dpi=150,
           bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/shap_waterfall_top_case.png")
print(f"  Case explained: {top_invoice_number} "
      f"(risk score: {top_case['composite_risk_score']:.1f})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — PLAIN ENGLISH TRANSLATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def translate_shap_to_english(invoice_number: str, top_n: int = 3) -> str:
    """
    Convert SHAP values for one transaction into a plain English
    audit explanation — e.g. 'flagged primarily because the amount
    is 340% above this vendor's historical average and was posted
    at 2am on a Sunday.'
    """
    row_idx = df[df["invoice_number"] == invoice_number].index
    if len(row_idx) == 0:
        return "Invoice not found."
    row_idx = row_idx[0]

    case_X      = X_scaled_df.iloc[[row_idx]]
    case_shap   = explainer.shap_values(case_X)[0]
    raw_values  = df.loc[row_idx, FEATURE_COLS]

    # Rank features by absolute SHAP contribution
    contributions = pd.Series(case_shap, index=FEATURE_COLS).sort_values(
        key=abs, ascending=False
    )

    explanations = []
    for feat in contributions.index[:top_n]:
        shap_val = contributions[feat]
        raw_val  = raw_values[feat]
        if shap_val <= 0:
            continue   # only explain features pushing TOWARD anomaly

        if feat == "feat_amount_zscore":
            explanations.append(
                f"the invoice amount is {raw_val:.1f} standard deviations "
                f"from this vendor's historical average"
            )
        elif feat == "feat_invoice_to_po_ratio":
            pct = (raw_val - 1) * 100
            explanations.append(
                f"the invoice amount is {pct:.0f}% {'above' if pct>0 else 'below'} "
                f"the original PO value"
            )
        elif feat == "feat_vendor_age_days":
            explanations.append(
                f"the vendor was created only {raw_val:.0f} days "
                f"before this invoice was submitted"
            )
        elif feat == "feat_after_hours":
            if raw_val == 1:
                explanations.append("it was posted outside normal business hours")
        elif feat == "feat_no_po":
            if raw_val == 1:
                explanations.append("it has no corresponding Purchase Order")
        elif feat == "feat_round_number":
            if raw_val == 1:
                explanations.append("the amount is a suspiciously round number")
        elif feat == "feat_vendor_monthly_freq":
            explanations.append(
                f"this vendor submitted {raw_val:.0f} invoices in the same month"
            )
        elif feat == "feat_pct_of_cc_budget":
            explanations.append(
                f"this single invoice consumes {raw_val*100:.1f}% "
                f"of its cost centre's annual budget"
            )
        elif feat == "feat_benford_dev":
            explanations.append(
                "the amount's leading digit deviates from the expected "
                "Benford's Law distribution"
            )

    if not explanations:
        return "No single dominant risk factor identified."

    return "This invoice was flagged primarily because " + \
           "; and ".join(explanations) + "."

# Demonstrate on the top 5 highest-risk transactions
print("\n=== Sample Plain-English Explanations (Top 5 Cases) ===")
top5 = risk_df.sort_values("composite_risk_score", ascending=False).head(5)
plain_explanations = []
for _, case in top5.iterrows():
    explanation = translate_shap_to_english(case["invoice_number"])
    plain_explanations.append({
        "invoice_number": case["invoice_number"],
        "risk_score":     case["composite_risk_score"],
        "shap_explanation": explanation,
    })
    print(f"\n  {case['invoice_number']} (Risk Score: {case['composite_risk_score']:.1f})")
    print(f"  {explanation}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — ADD SHAP EXPLANATIONS TO THE FULL CASE MANAGEMENT OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating SHAP explanations for all Critical/High tier cases...")

critical_high = risk_df[risk_df["risk_tier"].isin(["Critical","High"])].copy()
print(f"  Processing {len(critical_high)} transactions "
      f"(this may take a moment)...")

shap_explanations = []
for inv_num in critical_high["invoice_number"]:
    shap_explanations.append(translate_shap_to_english(inv_num))

critical_high["shap_explanation"] = shap_explanations

final_case_output_path = f"{OUTPUT_DIR}/case_management_with_shap.csv"
critical_high.to_csv(final_case_output_path, index=False)
print(f"\n  Saved: {final_case_output_path}")
print(f"  Every Critical/High transaction now has a SHAP-backed explanation.")

print("\n✓ Phase 9 complete.")