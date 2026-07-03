# phase7_risk_scoring.py
# Combines ML anomaly scores, rule-based flags, and transaction value
# into a single 0-100 composite risk score, then builds a prioritised
# case management output for auditor review.

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

ML_OUTPUT_DIR    = "outputs/ml"
RULES_OUTPUT_DIR = "outputs/rules"
OUTPUT_DIR       = "outputs/risk_scoring"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD ML-SCORED DATA AND RULE EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("Loading scored data...")
df = pd.read_csv(f"{ML_OUTPUT_DIR}/invoices_scored.csv",
                 parse_dates=["invoice_date", "due_date", "payment_date"])
rules_df = pd.read_csv(f"{RULES_OUTPUT_DIR}/master_exceptions_report.csv")

print(f"  Scored invoices: {len(df):,}")
print(f"  Rule exceptions: {len(rules_df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MERGE RULE FLAGS ONTO ML-SCORED DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
# We need: which rule(s) fired, and the rule-based risk score (1-10),
# for every invoice. Not every invoice has a rule hit — that's fine.
rule_lookup = rules_df.set_index("invoice_number")[
    ["rule_triggered", "risk_score", "explanation", "recommended_action"]
].rename(columns={
    "risk_score":         "rule_risk_score",
    "explanation":         "rule_explanation",
    "recommended_action":  "rule_recommended_action",
})

df = df.merge(rule_lookup, left_on="invoice_number", right_index=True, how="left")
df["rule_risk_score"] = df["rule_risk_score"].fillna(0)
df["rule_triggered"]  = df["rule_triggered"].fillna("None")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD THE COMPOSITE RISK SCORE (0-100)
# ─────────────────────────────────────────────────────────────────────────────
# Three components, each normalised to 0-100, then weighted:
#
#   1. ML Anomaly Component (40%) — how unusual is this transaction
#      statistically? Uses the ensemble_score from Phase 6.
#   2. Rule Component (35%) — did this transaction violate a known,
#      named control? Rule-based scores carry real audit weight because
#      they map to specific, defensible findings.
#   3. Value Component (25%) — a ₹50,000 anomaly and a ₹50,00,000 anomaly
#      are not equally urgent. Larger transactions get a higher score
#      even at the same anomaly strength, because financial exposure
#      is what a CFO ultimately cares about.
#
# Why these weights: rules and ML together (75%) dominate because they
# represent actual evidence of irregularity. Value (25%) is a multiplier
# of urgency, not the primary driver — a small but blatant duplicate
# invoice still deserves serious attention.

WEIGHT_ML    = 0.40
WEIGHT_RULE  = 0.35
WEIGHT_VALUE = 0.25

def normalise_0_100(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    return ((series - mn) / (mx - mn + 1e-10)) * 100

# Component 1: ML score, normalised
df["ml_component"] = normalise_0_100(df["ensemble_score"])

# Component 2: Rule score, normalised (rule scores are already 0-10, scale to 0-100)
df["rule_component"] = df["rule_risk_score"] * 10

# Component 3: Value score — log scale because invoice amounts are
# extremely right-skewed (a few huge invoices would otherwise dominate
# the raw scale and flatten everything else to near-zero)
df["log_amount"]    = np.log1p(df["invoice_amount"].clip(lower=0))
df["value_component"] = normalise_0_100(df["log_amount"])

# Composite score
df["composite_risk_score"] = (
    WEIGHT_ML    * df["ml_component"] +
    WEIGHT_RULE  * df["rule_component"] +
    WEIGHT_VALUE * df["value_component"]
).round(1)

print(f"\nComposite risk score computed.")
print(f"  Range: {df['composite_risk_score'].min():.1f} "
      f"to {df['composite_risk_score'].max():.1f}")
print(f"  Mean:  {df['composite_risk_score'].mean():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — RISK TIER CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
# Business definitions for each tier — this is what goes in your audit report.

def classify_tier(score):
    if score >= 75:
        return "Critical"
    elif score >= 55:
        return "High"
    elif score >= 35:
        return "Medium"
    else:
        return "Low"

df["risk_tier"] = df["composite_risk_score"].apply(classify_tier)

TIER_DEFINITIONS = {
    "Critical": "Strong statistical anomaly AND a named control violation, "
                "high financial value. Immediate investigation required; "
                "payment hold recommended.",
    "High":     "Either a strong ML anomaly signal or a confirmed rule "
                "violation, moderate-to-high value. Investigate within "
                "5 business days.",
    "Medium":   "Some anomaly signal present but not corroborated by "
                "multiple sources. Review within standard audit cycle.",
    "Low":      "Statistically unremarkable, no rule violations. "
                "No action required; retained for audit trail completeness.",
}

RECOMMENDED_ACTION = {
    "Critical": "Hold payment immediately. Escalate to AP Manager and "
                "Internal Audit. Investigate within 24-48 hours.",
    "High":     "Flag for priority review. Notify cost centre owner. "
                "Investigate within 5 business days.",
    "Medium":   "Add to standard audit sample for the period. "
                "No immediate hold required.",
    "Low":      "No action required. Retain in audit trail.",
}

df["tier_definition"]      = df["risk_tier"].map(TIER_DEFINITIONS)
df["tier_recommended_action"] = df["risk_tier"].map(RECOMMENDED_ACTION)

print("\nRisk Tier Distribution:")
tier_counts = df["risk_tier"].value_counts()
for tier in ["Critical", "High", "Medium", "Low"]:
    count = tier_counts.get(tier, 0)
    pct   = count / len(df) * 100
    print(f"  {tier:<10} {count:>6,}  ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FINANCIAL EXPOSURE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
print("\nFinancial Exposure by Tier:")
exposure_by_tier = df.groupby("risk_tier")["invoice_amount"].agg(
    ["sum", "mean", "count"]
).reindex(["Critical", "High", "Medium", "Low"])

for tier, row in exposure_by_tier.iterrows():
    print(f"  {tier:<10} Total: ₹{row['sum']:>15,.0f}   "
          f"Avg: ₹{row['mean']:>12,.0f}   Count: {row['count']:>6,.0f}")

total_exposure          = df[df["risk_tier"].isin(["Critical","High"])]["invoice_amount"].sum()
critical_high_count     = df["risk_tier"].isin(["Critical","High"]).sum()

print(f"\n  TOTAL FINANCIAL EXPOSURE (Critical + High tiers): "
      f"₹{total_exposure:,.0f}")
print(f"  Transactions requiring priority investigation: {critical_high_count:,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — AUDIT EFFICIENCY METRIC
# ─────────────────────────────────────────────────────────────────────────────
# This is the headline number for your resume:
# "the AI filtered X% of transactions, leaving only Y for manual review"
total_transactions = len(df)
needs_review        = (df["risk_tier"].isin(["Critical", "High"])).sum()
auto_cleared        = total_transactions - needs_review
efficiency_pct       = (auto_cleared / total_transactions) * 100

print(f"\n=== AUDIT EFFICIENCY METRIC ===")
print(f"  Total transactions analysed:        {total_transactions:,}")
print(f"  Transactions requiring human review: {needs_review:,}")
print(f"  Transactions auto-cleared by AI:     {auto_cleared:,}")
print(f"  Efficiency gain:                     {efficiency_pct:.1f}%")
print(f"  → A human auditor reviewing 100% of transactions manually would")
print(f"    need to examine {total_transactions:,} invoices.")
print(f"    With this system, only {needs_review:,} require review —")
print(f"    a {efficiency_pct:.1f}% reduction in manual audit workload.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PLAIN ENGLISH EXPLANATION FOR EACH FLAGGED TRANSACTION
# ─────────────────────────────────────────────────────────────────────────────
def build_explanation(row):
    """Combine ML and rule signals into one auditor-readable explanation."""
    parts = []
    if row["rule_triggered"] != "None":
        parts.append(row["rule_explanation"])
    if row["ml_component"] > 70:
        parts.append(
            f"Statistical model flagged this transaction as a strong outlier "
            f"(ensemble anomaly score in top {100-row['ml_component']:.0f}th percentile)."
        )
    if not parts:
        parts.append("No specific rule or strong statistical signal — "
                     "included for audit trail completeness only.")
    return " ".join(parts)

df["full_explanation"] = df.apply(build_explanation, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — BUILD CASE MANAGEMENT OUTPUT (Auditor's Daily Work Queue)
# ─────────────────────────────────────────────────────────────────────────────
case_management_cols = [
    "invoice_number", "invoice_date", "vendor_id", "invoice_amount",
    "composite_risk_score", "risk_tier", "rule_triggered",
    "full_explanation", "tier_recommended_action", "cost_centre",
    "invoice_status",
]

case_queue = df[case_management_cols].sort_values(
    "composite_risk_score", ascending=False
).reset_index(drop=True)
case_queue.insert(0, "priority_rank", range(1, len(case_queue) + 1))

case_queue_path = f"{OUTPUT_DIR}/case_management_queue.csv"
case_queue.to_csv(case_queue_path, index=False)
print(f"\nSaved case management queue → {case_queue_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — SAVE FULL SCORED DATASET (for Power BI in Phase 8)
# ─────────────────────────────────────────────────────────────────────────────
# This is the master export — every column Power BI needs lives here.
powerbi_export_cols = [
    "invoice_number", "invoice_date", "due_date", "payment_date",
    "vendor_id", "po_number", "gr_number",
    "invoice_amount", "tax_amount", "total_amount",
    "invoice_status", "cost_centre", "department",
    "composite_risk_score", "risk_tier", "ml_component",
    "rule_component", "value_component",
    "rule_triggered", "rule_risk_score", "full_explanation",
    "tier_recommended_action", "anomaly_flag", "anomaly_type",
    "iso_score", "ae_score", "ensemble_score",
    "feat_amount_zscore", "feat_invoice_to_po_ratio",
    "feat_vendor_age_days", "feat_days_to_payment",
    "feat_after_hours", "feat_no_po", "feat_round_number",
]

# Some cols (like 'department') may not exist on df yet — add if missing
if "department" not in df.columns:
    po_dept = pd.read_csv("data/raw/purchase_orders.csv")[
        ["po_number", "department"]
    ].drop_duplicates()
    df = df.merge(po_dept, on="po_number", how="left")

available_cols = [c for c in powerbi_export_cols if c in df.columns]
powerbi_export = df[available_cols].copy()

powerbi_export_path = f"{OUTPUT_DIR}/powerbi_master_export.csv"
powerbi_export.to_csv(powerbi_export_path, index=False)
print(f"Saved Power BI master export → {powerbi_export_path}")
print(f"  Columns: {len(available_cols)}  |  Rows: {len(powerbi_export):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — SUPPORTING TABLES FOR POWER BI (vendor risk, budget, etc.)
# ─────────────────────────────────────────────────────────────────────────────
# Vendor-level risk rollup
vendor_risk = df.groupby("vendor_id").agg(
    total_invoiced     = ("invoice_amount", "sum"),
    invoice_count       = ("invoice_number", "count"),
    avg_risk_score      = ("composite_risk_score", "mean"),
    max_risk_score       = ("composite_risk_score", "max"),
    critical_count       = ("risk_tier", lambda x: (x == "Critical").sum()),
    high_count           = ("risk_tier", lambda x: (x == "High").sum()),
).reset_index()

vendors_master = pd.read_csv("data/raw/vendors.csv",
                             parse_dates=["vendor_creation_date"])
vendor_risk = vendor_risk.merge(
    vendors_master[["vendor_id","vendor_name","vendor_category",
                     "vendor_creation_date","is_ghost_vendor"]],
    on="vendor_id", how="left"
)
vendor_risk["avg_risk_score"] = vendor_risk["avg_risk_score"].round(1)
vendor_risk_path = f"{OUTPUT_DIR}/vendor_risk_summary.csv"
vendor_risk.to_csv(vendor_risk_path, index=False)
print(f"Saved vendor risk summary → {vendor_risk_path}")

# Budget table pass-through (already has is_overrun from Phase 2 patch)
budget_df = pd.read_csv("data/raw/budget.csv")
budget_path = f"{OUTPUT_DIR}/budget_for_powerbi.csv"
budget_df.to_csv(budget_path, index=False)
print(f"Saved budget table → {budget_path}")

print("\n✓ Phase 7 complete. All outputs ready for Power BI import.")