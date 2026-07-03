# phase5_rules.py
# Rule-based anomaly detection engine.
# Each rule replicates a standard internal audit control test.
# Output: a master exceptions report with risk scores and explanations.

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings, os
warnings.filterwarnings("ignore")

DATA_DIR   = "data/raw"
OUTPUT_DIR = "outputs/rules"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Configurable thresholds ───────────────────────────────────────────────────
# These live here so an auditor can tune them without touching detection logic
CFG = {
    "match_tolerance":        0.05,   # 5% three-way match tolerance
    "approval_threshold":  100_000,   # PO approval threshold in INR
    "split_window_days":       30,    # days window for split PO detection
    "biz_hour_start":           8,    # business hours start
    "biz_hour_end":            19,    # business hours end
    "new_vendor_days":          7,    # days from creation to first invoice
    "dup_amount_tolerance":  0.02,    # 2% tolerance for near-duplicate detection
    "dup_date_window_days":    30,    # days window for duplicate detection
    "freq_z_threshold":       2.5,    # z-score for high frequency detection
    "round_number_modulos": [1000, 5000, 10000, 50000, 100000],
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
inv = pd.read_csv(f"{DATA_DIR}/invoices.csv",
                  parse_dates=["invoice_date","due_date","payment_date"])
po  = pd.read_csv(f"{DATA_DIR}/purchase_orders.csv", parse_dates=["po_date"])
ven = pd.read_csv(f"{DATA_DIR}/vendors.csv",
                  parse_dates=["vendor_creation_date"])
bud = pd.read_csv(f"{DATA_DIR}/budget.csv")

inv["invoice_hour"] = inv["invoice_date"].dt.hour
inv["invoice_dow"]  = inv["invoice_date"].dt.dayofweek

# Master exceptions list — every rule appends to this
all_exceptions = []

def add_exception(invoice_number, rule_name, risk_score, explanation,
                  recommended_action, financial_value):
    """Append a flagged transaction to the exceptions list."""
    all_exceptions.append({
        "invoice_number":     invoice_number,
        "rule_triggered":     rule_name,
        "risk_score":         risk_score,
        "explanation":        explanation,
        "recommended_action": recommended_action,
        "financial_value":    round(financial_value, 2),
    })

# =============================================================================
# RULE 1 — EXACT DUPLICATE INVOICE DETECTION
# =============================================================================
# Logic: same vendor + same amount + same vendor_invoice_number within 30 days
# Why it matters: paying twice for the same service
# Risk score: 8 — high certainty if exact match on all three fields
print("\nRule 1: Exact Duplicate Invoices...")

inv_sorted = inv.sort_values(["vendor_id","invoice_amount","invoice_date"])

for _, grp in inv_sorted.groupby(["vendor_id","vendor_invoice_number"]):
    if len(grp) < 2:
        continue
    grp = grp.sort_values("invoice_date")
    for i in range(1, len(grp)):
        row_curr = grp.iloc[i]
        row_prev = grp.iloc[i-1]
        days_apart = (row_curr["invoice_date"] - row_prev["invoice_date"]).days
        if (abs(row_curr["invoice_amount"] - row_prev["invoice_amount"]) < 0.01
                and days_apart <= CFG["dup_date_window_days"]):
            add_exception(
                invoice_number    = row_curr["invoice_number"],
                rule_name         = "Exact Duplicate Invoice",
                risk_score        = 8,
                explanation       = (
                    f"Invoice {row_curr['invoice_number']} from vendor "
                    f"{row_curr['vendor_id']} for ₹{row_curr['invoice_amount']:,.0f} "
                    f"appears to be an exact duplicate of {row_prev['invoice_number']} "
                    f"posted {days_apart} days earlier with the same vendor invoice "
                    f"reference number."
                ),
                recommended_action = (
                    "Place payment hold. Contact vendor to confirm only one "
                    "invoice is outstanding. Investigate AP processor who posted both."
                ),
                financial_value   = row_curr["invoice_amount"],
            )

rule1_count = sum(1 for e in all_exceptions if e["rule_triggered"] == "Exact Duplicate Invoice")
print(f"  → {rule1_count} exact duplicates flagged.")

# =============================================================================
# RULE 2 — NEAR-DUPLICATE INVOICE DETECTION
# =============================================================================
# Logic: same vendor, amounts within 2%, within 30 days
# Catches sophisticated duplicates where amount was slightly altered
print("Rule 2: Near-Duplicate Invoices...")

near_dup_count = 0
for vendor_id, grp in inv.groupby("vendor_id"):
    if len(grp) < 2:
        continue
    grp = grp.sort_values("invoice_date").reset_index(drop=True)
    for i in range(len(grp)):
        for j in range(i+1, len(grp)):
            ri, rj = grp.iloc[i], grp.iloc[j]
            days_apart = abs((rj["invoice_date"] - ri["invoice_date"]).days)
            if days_apart > CFG["dup_date_window_days"]:
                break   # sorted by date, so no need to look further
            if ri["vendor_invoice_number"] == rj["vendor_invoice_number"]:
                continue   # already caught by Rule 1
            if ri["invoice_amount"] == 0:
                continue
            pct_diff = abs(rj["invoice_amount"] - ri["invoice_amount"]) \
                       / ri["invoice_amount"]
            if pct_diff <= CFG["dup_amount_tolerance"] and pct_diff > 0:
                add_exception(
                    invoice_number    = rj["invoice_number"],
                    rule_name         = "Near-Duplicate Invoice",
                    risk_score        = 6,
                    explanation       = (
                        f"Invoice {rj['invoice_number']} (₹{rj['invoice_amount']:,.0f}) "
                        f"is {pct_diff*100:.1f}% different from invoice "
                        f"{ri['invoice_number']} (₹{ri['invoice_amount']:,.0f}) "
                        f"from the same vendor, posted {days_apart} days apart. "
                        f"Small amount difference may be deliberate to evade "
                        f"exact duplicate detection."
                    ),
                    recommended_action = (
                        "Compare both invoices against original PO and GR. "
                        "Confirm only one delivery occurred. Escalate to AP Manager."
                    ),
                    financial_value   = rj["invoice_amount"],
                )
                near_dup_count += 1

print(f"  → {near_dup_count} near-duplicates flagged.")

# =============================================================================
# RULE 3 — SPLIT PO DETECTION
# =============================================================================
# Logic: multiple POs to same vendor in same 30-day window,
#        each below approval threshold, combined above it
print("Rule 3: Split Purchase Orders...")

split_count = 0
po["po_date"] = pd.to_datetime(po["po_date"])

for vendor_id, grp in po.groupby("vendor_id"):
    below_threshold = grp[grp["total_po_value"] < CFG["approval_threshold"]]\
                      .sort_values("po_date")
    if len(below_threshold) < 2:
        continue
    # Sliding 30-day window
    for i in range(len(below_threshold)):
        window_start = below_threshold.iloc[i]["po_date"]
        window_end   = window_start + pd.Timedelta(days=CFG["split_window_days"])
        window_pos   = below_threshold[
            (below_threshold["po_date"] >= window_start) &
            (below_threshold["po_date"] <= window_end)
        ]
        if len(window_pos) >= 2 and window_pos["total_po_value"].sum() \
                                     > CFG["approval_threshold"]:
            for _, po_row in window_pos.iterrows():
                add_exception(
                    invoice_number    = po_row["po_number"],
                    rule_name         = "Split Purchase Order",
                    risk_score        = 7,
                    explanation       = (
                        f"PO {po_row['po_number']} to vendor {vendor_id} "
                        f"for ₹{po_row['total_po_value']:,.0f} is part of a group "
                        f"of {len(window_pos)} POs within {CFG['split_window_days']} "
                        f"days, each below the ₹{CFG['approval_threshold']:,} "
                        f"approval threshold. Combined value: "
                        f"₹{window_pos['total_po_value'].sum():,.0f}."
                    ),
                    recommended_action = (
                        "Refer to CFO for retrospective approval. Investigate "
                        "whether this is a genuine business need or deliberate "
                        "control circumvention. Review approver's other transactions."
                    ),
                    financial_value   = po_row["total_po_value"],
                )
            split_count += len(window_pos)
            break   # avoid double-counting within same vendor

print(f"  → {split_count} split PO line items flagged.")

# =============================================================================
# RULE 4 — THREE-WAY MATCH FAILURE
# =============================================================================
print("Rule 4: Three-Way Match Failures...")

inv_with_po = inv[inv["po_number"].notna()].merge(
    po[["po_number","total_po_value"]], on="po_number", how="left"
)
inv_with_po["deviation"] = (
    (inv_with_po["invoice_amount"] - inv_with_po["total_po_value"])
    / inv_with_po["total_po_value"].replace(0, np.nan)
)

match_failures = inv_with_po[
    inv_with_po["deviation"].abs() > CFG["match_tolerance"]
].copy()

for _, row in match_failures.iterrows():
    dev_pct = row["deviation"] * 100
    direction = "OVER" if dev_pct > 0 else "UNDER"
    add_exception(
        invoice_number    = row["invoice_number"],
        rule_name         = "Three-Way Match Failure",
        risk_score        = 7 if abs(dev_pct) > 20 else 5,
        explanation       = (
            f"Invoice {row['invoice_number']} amount ₹{row['invoice_amount']:,.0f} "
            f"is {abs(dev_pct):.1f}% {direction} the PO value "
            f"₹{row['total_po_value']:,.0f}, exceeding the {CFG['match_tolerance']*100:.0f}% "
            f"tolerance. This indicates possible price manipulation or "
            f"goods shortfall."
        ),
        recommended_action = (
            "Block payment pending investigation. Verify against original PO "
            "and goods receipt. Obtain written vendor explanation for discrepancy."
        ),
        financial_value   = row["invoice_amount"],
    )

print(f"  → {len(match_failures)} three-way match failures flagged.")

# =============================================================================
# RULE 5 — MAVERICK BUYING
# =============================================================================
print("Rule 5: Maverick Buying...")

maverick = inv[inv["po_number"].isna()].copy()
for _, row in maverick.iterrows():
    add_exception(
        invoice_number    = row["invoice_number"],
        rule_name         = "Maverick Buying",
        risk_score        = 5,
        explanation       = (
            f"Invoice {row['invoice_number']} for ₹{row['invoice_amount']:,.0f} "
            f"from vendor {row['vendor_id']} has no corresponding Purchase Order. "
            f"Goods or services were procured outside the approved procurement "
            f"process, bypassing price negotiation and approval controls."
        ),
        recommended_action = (
            "Retrospective PO required from department head. Review whether "
            "this vendor is on the approved vendor list. "
            "Flag department for procurement compliance training."
        ),
        financial_value   = row["invoice_amount"],
    )

print(f"  → {len(maverick)} maverick invoices flagged.")

# =============================================================================
# RULE 6 — AFTER-HOURS TRANSACTIONS
# =============================================================================
print("Rule 6: After-Hours Transactions...")

after_hours = inv[
    (inv["invoice_dow"] >= 5) |
    (inv["invoice_hour"] < CFG["biz_hour_start"]) |
    (inv["invoice_hour"] >= CFG["biz_hour_end"])
].copy()

for _, row in after_hours.iterrows():
    if row["invoice_dow"] >= 5:
        timing_desc = f"on a {'Saturday' if row['invoice_dow']==5 else 'Sunday'}"
    elif row["invoice_hour"] < CFG["biz_hour_start"]:
        timing_desc = f"at {row['invoice_hour']:02d}:00 (before business hours)"
    else:
        timing_desc = f"at {row['invoice_hour']:02d}:00 (after business hours)"

    add_exception(
        invoice_number    = row["invoice_number"],
        rule_name         = "After-Hours Transaction",
        risk_score        = 4,
        explanation       = (
            f"Invoice {row['invoice_number']} for ₹{row['invoice_amount']:,.0f} "
            f"was posted {timing_desc}. AP postings should occur during business "
            f"hours (08:00–19:00, Mon–Fri). After-hours activity may indicate "
            f"unauthorised access or deliberate concealment."
        ),
        recommended_action = (
            "Verify the user who posted this transaction was authorised to do so. "
            "Review access logs for the posting user ID. "
            "Confirm business justification for after-hours posting."
        ),
        financial_value   = row["invoice_amount"],
    )

print(f"  → {len(after_hours)} after-hours transactions flagged.")

# =============================================================================
# RULE 7 — BUDGET OVERRUN DETECTION
# =============================================================================
print("Rule 7: Budget Overruns...")

overruns = bud[bud["actual_spend"] > bud["approved_budget"]].copy()
for _, row in overruns.iterrows():
    overrun_amt = row["actual_spend"] - row["approved_budget"]
    overrun_pct = (overrun_amt / row["approved_budget"]) * 100
    add_exception(
        invoice_number    = f"BUDGET-{row['cost_centre']}-{row['budget_period']}",
        rule_name         = "Budget Overrun",
        risk_score        = 6 if overrun_pct > 20 else 4,
        explanation       = (
            f"Cost centre {row['cost_centre']} ({row['department']}) has spent "
            f"₹{row['actual_spend']:,.0f} against an approved budget of "
            f"₹{row['approved_budget']:,.0f} in period {row['budget_period']} — "
            f"an overrun of ₹{overrun_amt:,.0f} ({overrun_pct:.1f}%)."
        ),
        recommended_action = (
            "Obtain supplementary budget approval from CFO. "
            "Review all invoices posted to this cost centre in the period. "
            "Identify the transactions that caused the overrun."
        ),
        financial_value   = overrun_amt,
    )

print(f"  → {len(overruns)} budget overruns flagged.")

# =============================================================================
# RULE 8 — ROUND NUMBER DETECTION
# =============================================================================
print("Rule 8: Round Number Invoices...")

def is_round(amount, modulos):
    """Return True if amount is exactly divisible by any round modulo."""
    return any(amount % m == 0 for m in modulos)

round_mask = inv["invoice_amount"].apply(
    lambda x: is_round(x, CFG["round_number_modulos"])
)
round_invoices = inv[round_mask].copy()

for _, row in round_invoices.iterrows():
    add_exception(
        invoice_number    = row["invoice_number"],
        rule_name         = "Round Number Invoice",
        risk_score        = 3,
        explanation       = (
            f"Invoice {row['invoice_number']} amount ₹{row['invoice_amount']:,.0f} "
            f"is a suspiciously round number. Genuine invoices for services or "
            f"goods rarely end in exact thousands or multiples of ₹10,000. "
            f"Round amounts are a recognised fraud indicator in forensic accounting."
        ),
        recommended_action = (
            "Request supporting documentation (timesheets, delivery notes) "
            "to verify the invoice amount is legitimate. "
            "Check vendor's other invoices for similar round-number pattern."
        ),
        financial_value   = row["invoice_amount"],
    )

print(f"  → {len(round_invoices)} round number invoices flagged.")

# =============================================================================
# RULE 9 — NEW VENDOR FAST PAYMENT
# =============================================================================
print("Rule 9: New Vendor Fast Payment...")

inv_ven = inv.merge(
    ven[["vendor_id","vendor_creation_date","is_ghost_vendor"]],
    on="vendor_id", how="left"
)
inv_ven["vendor_age_days"] = (
    inv_ven["invoice_date"] - pd.to_datetime(inv_ven["vendor_creation_date"])
).dt.days

new_vendor_inv = inv_ven[
    inv_ven["vendor_age_days"] <= CFG["new_vendor_days"]
].copy()

for _, row in new_vendor_inv.iterrows():
    ghost_note = " (VENDOR FLAGGED AS POTENTIAL GHOST VENDOR)" \
                 if row.get("is_ghost_vendor", False) else ""
    add_exception(
        invoice_number    = row["invoice_number"],
        rule_name         = "New Vendor Fast Payment",
        risk_score        = 9 if row.get("is_ghost_vendor") else 6,
        explanation       = (
            f"Invoice {row['invoice_number']} for ₹{row['invoice_amount']:,.0f} "
            f"was received from vendor {row['vendor_id']} only "
            f"{row['vendor_age_days']:.0f} days after the vendor was created "
            f"in the system{ghost_note}. New vendors transacting immediately "
            f"are a primary ghost vendor indicator."
        ),
        recommended_action = (
            "Verify vendor legitimacy with external business registration check. "
            "Confirm bank account details with vendor via independent channel. "
            "Review who created this vendor and whether they also processed "
            "the invoice (SoD violation)."
        ),
        financial_value   = row["invoice_amount"],
    )

print(f"  → {len(new_vendor_inv)} new vendor fast payment transactions flagged.")

# =============================================================================
# MASTER EXCEPTIONS REPORT
# =============================================================================
print("\nBuilding master exceptions report...")

exceptions_df = pd.DataFrame(all_exceptions)

# Deduplicate: if same invoice flagged by multiple rules, keep highest risk score
exceptions_df = exceptions_df.sort_values("risk_score", ascending=False)\
                              .drop_duplicates(subset="invoice_number", keep="first")

# Risk tier
def assign_tier(score):
    if score >= 8: return "Critical"
    if score >= 6: return "High"
    if score >= 4: return "Medium"
    return "Low"

exceptions_df["risk_tier"] = exceptions_df["risk_score"].apply(assign_tier)

# Save
exceptions_path = f"{OUTPUT_DIR}/master_exceptions_report.csv"
exceptions_df.to_csv(exceptions_path, index=False)
print(f"  Saved: {exceptions_path}")
print(f"  Total exceptions: {len(exceptions_df):,}")
print(f"\n  Risk Tier Distribution:")
print(exceptions_df["risk_tier"].value_counts().to_string())
print(f"\n  Total Financial Exposure Flagged: "
      f"₹{exceptions_df['financial_value'].sum():,.0f}")

# =============================================================================
# EVALUATION AGAINST GROUND TRUTH
# =============================================================================
# This is only possible because we injected labels in Phase 2.
# In a real engagement, you would not have this luxury.
print("\n=== Evaluation Against Ground Truth Labels ===")

# Join exceptions back to invoice ground truth
inv_truth = inv[["invoice_number","anomaly_flag"]].copy()

# Every invoice: predicted 1 if in exceptions, else 0
all_inv_numbers      = set(inv["invoice_number"])
flagged_inv_numbers  = set(exceptions_df["invoice_number"]) & all_inv_numbers

inv_truth["predicted"] = inv_truth["invoice_number"].apply(
    lambda x: 1 if x in flagged_inv_numbers else 0
)

y_true = inv_truth["anomaly_flag"]
y_pred = inv_truth["predicted"]

precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)
cm        = confusion_matrix(y_true, y_pred)

print(f"\n  Precision: {precision:.3f}  "
      f"→ Of transactions we flagged, {precision*100:.1f}% are truly anomalous")
print(f"  Recall:    {recall:.3f}  "
      f"→ Of all true anomalies, we caught {recall*100:.1f}%")
print(f"  F1 Score:  {f1:.3f}")
print(f"\n  Confusion Matrix:")
print(f"  True Negatives  (correct non-flags): {cm[0][0]:>6,}")
print(f"  False Positives (incorrectly flagged): {cm[0][1]:>5,}")
print(f"  False Negatives (missed anomalies):   {cm[1][0]:>5,}")
print(f"  True Positives  (correctly caught):   {cm[1][1]:>5,}")
print(f"\n  → In fraud detection, recall matters more than precision.")
print(f"    Missing real fraud (FN) costs money. Reviewing a false alarm (FP)")
print(f"    costs only auditor time.")

print("\n✓ Phase 5 complete.")