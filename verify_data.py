# verify_data.py
# Sanity-checks the generated CSVs before loading into PostgreSQL.
# Confirms each injected anomaly type actually looks the way it should
# in the raw data — not just that the row counts match.

import pandas as pd
import numpy as np
import os

DATA_DIR = "data/raw"

def section(title):
    print(f"\n{'='*60}")
    print(title)
    print('='*60)

# ── Load everything ────────────────────────────────────────────────────────
users = pd.read_csv(f"{DATA_DIR}/users.csv")
vendors = pd.read_csv(f"{DATA_DIR}/vendors.csv", parse_dates=["vendor_creation_date"])
budget = pd.read_csv(f"{DATA_DIR}/budget.csv")
po = pd.read_csv(f"{DATA_DIR}/purchase_orders.csv", parse_dates=["po_date"])
gr = pd.read_csv(f"{DATA_DIR}/goods_receipts.csv", parse_dates=["gr_date"])
inv = pd.read_csv(f"{DATA_DIR}/invoices.csv",
                   parse_dates=["invoice_date", "due_date", "payment_date"])

# =============================================================================
# CHECK 1 — ROW COUNTS AND NULLS
# =============================================================================
section("CHECK 1: Basic Shape and Nulls")

tables = {"users": users, "vendors": vendors, "budget": budget,
          "purchase_orders": po, "goods_receipts": gr, "invoices": inv}

for name, df in tables.items():
    print(f"\n{name}: {len(df):,} rows, {df.shape[1]} columns")
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        for col in null_cols:
            n_null = df[col].isnull().sum()
            pct = n_null / len(df) * 100
            print(f"  NULLs in '{col}': {n_null} ({pct:.1f}%)")

print("\n  Expected NULLs (these are fine, not bugs):")
print("  - invoices.po_number / gr_number → maverick + ghost vendor + round number invoices")
print("  - invoices.payment_date / approved_by → blocked match-failure invoices")

# =============================================================================
# CHECK 2 — PRIMARY KEY UNIQUENESS
# =============================================================================
section("CHECK 2: Primary Key Uniqueness")

pk_checks = {
    "users.user_id": users["user_id"],
    "vendors.vendor_id": vendors["vendor_id"],
    "purchase_orders.po_number": po["po_number"],
    "goods_receipts.gr_number": gr["gr_number"],
    "invoices.invoice_number": inv["invoice_number"],
}
for label, series in pk_checks.items():
    dupes = series.duplicated().sum()
    status = "✓ OK" if dupes == 0 else f"✗ FAIL — {dupes} duplicates"
    print(f"  {label:<35} {status}")

# =============================================================================
# CHECK 3 — REFERENTIAL INTEGRITY (Foreign Keys)
# =============================================================================
section("CHECK 3: Referential Integrity")

# PO → Vendor
orphan_po_vendor = ~po["vendor_id"].isin(vendors["vendor_id"])
print(f"  PO records with invalid vendor_id: {orphan_po_vendor.sum()}")

# GR → PO
orphan_gr_po = ~gr["po_number"].isin(po["po_number"])
print(f"  GR records with invalid po_number: {orphan_gr_po.sum()}")

# Invoice → PO (only check where po_number is not null)
inv_with_po = inv[inv["po_number"].notna()]
orphan_inv_po = ~inv_with_po["po_number"].isin(po["po_number"])
print(f"  Invoice records with invalid po_number "
      f"(excl. NULLs): {orphan_inv_po.sum()}")

# Invoice → Vendor
orphan_inv_vendor = ~inv["vendor_id"].isin(vendors["vendor_id"])
print(f"  Invoice records with invalid vendor_id: {orphan_inv_vendor.sum()}")

# Invoice → GR (only check where gr_number is not null)
inv_with_gr = inv[inv["gr_number"].notna()]
orphan_inv_gr = ~inv_with_gr["gr_number"].isin(gr["gr_number"])
print(f"  Invoice records with invalid gr_number "
      f"(excl. NULLs): {orphan_inv_gr.sum()}")

print("\n  All counts above should be 0. Non-zero = broken foreign key.")

# =============================================================================
# CHECK 4 — DUPLICATE INVOICE INJECTION
# =============================================================================
section("CHECK 4: Duplicate Invoice Anomaly")

dup_invoices = inv[inv["is_duplicate"] == True]
print(f"  Labelled duplicate invoices: {len(dup_invoices)}")

if len(dup_invoices) > 0:
    sample = dup_invoices.iloc[0]
    same_vendor_invoice = inv[
        (inv["vendor_id"] == sample["vendor_id"]) &
        (inv["vendor_invoice_number"] == sample["vendor_invoice_number"])
    ]
    print(f"\n  Sample check — invoice {sample['invoice_number']}:")
    print(f"  Same vendor_invoice_number '{sample['vendor_invoice_number']}' "
          f"appears {len(same_vendor_invoice)} times (should be ≥2)")
    print(same_vendor_invoice[["invoice_number", "vendor_id",
                                "invoice_amount", "invoice_date"]].to_string(index=False))

# =============================================================================
# CHECK 5 — NEAR-DUPLICATE INVOICE INJECTION
# =============================================================================
section("CHECK 5: Near-Duplicate Invoice Anomaly")

near_dup = inv[inv["is_near_duplicate"] == True]
print(f"  Labelled near-duplicate invoices: {len(near_dup)}")
if len(near_dup) > 0:
    print(f"\n  Amount tweak range check (should be 0.5%-2% from original):")
    print(near_dup[["invoice_number", "vendor_id", "invoice_amount"]].head(5).to_string(index=False))

# =============================================================================
# CHECK 6 — SPLIT PO INJECTION
# =============================================================================
section("CHECK 6: Split PO Anomaly")

split_pos = po[po["is_split_po"] == True]
print(f"  Labelled split POs: {len(split_pos)}")
print(f"  All below ₹100,000 threshold: "
      f"{(split_pos['total_po_value'] < 100000).all()}")

if len(split_pos) > 0:
    # Check grouping — same vendor, same approx period, summing above threshold
    grp_check = split_pos.groupby("vendor_id")["total_po_value"].agg(["count", "sum"])
    grp_check = grp_check[grp_check["sum"] > 100000]
    print(f"\n  Vendor groups where split POs sum above threshold: {len(grp_check)}")
    print(grp_check.head(5).to_string())

# =============================================================================
# CHECK 7 — THREE-WAY MATCH FAILURE INJECTION
# =============================================================================
section("CHECK 7: Three-Way Match Failure Anomaly")

match_fail = inv[inv["is_match_failure"] == True]
print(f"  Labelled match failure invoices: {len(match_fail)}")

if len(match_fail) > 0:
    check = match_fail.merge(
        po[["po_number", "total_po_value"]], on="po_number", how="left"
    )
    check["deviation_pct"] = (
        (check["invoice_amount"] - check["total_po_value"])
        / check["total_po_value"] * 100
    )
    outside_tolerance = (check["deviation_pct"].abs() > 5).sum()
    print(f"  Of these, {outside_tolerance}/{len(check)} actually exceed "
          f"the 5% tolerance (should be {len(check)}/{len(check)})")
    print(f"  Deviation range: {check['deviation_pct'].min():.1f}% to "
          f"{check['deviation_pct'].max():.1f}%")
    print(f"  All marked 'Blocked' status: "
          f"{(match_fail['invoice_status'] == 'Blocked').all()}")

# =============================================================================
# CHECK 8 — MAVERICK BUYING INJECTION
# =============================================================================
section("CHECK 8: Maverick Buying Anomaly")

maverick = inv[inv["is_maverick"] == True]
print(f"  Labelled maverick invoices: {len(maverick)}")
print(f"  All have NULL po_number: {maverick['po_number'].isna().all()}")
print(f"  All have NULL gr_number: {maverick['gr_number'].isna().all()}")

# =============================================================================
# CHECK 9 — AFTER-HOURS INJECTION
# =============================================================================
section("CHECK 9: After-Hours Anomaly")

after_hrs = inv[inv["is_after_hours"] == True]
print(f"  Labelled after-hours invoices: {len(after_hrs)}")

if len(after_hrs) > 0:
    hours = after_hrs["invoice_date"].dt.hour
    dow = after_hrs["invoice_date"].dt.dayofweek
    is_actually_after_hours = (
        (dow >= 5) | (hours < 8) | (hours >= 19)
    )
    print(f"  Of these, {is_actually_after_hours.sum()}/{len(after_hrs)} "
          f"are genuinely outside business hours (should be {len(after_hrs)}/{len(after_hrs)})")

# =============================================================================
# CHECK 10 — ROUND NUMBER INJECTION
# =============================================================================
section("CHECK 10: Round Number Anomaly")

round_num = inv[inv["is_round_number"] == True]
print(f"  Labelled round number invoices: {len(round_num)}")
if len(round_num) > 0:
    print(f"  Sample amounts: {round_num['invoice_amount'].head(10).tolist()}")

# =============================================================================
# CHECK 11 — GHOST VENDOR INJECTION
# =============================================================================
section("CHECK 11: Ghost Vendor Anomaly")

ghost_vendors = vendors[vendors["is_ghost_vendor"] == True]
ghost_invoices = inv[inv["is_ghost_vendor_inv"] == True]
print(f"  Labelled ghost vendors: {len(ghost_vendors)}")
print(f"  Labelled ghost vendor invoices: {len(ghost_invoices)}")

if len(ghost_invoices) > 0:
    check = ghost_invoices.merge(
        vendors[["vendor_id", "vendor_creation_date"]], on="vendor_id", how="left"
    )
    check["days_to_invoice"] = (
        check["invoice_date"] - check["vendor_creation_date"]
    ).dt.days
    print(f"  Days from vendor creation to invoice — "
          f"min: {check['days_to_invoice'].min()}, "
          f"max: {check['days_to_invoice'].max()} "
          f"(should be roughly 3-20)")
    fast_payment = (check["payment_terms_days"] == 15).all()
    print(f"  All have 15-day fast payment terms: {fast_payment}")

# =============================================================================
# CHECK 12 — BUDGET OVERRUN
# =============================================================================
section("CHECK 12: Budget Table Sanity")

print(f"  Budget rows where actual_spend > approved_budget: "
      f"{(budget['actual_spend'] > budget['approved_budget']).sum()}")
print(f"  (Note: Phase 2 didn't explicitly inject overruns into budget.csv —")
print(f"   actual_spend was generated as 60-95% of committed, which itself")
print(f"   is 30-75% of approved. This means budget.csv currently has ZERO")
print(f"   organic overruns. This is something to fix before Phase 5/7.)")

# =============================================================================
# CHECK 13 — OVERALL ANOMALY FLAG CONSISTENCY
# =============================================================================
section("CHECK 13: Anomaly Flag Consistency")

# Every row with anomaly_flag=1 should have exactly one is_* column True
anomaly_cols = ["is_duplicate", "is_near_duplicate", "is_match_failure",
                 "is_maverick", "is_after_hours", "is_round_number",
                 "is_ghost_vendor_inv"]

flagged = inv[inv["anomaly_flag"] == 1]
flag_sum = flagged[anomaly_cols].sum(axis=1)
print(f"  Anomalous invoices: {len(flagged)}")
print(f"  Invoices with exactly 1 anomaly sub-flag True: {(flag_sum == 1).sum()}")
print(f"  Invoices with 0 sub-flags True (BUG if >0): {(flag_sum == 0).sum()}")
print(f"  Invoices with >1 sub-flags True (overlap, not necessarily bad): "
      f"{(flag_sum > 1).sum()}")

normal = inv[inv["anomaly_flag"] == 0]
normal_flag_sum = normal[anomaly_cols].sum(axis=1)
print(f"\n  Normal invoices with any sub-flag True "
      f"(BUG if >0): {(normal_flag_sum > 0).sum()}")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
print("Review any FAIL or BUG flags above before proceeding to Phase 3.")