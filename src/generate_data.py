# generate_data.py
# Generates all six ERP tables with realistic distributions,
# injects labelled anomalies, and saves to CSV.
# Run this file directly: python generate_data.py

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os
import warnings
warnings.filterwarnings("ignore")

from config import *

# Reproducibility — same seed produces same dataset every run
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker("en_IN")   # Indian locale for realistic names/addresses
Faker.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def random_date(start: str, end: str) -> datetime:
    """Return a random datetime between two date strings."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    delta    = (end_dt - start_dt).days
    return start_dt + timedelta(days=random.randint(0, delta))

def business_hours_timestamp(base_date: datetime) -> datetime:
    """Return a timestamp during normal business hours on a weekday."""
    # Shift to nearest weekday if needed
    while base_date.weekday() not in BUSINESS_DAYS:
        base_date += timedelta(days=1)
    hour   = random.randint(BUSINESS_HOURS_START, BUSINESS_HOURS_END - 1)
    minute = random.randint(0, 59)
    return base_date.replace(hour=hour, minute=minute, second=0)

def after_hours_timestamp(base_date: datetime) -> datetime:
    """Return a timestamp outside business hours (night, weekend, or holiday)."""
    choice = random.choice(["night", "weekend"])
    if choice == "night":
        hour = random.choice(
            list(range(0, BUSINESS_HOURS_START)) +
            list(range(BUSINESS_HOURS_END, 24))
        )
        return base_date.replace(hour=hour, minute=random.randint(0, 59))
    else:
        # Push to Saturday or Sunday
        days_to_weekend = (5 - base_date.weekday()) % 7
        if days_to_weekend == 0:
            days_to_weekend = 0
        weekend_date = base_date + timedelta(days=days_to_weekend if days_to_weekend > 0 else 1)
        if weekend_date.weekday() == 0:   # landed on Monday, go back to Sunday
            weekend_date -= timedelta(days=1)
        hour = random.randint(9, 18)
        return weekend_date.replace(hour=hour, minute=random.randint(0, 59))

def invoice_amount_normal(vendor_category: str) -> float:
    """
    Return a realistic invoice amount for a vendor category.
    Different categories have very different typical spend ranges.
    """
    category_ranges = {
        "IT Services":             (50_000,  800_000),
        "Office Supplies":         (5_000,   80_000),
        "Facilities Management":   (30_000,  500_000),
        "Professional Services":   (100_000, 2_000_000),
        "Raw Materials":           (200_000, 5_000_000),
        "Logistics":               (20_000,  300_000),
        "Marketing":               (50_000,  1_000_000),
        "HR & Recruitment":        (30_000,  400_000),
        "Maintenance & Repair":    (10_000,  200_000),
        "Utilities":               (15_000,  150_000),
    }
    low, high = category_ranges.get(vendor_category, (10_000, 500_000))
    # Log-normal distribution — most invoices are small, a few are large
    # This mirrors real procurement spend distributions (Pareto-like)
    mean_log = np.log((low + high) / 2)
    std_log  = 0.6
    amount   = np.random.lognormal(mean=mean_log, sigma=std_log)
    # Clip to realistic range
    return round(float(np.clip(amount, low, high)), 2)

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — USERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_users() -> pd.DataFrame:
    """
    Generates the SAP user master table.
    In SAP this maps to USR02 and AGR_USERS.
    Each user has a role that determines what actions they can perform.
    """
    print("Generating Users table...")
    rows = []
    user_id = 1000

    for role, _ in USER_ROLES.items():
        # Distribute users across roles — more clerks than CFOs
        count_map = {
            "Procurement Officer": 15,
            "Senior Buyer":        8,
            "AP Clerk":            12,
            "AP Manager":          4,
            "Finance Manager":     5,
            "CFO":                 2,
            "Vendor Admin":        6,
            "System Admin":        3,
        }
        n = count_map.get(role, 5)
        for _ in range(n):
            dept = random.choice(DEPARTMENTS)
            rows.append({
                "user_id":         f"USR{user_id:04d}",
                "user_name":       fake.name(),
                "role":            role,
                "department":      dept,
                # Access level: 1=read only, 2=create, 3=approve, 4=admin
                "access_level":    4 if role == "System Admin" else
                                   3 if "Manager" in role or role == "CFO" else
                                   2 if "Officer" in role or "Buyer" in role
                                     or "Clerk" in role or "Admin" in role
                                   else 1,
                "last_login_date": random_date(START_DATE, "2024-01-01"),
                "is_active":       random.choice([True, True, True, False]),
            })
            user_id += 1

    df = pd.DataFrame(rows)
    print(f"  → {len(df)} user records generated.")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2 — VENDORS
# ─────────────────────────────────────────────────────────────────────────────

def generate_vendors(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the SAP vendor master.
    In SAP this maps to LFA1 (general data) and LFB1 (company code data).
    Includes a small number of ghost vendor records injected with anomaly labels.
    """
    print("Generating Vendors table...")

    # Users who can create vendors
    vendor_admins = users_df[
        users_df["role"].isin(["Vendor Admin", "System Admin"])
    ]["user_id"].tolist()

    end_date = "2023-12-31"
    rows = []

    for i in range(1, N_VENDORS + 1):
        creation_date = random_date(START_DATE, "2022-06-01")  # most vendors are established
        category      = random.choice(VENDOR_CATEGORIES)
        rows.append({
            "vendor_id":           f"VEND{i:04d}",
            "vendor_name":         fake.company() + random.choice(
                                       [" Ltd", " Pvt Ltd", " Solutions",
                                        " Services", " Corp", " Enterprises"]),
            "vendor_creation_date": creation_date,
            "vendor_category":     category,
            "bank_account_number": fake.bban(),
            "registered_address":  fake.address().replace("\n", ", "),
            "contact_email":       fake.company_email(),
            "payment_method":      random.choice(["NEFT", "RTGS", "Cheque", "IMPS"]),
            "is_active":           True,
            "created_by_user":     random.choice(vendor_admins),
            "is_ghost_vendor":     False,   # anomaly label
            "anomaly_flag":        0,
        })

    # ── Inject Ghost Vendor anomaly ───────────────────────────────────────────
    # Ghost vendors are created just before they are first used.
    # We mark them here; they get linked to invoices in the invoice generation step.
    ghost_count = 15
    ghost_vendor_admins_with_ap = users_df[
        users_df["role"] == "System Admin"   # SoD violation: same user creates and pays
    ]["user_id"].tolist()

    for j in range(N_VENDORS + 1, N_VENDORS + ghost_count + 1):
        # Ghost vendor created very recently — a red flag
        creation_date = random_date("2023-10-01", "2023-12-15")
        rows.append({
            "vendor_id":           f"VEND{j:04d}",
            "vendor_name":         fake.company() + " Consulting",
            "vendor_creation_date": creation_date,
            "vendor_category":     "Professional Services",
            # Ghost vendor bank account — residential address pattern
            "bank_account_number": fake.bban(),
            "registered_address":  fake.street_address() + ", " + fake.city(),
            "contact_email":       fake.free_email(),    # Gmail/Yahoo not corporate
            "payment_method":      "NEFT",
            "is_active":           True,
            "created_by_user":     random.choice(
                                       ghost_vendor_admins_with_ap
                                       if ghost_vendor_admins_with_ap
                                       else vendor_admins),
            "is_ghost_vendor":     True,
            "anomaly_flag":        1,   # 1 = anomalous
        })

    df = pd.DataFrame(rows)
    print(f"  → {len(df)} vendor records generated ({ghost_count} ghost vendors injected).")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — BUDGET
# ─────────────────────────────────────────────────────────────────────────────

# PATCH for generate_data.py
# Replace the generate_budget() function with this version.
# Adds genuine overrun injection for N_BUDGET_OVERRUNS cost-centre/period rows.

def generate_budget() -> pd.DataFrame:
    """
    Generates budget allocations by cost centre and period.
    In SAP this maps to CO module tables — BPGE (overall budget) and BPJA (annual).
    A subset of rows are deliberately pushed over budget to create the
    Budget Overrun anomaly with a ground-truth label.
    """
    print("Generating Budget table...")
    rows = []
    cost_centres = [f"CC{i:03d}" for i in range(1, N_COST_CENTRES + 1)]

    # Build the full list of (cost_centre, year) combinations first,
    # then randomly select which ones become overruns.
    all_periods = [(cc, year) for year in [2022, 2023] for cc in cost_centres]
    overrun_periods = set(
        random.sample(all_periods, min(N_BUDGET_OVERRUNS, len(all_periods)))
    )

    for year in [2022, 2023]:
        for idx, cc in enumerate(cost_centres):
            dept = DEPARTMENTS[idx % len(DEPARTMENTS)]
            if dept in ["IT", "R&D", "Operations"]:
                approved = random.uniform(5_000_000, 20_000_000)
            elif dept in ["Marketing", "HR"]:
                approved = random.uniform(2_000_000, 8_000_000)
            else:
                approved = random.uniform(1_000_000, 5_000_000)

            is_overrun = (cc, year) in overrun_periods

            if is_overrun:
                # Push committed and actual spend deliberately above approved budget.
                # Overrun magnitude: 5%-30% above approved — realistic range,
                # large enough to be a genuine audit finding, not noise.
                committed = approved * random.uniform(1.05, 1.35)
                actual    = committed * random.uniform(0.92, 1.0)
            else:
                # Normal case — comfortably within budget
                committed = approved * random.uniform(0.3, 0.75)
                actual    = committed * random.uniform(0.6, 0.95)

            rows.append({
                "cost_centre":            cc,
                "department":             dept,
                "budget_period":          f"{year}",
                "approved_budget":        round(approved, 2),
                "committed_amount":       round(committed, 2),
                "actual_spend":           round(actual, 2),
                "remaining_budget":       round(approved - actual, 2),
                "budget_utilisation_pct": round((actual / approved) * 100, 2),
                "is_overrun":             actual > approved,   # ← now reflects reality
                "anomaly_flag":           1 if actual > approved else 0,
            })

    df = pd.DataFrame(rows)
    n_overrun = df["is_overrun"].sum()
    print(f"  → {len(df)} budget records generated ({n_overrun} overruns injected).")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4 — PURCHASE ORDERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_purchase_orders(
    vendors_df: pd.DataFrame,
    users_df:   pd.DataFrame,
    budget_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Generates PO records — the formal commitment to purchase from a vendor.
    In SAP: EKKO (header) + EKPO (line items), combined here for simplicity.
    Normal POs, then anomalous split POs are injected at the end.
    """
    print("Generating Purchase Orders table...")

    # Only use legitimate (non-ghost) vendors for normal POs
    legit_vendors = vendors_df[~vendors_df["is_ghost_vendor"]].copy()

    # Users who can create POs
    po_creators  = users_df[
        users_df["role"].isin(["Procurement Officer", "Senior Buyer", "System Admin"])
    ]["user_id"].tolist()
    # Users who can approve POs (above threshold)
    po_approvers = users_df[
        users_df["role"].isin(["Senior Buyer", "Finance Manager", "CFO", "System Admin"])
    ]["user_id"].tolist()

    cost_centres = budget_df["cost_centre"].unique().tolist()
    plant_codes  = ["PLANT01", "PLANT02", "PLANT03", "PLANT04"]
    end_date     = "2023-12-31"
    rows         = []

    for i in range(1, N_PURCHASE_ORDERS + 1):
        vendor   = legit_vendors.sample(1).iloc[0]
        category = vendor["vendor_category"]
        amount   = invoice_amount_normal(category)
        po_date  = random_date(START_DATE, end_date)
        po_ts    = business_hours_timestamp(po_date)
        cc       = random.choice(cost_centres)

        # Approval logic: above threshold needs a senior approver
        needs_senior = amount > PO_APPROVAL_THRESHOLD
        approver     = random.choice(po_approvers) if needs_senior \
                       else random.choice(po_creators + po_approvers)

        rows.append({
            "po_number":        f"PO{i:06d}",
            "po_date":          po_ts,
            "vendor_id":        vendor["vendor_id"],
            "vendor_name":      vendor["vendor_name"],
            "item_description": f"{category} services/goods — {fake.bs()}",
            "quantity_ordered":  random.randint(1, 100),
            "unit_price":        round(amount / random.randint(1, 100), 2),
            "total_po_value":    round(amount, 2),
            "department":        random.choice(DEPARTMENTS),
            "cost_centre":       cc,
            "budget_code":       f"BUD{random.randint(100, 999)}",
            "approval_status":   random.choice(
                                     ["Approved", "Approved", "Approved", "Pending"]
                                 ),
            "approver_id":       approver,
            "plant_location":    random.choice(plant_codes),
            "is_split_po":       False,   # anomaly label
            "anomaly_flag":      0,
        })

    # ── Inject Split PO anomaly ───────────────────────────────────────────────
    # Each group: 2–3 POs to same vendor in same month, each just below threshold.
    # Together they exceed the threshold, bypassing CFO approval.
    split_po_start_idx = N_PURCHASE_ORDERS + 1

    for grp in range(N_SPLIT_PO_GROUPS):
        vendor      = legit_vendors.sample(1).iloc[0]
        category    = vendor["vendor_category"]
        n_splits    = random.choice([2, 3])
        base_date   = random_date(START_DATE, "2023-11-30")
        # Each split PO is just below the threshold
        split_amount = PO_APPROVAL_THRESHOLD * (1 - random.uniform(0.01, SPLIT_PO_BUFFER))
        cc           = random.choice(cost_centres)

        for k in range(n_splits):
            po_date = base_date + timedelta(days=random.randint(0, 15))
            po_ts   = business_hours_timestamp(po_date)
            rows.append({
                "po_number":        f"PO{split_po_start_idx:06d}",
                "po_date":          po_ts,
                "vendor_id":        vendor["vendor_id"],
                "vendor_name":      vendor["vendor_name"],
                "item_description": f"{category} — split order part {k+1}",
                "quantity_ordered":  random.randint(1, 20),
                "unit_price":        round(split_amount / random.randint(1, 20), 2),
                "total_po_value":    round(split_amount * random.uniform(0.90, 0.99), 2),
                "department":        random.choice(DEPARTMENTS),
                "cost_centre":       cc,
                "budget_code":       f"BUD{random.randint(100, 999)}",
                "approval_status":   "Approved",
                "approver_id":       random.choice(po_creators),  # lower-level approval
                "plant_location":    random.choice(plant_codes),
                "is_split_po":       True,
                "anomaly_flag":      1,
            })
            split_po_start_idx += 1

    df = pd.DataFrame(rows)
    df["po_date"] = pd.to_datetime(df["po_date"])
    print(f"  → {len(df)} PO records generated "
          f"({N_SPLIT_PO_GROUPS} split PO groups injected).")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 5 — GOODS RECEIPTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_goods_receipts(po_df: pd.DataFrame,
                             users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates Goods Receipt records — physical confirmation of delivery.
    In SAP: MSEG + MKPF, posted via MIGO transaction.
    Most approved POs get a GR. Some POs intentionally get no GR
    (to simulate invoices arriving with no delivery confirmation).
    """
    print("Generating Goods Receipts table...")

    # Only approved POs get a GR
    approved_pos = po_df[po_df["approval_status"] == "Approved"].copy()

    # Warehouse users who post GRs
    receivers = users_df[
        users_df["role"].isin(["Procurement Officer", "AP Clerk", "System Admin"])
    ]["user_id"].tolist()

    rows = []
    gr_idx = 1

    for _, po in approved_pos.iterrows():
        # 85% of POs get a GR — 15% are missing (will create match failures)
        if random.random() > 0.85:
            continue

        # GR date is after PO date (typically 3-30 days later for delivery)
        po_date  = pd.to_datetime(po["po_date"])
        gr_date  = po_date + timedelta(days=random.randint(3, 30))
        gr_ts    = business_hours_timestamp(gr_date)

        # Received quantity — usually matches ordered, sometimes partial
        qty_ordered  = po["quantity_ordered"]
        qty_received = qty_ordered if random.random() > 0.1 \
                       else random.randint(int(qty_ordered * 0.5), qty_ordered)

        rows.append({
            "gr_number":          f"GR{gr_idx:06d}",
            "gr_date":            gr_ts,
            "po_number":          po["po_number"],
            "vendor_id":          po["vendor_id"],
            "quantity_received":  qty_received,
            # Unit price at receipt matches PO (small variance for FX/rounding)
            "actual_unit_price":  round(po["unit_price"] * random.uniform(0.99, 1.01), 2),
            "receiving_location": po["plant_location"],
            "received_by":        random.choice(receivers),
            "anomaly_flag":       0,
        })
        gr_idx += 1

    df = pd.DataFrame(rows)
    df["gr_date"] = pd.to_datetime(df["gr_date"])
    print(f"  → {len(df)} GR records generated.")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 6 — INVOICES
# ─────────────────────────────────────────────────────────────────────────────

def generate_invoices(
    po_df:      pd.DataFrame,
    gr_df:      pd.DataFrame,
    vendors_df: pd.DataFrame,
    users_df:   pd.DataFrame,
    budget_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    The most important table — mirrors SAP AP invoice posting (MIRO transaction).
    Contains normal invoices plus all financial anomaly types injected with labels.
    """
    print("Generating Invoices table...")

    ap_processors = users_df[
        users_df["role"].isin(["AP Clerk", "AP Manager", "System Admin"])
    ]["user_id"].tolist()
    ap_approvers  = users_df[
        users_df["role"].isin(["AP Manager", "Finance Manager", "CFO", "System Admin"])
    ]["user_id"].tolist()

    legit_vendors  = vendors_df[~vendors_df["is_ghost_vendor"]].copy()
    ghost_vendors  = vendors_df[ vendors_df["is_ghost_vendor"]].copy()

    end_date = "2023-12-31"
    rows     = []
    inv_idx  = 1

    # Build a lookup: PO number → GR record (for three-way match)
    gr_by_po = gr_df.set_index("po_number").to_dict("index")

    # ── Normal invoices (matched to approved POs with GRs) ────────────────────
    approved_pos_with_gr = po_df[
        (po_df["approval_status"] == "Approved") &
        (po_df["po_number"].isin(gr_df["po_number"]))
    ].copy()

    for _, po in approved_pos_with_gr.iterrows():
        gr       = gr_by_po.get(po["po_number"], {})
        po_date  = pd.to_datetime(po["po_date"])
        # Invoice arrives 5-45 days after GR
        inv_date = po_date + timedelta(days=random.randint(8, 50))
        inv_ts   = business_hours_timestamp(inv_date)

        # Normal invoice amount ≈ PO value (within 3% tolerance)
        inv_amount = round(po["total_po_value"] * random.uniform(0.97, 1.03), 2)
        tax_rate   = 0.18   # 18% GST (Indian tax context)
        tax_amount = round(inv_amount * tax_rate, 2)
        total      = round(inv_amount + tax_amount, 2)

        terms_days = random.choice(PAYMENT_TERMS_DAYS)
        due_date   = inv_date + timedelta(days=terms_days)
        paid       = random.random() > 0.05   # 95% of normal invoices get paid
        pay_date   = due_date + timedelta(days=random.randint(-5, 10)) if paid else None
        pay_amount = total if paid else 0.0

        rows.append({
            "invoice_number":        f"INV{inv_idx:07d}",
            "invoice_date":          inv_ts,
            "vendor_invoice_number": f"VINV-{random.randint(10000,99999)}",
            "vendor_id":             po["vendor_id"],
            "po_number":             po["po_number"],
            "gr_number":             gr.get("gr_number", None),
            "invoice_amount":        inv_amount,
            "tax_amount":            tax_amount,
            "total_amount":          total,
            "payment_terms_days":    terms_days,
            "due_date":              due_date,
            "payment_date":          pay_date,
            "payment_amount":        pay_amount,
            "invoice_status":        "Paid" if paid else "Open",
            "processed_by":          random.choice(ap_processors),
            "approved_by":           random.choice(ap_approvers),
            "cost_centre":           po["cost_centre"],
            # Anomaly label columns
            "is_duplicate":          False,
            "is_near_duplicate":     False,
            "is_match_failure":      False,
            "is_maverick":           False,
            "is_after_hours":        False,
            "is_round_number":       False,
            "is_ghost_vendor_inv":   False,
            "anomaly_flag":          0,
            "anomaly_type":          "Normal",
        })
        inv_idx += 1

    # ─────────────────────────────────────────────────────────────────────────
    # ANOMALY INJECTION BLOCK
    # Each anomaly type is clearly commented with what it looks like in the data
    # ─────────────────────────────────────────────────────────────────────────

    # ── 1. Exact Duplicate Invoices ───────────────────────────────────────────
    # What it looks like: two rows with same vendor_id, same invoice_amount,
    # same vendor_invoice_number (or very similar), within 30 days of each other.
    print("  Injecting duplicate invoices...")
    # Sample base invoices to duplicate
    base_invoices = [r for r in rows if r["anomaly_flag"] == 0]
    sampled_bases = random.sample(base_invoices, min(N_DUPLICATE_INVOICES,
                                                      len(base_invoices)))
    for base in sampled_bases:
        dup = base.copy()
        dup["invoice_number"]        = f"INV{inv_idx:07d}"
        dup["invoice_date"]          = pd.to_datetime(base["invoice_date"]) \
                                       + timedelta(days=random.randint(1, 25))
        # Same vendor invoice number — the primary duplicate signal
        # (vendor_invoice_number stays the same as the original)
        dup["is_duplicate"]          = True
        dup["anomaly_flag"]          = 1
        dup["anomaly_type"]          = "Duplicate Invoice"
        rows.append(dup)
        inv_idx += 1

    # ── 2. Near-Duplicate Invoices ────────────────────────────────────────────
    # What it looks like: same vendor, almost same amount (within 2%), different
    # invoice number. Designed to evade simple exact-match duplicate detection.
    print("  Injecting near-duplicate invoices...")
    sampled_near = random.sample(base_invoices, min(N_NEAR_DUPLICATE_INVOICES,
                                                     len(base_invoices)))
    for base in sampled_near:
        dup = base.copy()
        dup["invoice_number"]        = f"INV{inv_idx:07d}"
        dup["vendor_invoice_number"] = f"VINV-{random.randint(10000,99999)}"
        # Amount tweaked by 0.5–2% — evades exact duplicate check
        tweak = random.uniform(0.005, 0.02) * random.choice([-1, 1])
        dup["invoice_amount"]        = round(base["invoice_amount"] * (1 + tweak), 2)
        dup["total_amount"]          = round(dup["invoice_amount"] * 1.18, 2)
        dup["invoice_date"]          = pd.to_datetime(base["invoice_date"]) \
                                       + timedelta(days=random.randint(2, 20))
        dup["is_near_duplicate"]     = True
        dup["anomaly_flag"]          = 1
        dup["anomaly_type"]          = "Near-Duplicate Invoice"
        rows.append(dup)
        inv_idx += 1

    # ── 3. Three-Way Match Failures ───────────────────────────────────────────
    # What it looks like: invoice_amount differs from PO total_po_value by
    # more than the 5% tolerance. Can be over or under.
    print("  Injecting three-way match failures...")
    for _ in range(N_MATCH_FAILURES):
        po     = po_df[po_df["approval_status"] == "Approved"].sample(1).iloc[0]
        gr     = gr_by_po.get(po["po_number"], {})
        vendor = vendors_df[vendors_df["vendor_id"] == po["vendor_id"]].iloc[0]

        inv_date = pd.to_datetime(po["po_date"]) + timedelta(days=random.randint(10, 60))
        inv_ts   = business_hours_timestamp(inv_date)

        # Invoice amount is outside 5% tolerance — either inflated or deflated
        deviation = random.uniform(0.06, 0.35) * random.choice([1, -1])
        inv_amount = round(po["total_po_value"] * (1 + deviation), 2)
        tax_amount = round(inv_amount * 0.18, 2)

        rows.append({
            "invoice_number":        f"INV{inv_idx:07d}",
            "invoice_date":          inv_ts,
            "vendor_invoice_number": f"VINV-{random.randint(10000,99999)}",
            "vendor_id":             po["vendor_id"],
            "po_number":             po["po_number"],
            "gr_number":             gr.get("gr_number", None),
            "invoice_amount":        inv_amount,
            "tax_amount":            tax_amount,
            "total_amount":          round(inv_amount + tax_amount, 2),
            "payment_terms_days":    random.choice(PAYMENT_TERMS_DAYS),
            "due_date":              inv_date + timedelta(days=30),
            "payment_date":          None,
            "payment_amount":        0.0,
            "invoice_status":        "Blocked",   # SAP blocks mismatched invoices
            "processed_by":          random.choice(ap_processors),
            "approved_by":           None,
            "cost_centre":           po["cost_centre"],
            "is_duplicate":          False,
            "is_near_duplicate":     False,
            "is_match_failure":      True,
            "is_maverick":           False,
            "is_after_hours":        False,
            "is_round_number":       False,
            "is_ghost_vendor_inv":   False,
            "anomaly_flag":          1,
            "anomaly_type":          "Three-Way Match Failure",
        })
        inv_idx += 1

    # ── 4. Maverick Buying (No PO Invoices) ───────────────────────────────────
    # What it looks like: invoice with no po_number reference.
    # Vendor was contacted directly, bypassing procurement process.
    print("  Injecting maverick buying invoices...")
    for _ in range(N_MAVERICK_INVOICES):
        vendor   = legit_vendors.sample(1).iloc[0]
        inv_date = random_date(START_DATE, end_date)
        inv_ts   = business_hours_timestamp(inv_date)
        amount   = invoice_amount_normal(vendor["vendor_category"])
        tax      = round(amount * 0.18, 2)

        rows.append({
            "invoice_number":        f"INV{inv_idx:07d}",
            "invoice_date":          inv_ts,
            "vendor_invoice_number": f"VINV-{random.randint(10000,99999)}",
            "vendor_id":             vendor["vendor_id"],
            "po_number":             None,   # ← key signal: no PO reference
            "gr_number":             None,   # ← no GR either
            "invoice_amount":        round(amount, 2),
            "tax_amount":            tax,
            "total_amount":          round(amount + tax, 2),
            "payment_terms_days":    random.choice(PAYMENT_TERMS_DAYS),
            "due_date":              inv_date + timedelta(days=30),
            "payment_date":          inv_date + timedelta(days=random.randint(20, 40)),
            "payment_amount":        round(amount + tax, 2),
            "invoice_status":        "Paid",
            "processed_by":          random.choice(ap_processors),
            "approved_by":           random.choice(ap_approvers),
            "cost_centre":           random.choice(
                                         budget_df["cost_centre"].tolist()),
            "is_duplicate":          False,
            "is_near_duplicate":     False,
            "is_match_failure":      False,
            "is_maverick":           True,
            "is_after_hours":        False,
            "is_round_number":       False,
            "is_ghost_vendor_inv":   False,
            "anomaly_flag":          1,
            "anomaly_type":          "Maverick Buying",
        })
        inv_idx += 1

    # ── 5. After-Hours Transactions ───────────────────────────────────────────
    # What it looks like: invoice_date timestamp is outside 8AM-7PM on a weekday,
    # or falls on a weekend. Legitimate invoices are almost always posted during
    # business hours by AP staff.
    print("  Injecting after-hours transactions...")
    for _ in range(N_AFTER_HOURS):
        po     = po_df[po_df["approval_status"] == "Approved"].sample(1).iloc[0]
        gr     = gr_by_po.get(po["po_number"], {})
        amount = invoice_amount_normal(
            vendors_df[vendors_df["vendor_id"] == po["vendor_id"]
                      ].iloc[0]["vendor_category"])

        base_date = random_date(START_DATE, end_date)
        inv_ts    = after_hours_timestamp(base_date)  # ← after hours
        tax       = round(amount * 0.18, 2)

        rows.append({
            "invoice_number":        f"INV{inv_idx:07d}",
            "invoice_date":          inv_ts,
            "vendor_invoice_number": f"VINV-{random.randint(10000,99999)}",
            "vendor_id":             po["vendor_id"],
            "po_number":             po["po_number"],
            "gr_number":             gr.get("gr_number", None),
            "invoice_amount":        round(amount, 2),
            "tax_amount":            tax,
            "total_amount":          round(amount + tax, 2),
            "payment_terms_days":    random.choice(PAYMENT_TERMS_DAYS),
            "due_date":              base_date + timedelta(days=30),
            "payment_date":          base_date + timedelta(days=random.randint(25, 45)),
            "payment_amount":        round(amount + tax, 2),
            "invoice_status":        "Paid",
            "processed_by":          random.choice(ap_processors),
            "approved_by":           random.choice(ap_approvers),
            "cost_centre":           po["cost_centre"],
            "is_duplicate":          False,
            "is_near_duplicate":     False,
            "is_match_failure":      False,
            "is_maverick":           False,
            "is_after_hours":        True,
            "is_round_number":       False,
            "is_ghost_vendor_inv":   False,
            "anomaly_flag":          1,
            "anomaly_type":          "After-Hours Transaction",
        })
        inv_idx += 1

    # ── 6. Round Number Invoices ──────────────────────────────────────────────
    # What it looks like: invoice_amount is a suspiciously round number —
    # exactly 50,000 or 100,000 or 500,000. Real invoices almost never end
    # in exactly .00 at large magnitudes. Benford's Law analysis also flags these.
    print("  Injecting round number invoices...")
    round_amounts = [
        10_000, 20_000, 25_000, 50_000, 75_000,
        100_000, 150_000, 200_000, 250_000, 500_000,
        1_000_000, 2_000_000, 5_000_000
    ]
    for _ in range(N_ROUND_NUMBER):
        vendor   = legit_vendors.sample(1).iloc[0]
        inv_date = random_date(START_DATE, end_date)
        inv_ts   = business_hours_timestamp(inv_date)
        amount   = float(random.choice(round_amounts))
        tax      = round(amount * 0.18, 2)

        rows.append({
            "invoice_number":        f"INV{inv_idx:07d}",
            "invoice_date":          inv_ts,
            "vendor_invoice_number": f"VINV-{random.randint(10000,99999)}",
            "vendor_id":             vendor["vendor_id"],
            "po_number":             None,
            "gr_number":             None,
            "invoice_amount":        amount,
            "tax_amount":            tax,
            "total_amount":          round(amount + tax, 2),
            "payment_terms_days":    random.choice(PAYMENT_TERMS_DAYS),
            "due_date":              inv_date + timedelta(days=30),
            "payment_date":          inv_date + timedelta(days=random.randint(20, 40)),
            "payment_amount":        round(amount + tax, 2),
            "invoice_status":        "Paid",
            "processed_by":          random.choice(ap_processors),
            "approved_by":           random.choice(ap_approvers),
            "cost_centre":           random.choice(budget_df["cost_centre"].tolist()),
            "is_duplicate":          False,
            "is_near_duplicate":     False,
            "is_match_failure":      False,
            "is_maverick":           False,
            "is_after_hours":        False,
            "is_round_number":       True,
            "is_ghost_vendor_inv":   False,
            "anomaly_flag":          1,
            "anomaly_type":          "Round Number",
        })
        inv_idx += 1

    # ── 7. Ghost Vendor Invoices ──────────────────────────────────────────────
    # What it looks like: invoice from a vendor created very recently
    # (within 30 days), large amount, paid quickly.
    print("  Injecting ghost vendor invoices...")
    for _ in range(N_GHOST_VENDOR_INVOICES):
        if len(ghost_vendors) == 0:
            break
        vendor   = ghost_vendors.sample(1).iloc[0]
        inv_date = pd.to_datetime(vendor["vendor_creation_date"]) \
                   + timedelta(days=random.randint(3, 20))  # ← created and used fast
        inv_ts   = business_hours_timestamp(inv_date)
        amount   = random.uniform(200_000, 2_000_000)
        tax      = round(amount * 0.18, 2)

        rows.append({
            "invoice_number":        f"INV{inv_idx:07d}",
            "invoice_date":          inv_ts,
            "vendor_invoice_number": f"VINV-{random.randint(10000,99999)}",
            "vendor_id":             vendor["vendor_id"],
            "po_number":             None,
            "gr_number":             None,
            "invoice_amount":        round(amount, 2),
            "tax_amount":            tax,
            "total_amount":          round(amount + tax, 2),
            "payment_terms_days":    15,   # unusually short — fast payment
            "due_date":              inv_date + timedelta(days=15),
            "payment_date":          inv_date + timedelta(days=random.randint(5, 15)),
            "payment_amount":        round(amount + tax, 2),
            "invoice_status":        "Paid",
            "processed_by":          random.choice(ap_processors),
            "approved_by":           random.choice(ap_approvers),
            "cost_centre":           random.choice(budget_df["cost_centre"].tolist()),
            "is_duplicate":          False,
            "is_near_duplicate":     False,
            "is_match_failure":      False,
            "is_maverick":           False,
            "is_after_hours":        False,
            "is_round_number":       False,
            "is_ghost_vendor_inv":   True,
            "anomaly_flag":          1,
            "anomaly_type":          "Ghost Vendor Invoice",
        })
        inv_idx += 1

    # ─────────────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["due_date"]     = pd.to_datetime(df["due_date"])
    df["payment_date"] = pd.to_datetime(df["payment_date"])

    normal_count   = (df["anomaly_flag"] == 0).sum()
    anomaly_count  = (df["anomaly_flag"] == 1).sum()
    print(f"  → {len(df)} invoice records generated.")
    print(f"     Normal: {normal_count} | Anomalous: {anomaly_count}")
    print(f"     Anomaly rate: {anomaly_count/len(df)*100:.1f}%")
    print(f"\n  Anomaly breakdown:")
    print(df[df["anomaly_flag"]==1]["anomaly_type"].value_counts().to_string())
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SAVE TO CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_all(users_df, vendors_df, budget_df,
             po_df, gr_df, invoices_df) -> None:
    """Save all six tables as CSV files in the output directory."""
    tables = {
        "users":           users_df,
        "vendors":         vendors_df,
        "budget":          budget_df,
        "purchase_orders": po_df,
        "goods_receipts":  gr_df,
        "invoices":        invoices_df,
    }
    for name, df in tables.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {name}.csv — {len(df):,} rows — {path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ERP Anomaly Detection — Data Generation Pipeline")
    print("=" * 60)

    users_df    = generate_users()
    vendors_df  = generate_vendors(users_df)
    budget_df   = generate_budget()
    po_df       = generate_purchase_orders(vendors_df, users_df, budget_df)
    gr_df       = generate_goods_receipts(po_df, users_df)
    invoices_df = generate_invoices(po_df, gr_df, vendors_df, users_df, budget_df)

    print("\nSaving all tables to CSV...")
    save_all(users_df, vendors_df, budget_df, po_df, gr_df, invoices_df)

    # ── Quick summary statistics ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    total = (len(users_df) + len(vendors_df) + len(budget_df)
             + len(po_df) + len(gr_df) + len(invoices_df))
    print(f"Total records across all tables: {total:,}")
    print(f"  Users:           {len(users_df):>7,}")
    print(f"  Vendors:         {len(vendors_df):>7,}  "
          f"(incl. {vendors_df['is_ghost_vendor'].sum()} ghost vendors)")
    print(f"  Budget lines:    {len(budget_df):>7,}")
    print(f"  Purchase Orders: {len(po_df):>7,}  "
          f"(incl. {po_df['is_split_po'].sum()} split POs)")
    print(f"  Goods Receipts:  {len(gr_df):>7,}")
    print(f"  Invoices:        {len(invoices_df):>7,}  "
          f"(incl. {invoices_df['anomaly_flag'].sum()} anomalous)")
    print("\nData generation complete.")

if __name__ == "__main__":
    main()