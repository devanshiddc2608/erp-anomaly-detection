# config.py
# Central configuration file for the ERP Anomaly Detection project.
# All thresholds, parameters, and file paths live here.
# Changing a value here propagates through the entire pipeline.

import os

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "data/raw"          # where generated CSVs are saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset scale ─────────────────────────────────────────────────────────────
N_PURCHASE_ORDERS   = 20_000     # number of PO records to generate
N_VENDORS           = 300        # number of vendors in the master
N_USERS             = 80         # number of system users
N_COST_CENTRES      = 20         # number of cost centres / departments
SIMULATION_YEARS    = 2          # how many years of history to simulate
START_DATE          = "2022-01-01"

# ── Business rules (mirrors real SAP configuration) ───────────────────────────
PO_APPROVAL_THRESHOLD     = 100_000   # INR — POs above this need CFO approval
SPLIT_PO_BUFFER           = 0.05      # split POs land within 5% below threshold
THREE_WAY_MATCH_TOLERANCE = 0.05      # 5% tolerance on three-way match
PAYMENT_TERMS_DAYS        = [30, 45, 60, 90]   # standard vendor payment terms
BUSINESS_HOURS_START      = 8         # 8 AM
BUSINESS_HOURS_END        = 19        # 7 PM
BUSINESS_DAYS             = [0,1,2,3,4]  # Monday=0 … Friday=4

# ── Anomaly injection volumes ─────────────────────────────────────────────────
N_DUPLICATE_INVOICES       = 120   # exact duplicate invoice pairs
N_NEAR_DUPLICATE_INVOICES  = 80    # near-duplicate (small amount tweak)
N_SPLIT_PO_GROUPS          = 60    # groups of 2–3 split POs
N_MATCH_FAILURES           = 200   # three-way match failures
N_BUDGET_OVERRUNS          = 8    # cost centres pushed over budget
N_GHOST_VENDOR_INVOICES    = 50    # invoices from ghost vendors
N_MAVERICK_INVOICES        = 150   # invoices with no PO
N_AFTER_HOURS              = 100   # transactions outside business hours
N_ROUND_NUMBER             = 90    # suspiciously round invoice amounts
N_HIGH_FREQUENCY_VENDOR    = 3     # vendors with abnormally high invoice count

# ── Vendor categories ─────────────────────────────────────────────────────────
VENDOR_CATEGORIES = [
    "IT Services", "Office Supplies", "Facilities Management",
    "Professional Services", "Raw Materials", "Logistics",
    "Marketing", "HR & Recruitment", "Maintenance & Repair", "Utilities"
]

# ── Departments ───────────────────────────────────────────────────────────────
DEPARTMENTS = [
    "Finance", "IT", "Operations", "HR", "Marketing",
    "Procurement", "Legal", "Admin", "R&D", "Sales"
]

# ── User roles ────────────────────────────────────────────────────────────────
USER_ROLES = {
    "Procurement Officer": {"can_create_po": True,  "can_approve_po": False,
                            "can_create_vendor": False, "can_approve_invoice": False},
    "Senior Buyer":        {"can_create_po": True,  "can_approve_po": True,
                            "can_create_vendor": False, "can_approve_invoice": False},
    "AP Clerk":            {"can_create_po": False, "can_approve_po": False,
                            "can_create_vendor": False, "can_approve_invoice": True},
    "AP Manager":          {"can_create_po": False, "can_approve_po": False,
                            "can_create_vendor": False, "can_approve_invoice": True},
    "Finance Manager":     {"can_create_po": False, "can_approve_po": True,
                            "can_create_vendor": False, "can_approve_invoice": True},
    "CFO":                 {"can_create_po": False, "can_approve_po": True,
                            "can_create_vendor": False, "can_approve_invoice": True},
    "Vendor Admin":        {"can_create_po": False, "can_approve_po": False,
                            "can_create_vendor": True,  "can_approve_invoice": False},
    "System Admin":        {"can_create_po": True,  "can_approve_po": True,
                            "can_create_vendor": True,  "can_approve_invoice": True},
}