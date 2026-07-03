# phase4_eda.py
# Fraud-focused Exploratory Data Analysis on the ERP dataset.
# Every analysis section answers a specific audit question.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings, os
warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs/eda"
DATA_DIR   = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Audit-appropriate colour palette — red signals risk, not cheerful pastels
PALETTE    = {"normal": "#2C7BB6", "anomaly": "#D7191C", "warning": "#FDAE61"}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})

def save_fig(name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
inv = pd.read_csv(f"{DATA_DIR}/invoices.csv",
                  parse_dates=["invoice_date", "due_date", "payment_date"])
po  = pd.read_csv(f"{DATA_DIR}/purchase_orders.csv",
                  parse_dates=["po_date"])
gr  = pd.read_csv(f"{DATA_DIR}/goods_receipts.csv",
                  parse_dates=["gr_date"])
ven = pd.read_csv(f"{DATA_DIR}/vendors.csv",
                  parse_dates=["vendor_creation_date"])
bud = pd.read_csv(f"{DATA_DIR}/budget.csv")

# Derived columns used throughout analysis
inv["invoice_hour"]    = inv["invoice_date"].dt.hour
inv["invoice_dow"]     = inv["invoice_date"].dt.dayofweek  # 0=Mon
inv["invoice_month"]   = inv["invoice_date"].dt.to_period("M")
inv["invoice_year"]    = inv["invoice_date"].dt.year
inv["is_weekend"]      = inv["invoice_dow"] >= 5
inv["is_after_hours"]  = (
    (inv["invoice_hour"] < 8) | (inv["invoice_hour"] >= 19) | inv["is_weekend"]
)
inv["days_to_payment"] = (
    inv["payment_date"] - inv["invoice_date"]
).dt.days

print(f"  Invoices: {len(inv):,}  |  POs: {len(po):,}  "
      f"|  GRs: {len(gr):,}  |  Vendors: {len(ven):,}")

# =============================================================================
# SECTION 1 — TRANSACTION VOLUME AND ANOMALY OVERVIEW
# =============================================================================
print("\n--- Section 1: Transaction Overview ---")

# 1A — Anomaly rate by type
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

anomaly_counts = inv["anomaly_type"].value_counts()
colours = [PALETTE["anomaly"] if t != "Normal" else PALETTE["normal"]
           for t in anomaly_counts.index]
axes[0].barh(anomaly_counts.index, anomaly_counts.values, color=colours)
axes[0].set_xlabel("Number of Transactions")
axes[0].set_title("Transaction Count by Anomaly Type", fontweight="bold")
for i, v in enumerate(anomaly_counts.values):
    axes[0].text(v + 5, i, str(v), va="center")

# 1B — Anomaly vs normal financial exposure
exposure = inv.groupby("anomaly_type")["invoice_amount"].sum() / 1e6
exposure_sorted = exposure.sort_values(ascending=True)
axes[1].barh(
    exposure_sorted.index, exposure_sorted.values,
    color=[PALETTE["anomaly"] if t != "Normal" else PALETTE["normal"]
           for t in exposure_sorted.index]
)
axes[1].set_xlabel("Total Invoice Amount (₹ Millions)")
axes[1].set_title("Financial Exposure by Anomaly Type", fontweight="bold")

plt.suptitle("ERP Anomaly Detection — Dataset Overview", fontsize=14,
             fontweight="bold")
plt.tight_layout()
save_fig("01_anomaly_overview")

# Print summary statistics
total_exposure = inv[inv["anomaly_flag"]==1]["invoice_amount"].sum()
print(f"  Total anomalous financial exposure: ₹{total_exposure:,.0f}")
print(f"  Anomaly rate: {inv['anomaly_flag'].mean()*100:.1f}%")

# =============================================================================
# SECTION 2 — INVOICE AMOUNT DISTRIBUTION
# =============================================================================
print("\n--- Section 2: Amount Distribution Analysis ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2A — Log-scale histogram: normal vs anomalous amounts
normal_amounts  = inv[inv["anomaly_flag"]==0]["invoice_amount"]
anomaly_amounts = inv[inv["anomaly_flag"]==1]["invoice_amount"]

axes[0].hist(np.log10(normal_amounts[normal_amounts > 0] + 1),
             bins=50, alpha=0.6, color=PALETTE["normal"],  label="Normal")
axes[0].hist(np.log10(anomaly_amounts[anomaly_amounts > 0] + 1),
             bins=50, alpha=0.6, color=PALETTE["anomaly"], label="Anomalous")
axes[0].set_xlabel("Log₁₀(Invoice Amount ₹)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Invoice Amount Distribution (Log Scale)", fontweight="bold")
axes[0].legend()

# 2B — Box plot by anomaly type (log scale)
plot_data = inv[inv["invoice_amount"] > 0].copy()
plot_data["log_amount"] = np.log10(plot_data["invoice_amount"])
order = inv["anomaly_type"].value_counts().index.tolist()

sns.boxplot(
    data=plot_data, x="log_amount", y="anomaly_type",
    order=order, ax=axes[1],
    palette=["#D7191C" if t != "Normal" else "#2C7BB6" for t in order]
)
axes[1].set_xlabel("Log₁₀(Invoice Amount ₹)")
axes[1].set_ylabel("")
axes[1].set_title("Amount Distribution by Anomaly Type", fontweight="bold")

plt.tight_layout()
save_fig("02_amount_distribution")

# =============================================================================
# SECTION 3 — BENFORD'S LAW ANALYSIS
# =============================================================================
# Benford's Law: in naturally occurring financial data, the first digit
# of amounts follows a specific distribution (1 appears ~30% of the time,
# 9 appears only ~5%). Fraud often disrupts this because fraudsters tend
# to invent round numbers or cluster around approval thresholds.
# This is a recognised forensic accounting technique used by Big 4 firms.
print("\n--- Section 3: Benford's Law Analysis ---")

def get_first_digit(series: pd.Series) -> pd.Series:
    """Extract the first significant digit from a numeric series."""
    return series[series > 0].astype(str).str.lstrip("0").str[0].astype(int)

# Expected Benford distribution
benford_expected = {d: np.log10(1 + 1/d) for d in range(1, 10)}

def plot_benford(amounts: pd.Series, title: str, ax):
    """Plot observed first digit frequency vs Benford's Law expectation."""
    first_digits = get_first_digit(amounts)
    observed_pct = first_digits.value_counts(normalize=True).sort_index() * 100
    expected_pct = pd.Series(benford_expected) * 100

    x = range(1, 10)
    ax.bar(x, [observed_pct.get(d, 0) for d in x],
           alpha=0.7, color=PALETTE["normal"], label="Observed")
    ax.plot(x, [expected_pct[d] for d in x],
            "ro-", linewidth=2, markersize=6, label="Benford Expected")
    ax.set_xlabel("First Digit")
    ax.set_ylabel("Frequency (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.set_xticks(range(1, 10))

    # Chi-square test: does observed deviate significantly from expected?
    obs_counts = [first_digits.value_counts().get(d, 0) for d in range(1, 10)]
    exp_counts = [benford_expected[d] * len(first_digits) for d in range(1, 10)]
    chi2, p_val = stats.chisquare(obs_counts, exp_counts)
    ax.text(0.98, 0.95, f"χ²={chi2:.1f}, p={p_val:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    return chi2, p_val

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

chi2_normal,  p_normal  = plot_benford(
    inv[inv["anomaly_flag"]==0]["invoice_amount"], "Benford's Law — Normal Invoices",  axes[0])
chi2_anomaly, p_anomaly = plot_benford(
    inv[inv["anomaly_flag"]==1]["invoice_amount"], "Benford's Law — Anomalous Invoices", axes[1])

plt.suptitle("Benford's Law Analysis — First Digit Distribution",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("03_benfords_law")

print(f"  Normal invoices:    χ²={chi2_normal:.1f}, p={p_normal:.4f}")
print(f"  Anomalous invoices: χ²={chi2_anomaly:.1f}, p={p_anomaly:.4f}")
print("  Interpretation: p < 0.05 means the distribution deviates significantly")
print("  from Benford's Law — a red flag in forensic accounting.")

# =============================================================================
# SECTION 4 — TEMPORAL ANALYSIS
# =============================================================================
print("\n--- Section 4: Temporal Pattern Analysis ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4A — Invoices by hour of day
hourly = inv.groupby(["invoice_hour", "anomaly_flag"]).size().unstack(fill_value=0)
hourly.columns = ["Normal", "Anomalous"]
hourly.plot(kind="bar", ax=axes[0,0],
            color=[PALETTE["normal"], PALETTE["anomaly"]], alpha=0.8)
axes[0,0].axvspan(0,   7.5, alpha=0.05, color="red",   label="After hours")
axes[0,0].axvspan(18.5, 23, alpha=0.05, color="red")
axes[0,0].set_xlabel("Hour of Day")
axes[0,0].set_ylabel("Invoice Count")
axes[0,0].set_title("Invoice Volume by Hour — Normal vs Anomalous",
                     fontweight="bold")
axes[0,0].tick_params(axis="x", rotation=0)
axes[0,0].legend()

# 4B — Invoices by day of week
day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow_data   = inv.groupby(["invoice_dow","anomaly_flag"]).size().unstack(fill_value=0)
dow_data.index = [day_labels[i] for i in dow_data.index]
dow_data.columns = ["Normal", "Anomalous"]
dow_data.plot(kind="bar", ax=axes[0,1],
              color=[PALETTE["normal"], PALETTE["anomaly"]], alpha=0.8)
axes[0,1].set_xlabel("Day of Week")
axes[0,1].set_ylabel("Invoice Count")
axes[0,1].set_title("Invoice Volume by Day of Week", fontweight="bold")
axes[0,1].tick_params(axis="x", rotation=0)

# 4C — Monthly invoice volume trend
monthly = inv.groupby([inv["invoice_date"].dt.to_period("M"),
                        "anomaly_flag"]).size().unstack(fill_value=0)
monthly.index = monthly.index.astype(str)
monthly.columns = ["Normal", "Anomalous"]
monthly.plot(ax=axes[1,0], color=[PALETTE["normal"], PALETTE["anomaly"]])
axes[1,0].set_xlabel("Month")
axes[1,0].set_ylabel("Invoice Count")
axes[1,0].set_title("Monthly Invoice Volume Trend", fontweight="bold")
axes[1,0].tick_params(axis="x", rotation=45)

# 4D — Payment speed distribution (days to payment)
paid = inv[inv["days_to_payment"].notna() & (inv["days_to_payment"] > 0)]
axes[1,1].hist(
    paid[paid["anomaly_flag"]==0]["days_to_payment"].clip(0,120),
    bins=40, alpha=0.6, color=PALETTE["normal"],  label="Normal")
axes[1,1].hist(
    paid[paid["anomaly_flag"]==1]["days_to_payment"].clip(0,120),
    bins=40, alpha=0.6, color=PALETTE["anomaly"], label="Anomalous")
axes[1,1].set_xlabel("Days from Invoice to Payment")
axes[1,1].set_ylabel("Frequency")
axes[1,1].set_title("Payment Speed — Normal vs Anomalous", fontweight="bold")
axes[1,1].legend()

plt.suptitle("Temporal Pattern Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("04_temporal_analysis")

# =============================================================================
# SECTION 5 — VENDOR RISK ANALYSIS
# =============================================================================
print("\n--- Section 5: Vendor Risk Analysis ---")

vendor_stats = inv.groupby("vendor_id").agg(
    total_invoiced   = ("invoice_amount", "sum"),
    invoice_count    = ("invoice_number", "count"),
    anomaly_count    = ("anomaly_flag",   "sum"),
    avg_amount       = ("invoice_amount", "mean"),
).reset_index()
vendor_stats["anomaly_rate"] = (
    vendor_stats["anomaly_count"] / vendor_stats["invoice_count"]
)
vendor_stats = vendor_stats.merge(
    ven[["vendor_id","vendor_name","vendor_category","is_ghost_vendor"]], 
    on="vendor_id", how="left"
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 5A — Top 15 vendors by total spend
top_vendors = vendor_stats.nlargest(15, "total_invoiced")
colours_v   = [PALETTE["anomaly"] if g else PALETTE["normal"]
               for g in top_vendors["is_ghost_vendor"]]
axes[0].barh(
    range(len(top_vendors)),
    top_vendors["total_invoiced"] / 1e6,
    color=colours_v
)
axes[0].set_yticks(range(len(top_vendors)))
axes[0].set_yticklabels(
    [n[:30] for n in top_vendors["vendor_name"]], fontsize=8
)
axes[0].set_xlabel("Total Invoiced (₹ Millions)")
axes[0].set_title("Top 15 Vendors by Total Spend\n(Red = Ghost Vendor)",
                   fontweight="bold")

# 5B — Vendor anomaly rate scatter
sc = axes[1].scatter(
    vendor_stats["invoice_count"],
    vendor_stats["anomaly_rate"] * 100,
    c=vendor_stats["total_invoiced"],
    cmap="RdYlGn_r",
    alpha=0.6, s=30
)
axes[1].axhline(y=10, color="red", linestyle="--", alpha=0.5,
                label="10% anomaly rate threshold")
axes[1].set_xlabel("Number of Invoices")
axes[1].set_ylabel("Anomaly Rate (%)")
axes[1].set_title("Vendor Anomaly Rate vs Invoice Volume", fontweight="bold")
axes[1].legend()
plt.colorbar(sc, ax=axes[1], label="Total Spend (₹)")

plt.tight_layout()
save_fig("05_vendor_analysis")

high_risk_vendors = vendor_stats[vendor_stats["anomaly_rate"] > 0.1]
print(f"  Vendors with >10% anomaly rate: {len(high_risk_vendors)}")
print(f"  Top 10 vendor spend concentration: "
      f"{vendor_stats.nlargest(10,'total_invoiced')['total_invoiced'].sum() / vendor_stats['total_invoiced'].sum() * 100:.1f}%")

# =============================================================================
# SECTION 6 — THREE-WAY MATCH ANALYSIS
# =============================================================================
print("\n--- Section 6: Three-Way Match Analysis ---")

# Merge invoices with POs to compute match deviation
inv_po = inv[inv["po_number"].notna()].merge(
    po[["po_number","total_po_value"]], on="po_number", how="left"
)
inv_po["match_deviation_pct"] = (
    (inv_po["invoice_amount"] - inv_po["total_po_value"])
    / inv_po["total_po_value"] * 100
)

# Three-way match pass/fail
inv_po["match_pass"] = inv_po["match_deviation_pct"].abs() <= 5

match_rate = inv_po["match_pass"].mean() * 100
print(f"  Three-way match pass rate: {match_rate:.1f}%")
print(f"  Industry benchmark: >95% for a healthy control environment")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 6A — Match deviation distribution
axes[0].hist(
    inv_po["match_deviation_pct"].clip(-50, 50),
    bins=80, color=PALETTE["normal"], alpha=0.7, edgecolor="white"
)
axes[0].axvline(x=-5, color="red", linestyle="--", label="±5% tolerance")
axes[0].axvline(x= 5, color="red", linestyle="--")
axes[0].set_xlabel("Invoice vs PO Deviation (%)")
axes[0].set_ylabel("Frequency")
axes[0].set_title(f"Three-Way Match Deviation Distribution\n"
                   f"Pass Rate: {match_rate:.1f}%", fontweight="bold")
axes[0].legend()

# 6B — Match failure by vendor category
inv_po_ven = inv_po.merge(
    ven[["vendor_id","vendor_category"]], on="vendor_id", how="left"
)
fail_by_cat = inv_po_ven.groupby("vendor_category")["match_pass"].agg(
    ["sum","count"]
).reset_index()
fail_by_cat["fail_rate"] = (1 - fail_by_cat["sum"]/fail_by_cat["count"]) * 100
fail_by_cat = fail_by_cat.sort_values("fail_rate", ascending=True)

axes[1].barh(
    fail_by_cat["vendor_category"],
    fail_by_cat["fail_rate"],
    color=[PALETTE["anomaly"] if r > 10 else PALETTE["normal"]
           for r in fail_by_cat["fail_rate"]]
)
axes[1].axvline(x=10, color="orange", linestyle="--", label="10% threshold")
axes[1].set_xlabel("Match Failure Rate (%)")
axes[1].set_title("Three-Way Match Failure Rate by Vendor Category",
                   fontweight="bold")
axes[1].legend()

plt.tight_layout()
save_fig("06_three_way_match")

# =============================================================================
# SECTION 7 — BUDGET UTILISATION ANALYSIS
# =============================================================================
print("\n--- Section 7: Budget Utilisation Analysis ---")

fig, ax = plt.subplots(figsize=(12, 6))

bud_sorted = bud.sort_values("budget_utilisation_pct", ascending=True)
colours_b  = [
    PALETTE["anomaly"] if u > 100
    else PALETTE["warning"] if u > 90
    else PALETTE["normal"]
    for u in bud_sorted["budget_utilisation_pct"]
]
bars = ax.barh(
    bud_sorted["cost_centre"].astype(str) + " (" + bud_sorted["budget_period"].astype(str) + ")",
    bud_sorted["budget_utilisation_pct"],
    color=colours_b
)
ax.axvline(x=100, color="red",    linestyle="--", linewidth=2, label="Budget limit (100%)")
ax.axvline(x=90,  color="orange", linestyle="--", linewidth=1, label="Warning threshold (90%)")
ax.set_xlabel("Budget Utilisation (%)")
ax.set_title("Budget Utilisation by Cost Centre\n"
             "Red = Overrun, Orange = At Risk (>90%)", fontweight="bold")
ax.legend()

plt.tight_layout()
save_fig("07_budget_utilisation")

overrun_count  = (bud["budget_utilisation_pct"] > 100).sum()
at_risk_count  = (
    (bud["budget_utilisation_pct"] > 90) &
    (bud["budget_utilisation_pct"] <= 100)
).sum()
print(f"  Cost centres over budget:      {overrun_count}")
print(f"  Cost centres at risk (>90%):   {at_risk_count}")

# =============================================================================
# SECTION 8 — APPROVAL THRESHOLD CLUSTERING
# =============================================================================
# Do invoice amounts cluster just below the approval threshold?
# This is a statistical signal for deliberate threshold avoidance.
print("\n--- Section 8: Approval Threshold Clustering ---")

threshold = 100_000
window    = 0.10   # look at amounts within 10% below threshold

near_threshold = inv[
    (inv["invoice_amount"] >= threshold * (1 - window)) &
    (inv["invoice_amount"] <  threshold)
]
far_below = inv[
    (inv["invoice_amount"] >= threshold * (1 - window * 3)) &
    (inv["invoice_amount"] <  threshold * (1 - window))
]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(inv[(inv["invoice_amount"] < threshold * 1.2) &
             (inv["invoice_amount"] > threshold * 0.5)]["invoice_amount"],
        bins=60, color=PALETTE["normal"], alpha=0.7)
ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2,
           label=f"Approval threshold ₹{threshold:,}")
ax.axvspan(threshold * 0.9, threshold, alpha=0.15, color="red",
           label=f"Suspicious zone (within 10% below threshold)")
ax.set_xlabel("Invoice Amount (₹)")
ax.set_ylabel("Frequency")
ax.set_title("Invoice Amount Clustering Near Approval Threshold\n"
             "High concentration just below threshold = splitting signal",
             fontweight="bold")
ax.legend()
plt.tight_layout()
save_fig("08_threshold_clustering")

ratio = len(near_threshold) / max(len(far_below), 1)
print(f"  Transactions within 10% below threshold: {len(near_threshold)}")
print(f"  Transactions in equivalent band further below: {len(far_below)}")
print(f"  Concentration ratio: {ratio:.2f}x  "
      f"(>1.5x is suspicious)")

print("\n✓ EDA complete. All charts saved to outputs/eda/")
print("\n=== KEY BUSINESS INSIGHTS ===")
print(f"1. Anomaly rate:           {inv['anomaly_flag'].mean()*100:.1f}% of transactions flagged")
print(f"2. Financial exposure:     ₹{inv[inv['anomaly_flag']==1]['invoice_amount'].sum()/1e6:.1f}M at risk")
print(f"3. Three-way match rate:   {match_rate:.1f}% (benchmark: >95%)")
print(f"4. After-hours postings:   {inv['is_after_hours'].sum()} transactions")
print(f"5. Budget overruns:        {overrun_count} cost centres")
print(f"6. Maverick buying:        {inv['is_maverick'].sum()} invoices with no PO")