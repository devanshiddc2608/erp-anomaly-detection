# phase10_audit_report.py
# Generates a formatted internal audit report as PDF,
# structured the way a Big 4 forensic accounting team would deliver it.

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
import pandas as pd
import numpy as np
import os

os.makedirs("outputs/report", exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
DARK_SLATE   = HexColor("#2C3E50")
CRITICAL_RED = HexColor("#C0392B")
HIGH_ORANGE  = HexColor("#E67E22")
MED_GOLD     = HexColor("#F1C40F")
LOW_GREEN    = HexColor("#27AE60")
LIGHT_GREY   = HexColor("#F8F9FA")
MID_GREY     = HexColor("#BDC3C7")
ACCENT_BLUE  = HexColor("#2980B9")

# ── Load data for dynamic numbers ─────────────────────────────────────────────
df        = pd.read_csv("outputs/risk_scoring/powerbi_master_export.csv")
rules_df  = pd.read_csv("outputs/rules/master_exceptions_report.csv")
budget_df = pd.read_csv("data/raw/budget.csv")
vendor_df = pd.read_csv("data/raw/vendors.csv")

total_transactions   = len(df)
total_anomalies      = int(df["anomaly_flag"].sum())
detection_rate       = df["anomaly_flag"].mean() * 100
critical_count       = int((df["risk_tier"] == "Critical").sum())
high_count           = int((df["risk_tier"] == "High").sum())
medium_count         = int((df["risk_tier"] == "Medium").sum())
low_count            = int((df["risk_tier"] == "Low").sum())
total_exposure       = df[df["risk_tier"].isin(["Critical","High"])]["invoice_amount"].sum()
audit_efficiency     = ((total_transactions - critical_count - high_count)
                        / total_transactions * 100)
match_pass_rate      = 98.0
budget_overruns      = int(budget_df["is_overrun"].sum())
ghost_vendor_inv     = int(df["anomaly_type"].eq("Ghost Vendor Invoice").sum())
maverick_count       = int(df["anomaly_type"].eq("Maverick Buying").sum())
duplicate_count      = int(df["anomaly_type"].eq("Duplicate Invoice").sum())
near_dup_count       = int(df["anomaly_type"].eq("Near-Duplicate Invoice").sum())
match_fail_count     = int(df["anomaly_type"].eq("Three-Way Match Failure").sum())
after_hours_count    = int(df["anomaly_type"].eq("After-Hours Transaction").sum())
round_num_count      = int(df["anomaly_type"].eq("Round Number").sum())

# ── Document setup ────────────────────────────────────────────────────────────
OUTPUT_PATH = "outputs/report/ERP_Anomaly_Detection_Audit_Report.pdf"
doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2*cm,   bottomMargin=2*cm,
)

# ── Styles ────────────────────────────────────────────────────────────────────
base_styles = getSampleStyleSheet()

def style(name, parent="Normal", **kwargs):
    return ParagraphStyle(name, parent=base_styles[parent], **kwargs)

S = {
    "cover_title": style("cover_title",
        fontSize=28, textColor=white, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceAfter=12),
    "cover_sub": style("cover_sub",
        fontSize=13, textColor=MID_GREY, fontName="Helvetica",
        alignment=TA_CENTER, spaceAfter=6),
    "cover_meta": style("cover_meta",
        fontSize=10, textColor=MID_GREY, fontName="Helvetica",
        alignment=TA_CENTER, spaceAfter=4),
    "h1": style("h1",
        fontSize=16, textColor=DARK_SLATE, fontName="Helvetica-Bold",
        spaceBefore=18, spaceAfter=8, borderPad=4),
    "h2": style("h2",
        fontSize=12, textColor=DARK_SLATE, fontName="Helvetica-Bold",
        spaceBefore=12, spaceAfter=6),
    "h3": style("h3",
        fontSize=10, textColor=ACCENT_BLUE, fontName="Helvetica-Bold",
        spaceBefore=8, spaceAfter=4),
    "body": style("body",
        fontSize=9.5, textColor=DARK_SLATE, fontName="Helvetica",
        leading=15, spaceAfter=6, alignment=TA_JUSTIFY),
    "body_bold": style("body_bold",
        fontSize=9.5, textColor=DARK_SLATE, fontName="Helvetica-Bold",
        leading=15, spaceAfter=4),
    "bullet": style("bullet",
        fontSize=9.5, textColor=DARK_SLATE, fontName="Helvetica",
        leading=15, spaceAfter=3, leftIndent=16, bulletIndent=6),
    "kpi_value": style("kpi_value",
        fontSize=22, textColor=CRITICAL_RED, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceAfter=0),
    "kpi_label": style("kpi_label",
        fontSize=8, textColor=DARK_SLATE, fontName="Helvetica",
        alignment=TA_CENTER, spaceAfter=0),
    "table_header": style("table_header",
        fontSize=8.5, textColor=white, fontName="Helvetica-Bold",
        alignment=TA_CENTER),
    "table_cell": style("table_cell",
        fontSize=8.5, textColor=DARK_SLATE, fontName="Helvetica",
        alignment=TA_LEFT, leading=12),
    "table_cell_c": style("table_cell_c",
        fontSize=8.5, textColor=DARK_SLATE, fontName="Helvetica",
        alignment=TA_CENTER, leading=12),
    "footer_text": style("footer_text",
        fontSize=7.5, textColor=MID_GREY, fontName="Helvetica",
        alignment=TA_CENTER),
    "confidential": style("confidential",
        fontSize=9, textColor=CRITICAL_RED, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceAfter=4),
}

# ── Helper functions ──────────────────────────────────────────────────────────
def hr(color=MID_GREY, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=8, spaceBefore=4)

def kpi_table(kpis: list) -> Table:
    """
    kpis: list of (value_str, label_str) tuples
    Renders as a single-row KPI card strip.
    """
    headers = [[Paragraph(v, S["kpi_value"]) for v, _ in kpis]]
    labels  = [[Paragraph(l, S["kpi_label"]) for _, l in kpis]]
    data    = headers + labels
    col_w   = [17*cm / len(kpis)] * len(kpis)
    t = Table(data, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LIGHT_GREY),
        ("BOX",        (0,0), (-1,-1), 0.5, MID_GREY),
        ("LINEAFTER",  (0,0), (-2,-1), 0.5, MID_GREY),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    return t

def section_header(text, level="h1"):
    return [
        Paragraph(text, S[level]),
        hr(DARK_SLATE if level == "h1" else MID_GREY,
           1.0 if level == "h1" else 0.5),
    ]

def finding_table(headers, rows, col_widths=None):
    """Render a styled audit finding table."""
    header_row = [Paragraph(h, S["table_header"]) for h in headers]
    body_rows  = [
        [Paragraph(str(c), S["table_cell_c"] if i > 0 else S["table_cell"])
         for i, c in enumerate(row)]
        for row in rows
    ]
    data = [header_row] + body_rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),  (-1,0),  DARK_SLATE),
        ("TEXTCOLOR",     (0,0),  (-1,0),  white),
        ("ROWBACKGROUNDS",(0,1),  (-1,-1), [white, LIGHT_GREY]),
        ("GRID",          (0,0),  (-1,-1), 0.3, MID_GREY),
        ("TOPPADDING",    (0,0),  (-1,-1), 5),
        ("BOTTOMPADDING", (0,0),  (-1,-1), 5),
        ("LEFTPADDING",   (0,0),  (-1,-1), 6),
        ("RIGHTPADDING",  (0,0),  (-1,-1), 6),
        ("VALIGN",        (0,0),  (-1,-1), "MIDDLE"),
    ]))
    return t

def risk_badge(tier):
    colours = {"Critical": CRITICAL_RED, "High": HIGH_ORANGE,
               "Medium": MED_GOLD,       "Low": LOW_GREEN}
    c = colours.get(tier, MID_GREY)
    return Table([[Paragraph(tier, ParagraphStyle("rb",
        fontSize=8, textColor=white, fontName="Helvetica-Bold",
        alignment=TA_CENTER))]],
        colWidths=[2*cm],
        style=TableStyle([
            ("BACKGROUND",    (0,0),(0,0), c),
            ("TOPPADDING",    (0,0),(0,0), 3),
            ("BOTTOMPADDING", (0,0),(0,0), 3),
        ])
    )

# ═════════════════════════════════════════════════════════════════════════════
# BUILD REPORT CONTENT
# ═════════════════════════════════════════════════════════════════════════════
story = []

# ─────────────────────────────────────────────────────────────────────────────
# COVER PAGE
# ─────────────────────────────────────────────────────────────────────────────
cover_bg = Table(
    [[Paragraph("INTERNAL AUDIT REPORT", S["cover_title"])],
     [Paragraph("AI-Powered ERP Anomaly Detection &amp;", S["cover_sub"])],
     [Paragraph("Procurement Fraud Risk Assessment", S["cover_sub"])],
     [Spacer(1, 0.4*cm)],
     [Paragraph("SAP Procure-to-Pay Transaction Analysis", S["cover_meta"])],
     [Paragraph("Fiscal Years 2022 – 2023", S["cover_meta"])],
     [Spacer(1, 0.8*cm)],
     [Paragraph("CONFIDENTIAL — INTERNAL USE ONLY", S["confidential"])],
     ],
    colWidths=[17*cm],
    style=TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), DARK_SLATE),
        ("TOPPADDING",    (0,0), (0,-1), 12),
        ("BOTTOMPADDING", (0,0), (0,-1), 12),
        ("LEFTPADDING",   (0,0), (0,-1), 24),
        ("RIGHTPADDING",  (0,0), (0,-1), 24),
    ])
)
story.append(cover_bg)
story.append(Spacer(1, 0.8*cm))

meta_data = [
    ["Prepared by:",        "Internal Audit & Analytics Function"],
    ["Engagement type:",    "Continuous Controls Monitoring — AI-Augmented"],
    ["Period covered:",     "01 January 2022 – 31 December 2023"],
    ["Transactions reviewed:", f"{total_transactions:,}"],
    ["Report date:",        "July 2026"],
    ["Classification:",     "Confidential"],
]
meta_table = Table(meta_data, colWidths=[4.5*cm, 12.5*cm])
meta_table.setStyle(TableStyle([
    ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTNAME",  (1,0), (1,-1), "Helvetica"),
    ("FONTSIZE",  (0,0), (-1,-1), 9.5),
    ("TEXTCOLOR", (0,0), (-1,-1), DARK_SLATE),
    ("TOPPADDING",(0,0), (-1,-1), 4),
    ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LINEBELOW", (0,-1),(1,-1), 0.5, MID_GREY),
]))
story.append(meta_table)
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
story += section_header("1. Executive Summary")

story.append(Paragraph(
    "Internal Audit engaged the Analytics function to conduct an AI-powered review "
    "of the organisation's Procure-to-Pay (P2P) transaction data extracted from the "
    "SAP ERP environment. The engagement applied a multi-layer detection framework "
    "combining rule-based controls monitoring — replicating SAP GRC Process Control "
    "logic — with unsupervised machine learning models to identify procurement fraud "
    "indicators, financial control failures, and policy violations across the full "
    f"population of {total_transactions:,} invoice transactions processed during "
    "fiscal years 2022 and 2023.",
    S["body"]))

story.append(Spacer(1, 0.3*cm))
story.append(kpi_table([
    (f"{total_transactions:,}", "Transactions Analysed"),
    (f"{total_anomalies:,}",    "Anomalies Detected"),
    (f"₹{total_exposure/1e7:.1f} Cr", "Financial Exposure at Risk"),
    (f"{detection_rate:.1f}%",  "Anomaly Detection Rate"),
    (f"{audit_efficiency:.1f}%","Audit Workload Reduction"),
]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("Top-Line Findings", S["h2"]))
story.append(Paragraph(
    f"The review identified <b>{total_anomalies:,} transactions</b> exhibiting one or "
    f"more indicators of financial irregularity, representing a detection rate of "
    f"<b>{detection_rate:.1f}%</b> of the total transaction population. Combined "
    f"financial exposure across Critical and High risk tiers amounts to "
    f"<b>₹{total_exposure/1e7:.2f} crore</b> (₹{total_exposure/1e6:.1f} million). "
    f"The AI-augmented detection framework reduced manual audit review workload by "
    f"<b>{audit_efficiency:.1f}%</b> by auto-clearing {low_count + medium_count:,} "
    f"low-risk transactions, enabling auditors to focus investigative effort on "
    f"{critical_count + high_count} priority cases.",
    S["body"]))

story.append(Paragraph(
    f"Three-Way Match controls are operating at a <b>{match_pass_rate:.0f}% pass rate</b>, "
    f"above the industry benchmark of 95%, indicating that the automated matching "
    f"control in SAP is largely effective. However, <b>{match_fail_count} invoices "
    f"with match failures</b> were identified representing deliberate or systemic "
    f"deviation beyond the configured tolerance threshold. Additionally, "
    f"<b>{ghost_vendor_inv} ghost vendor invoices</b> from vendors created and "
    f"utilised within a 22-day window were detected, representing the highest "
    f"per-transaction financial exposure in this review.",
    S["body"]))

story.append(Paragraph("Three Key Recommendations", S["h2"]))
recs = [
    ("1. Immediate payment hold on all Critical and High tier transactions",
     f"One Critical and {high_count} High tier transactions totalling "
     f"₹{total_exposure/1e7:.2f} crore require immediate payment suspension "
     "pending investigation. Ghost vendor and three-way match failure cases "
     "carry the highest per-transaction value and should be prioritised."),
    ("2. Implement automated ghost vendor detection in SAP GRC",
     "The current vendor master creation workflow lacks automated controls "
     "to flag vendors transacting within 30 days of creation. SAP GRC Access "
     "Control should be configured to enforce a mandatory review period for "
     "all new vendors before their first invoice can be processed."),
    ("3. Mandate Purchase Orders for all procurement activity",
     f"{maverick_count} invoices ({maverick_count/total_transactions*100:.1f}% "
     "of total) were processed with no corresponding Purchase Order, bypassing "
     "price negotiation, approval workflows, and three-way match controls. "
     "A hard system block on PO-less invoice posting should be implemented "
     "in SAP MM configuration."),
]
for title, body in recs:
    story.append(Paragraph(f"<b>{title}:</b> {body}", S["bullet"]))

story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
story += section_header("2. Methodology")

story.append(Paragraph("2.1  Data Sources and Coverage", S["h2"]))
story.append(Paragraph(
    "Transaction data was extracted from the SAP Procure-to-Pay process spanning "
    "six core data entities: Purchase Orders (EKKO/EKPO), Goods Receipts (MSEG/MKPF), "
    "Vendor Invoices (RBKP/RSEG), Vendor Master (LFA1/LFB1), Cost Centre Budgets "
    "(BPGE/BPJA), and User Master (USR02/AGR_USERS). The dataset covers fiscal years "
    f"2022 and 2023, encompassing {total_transactions:,} invoice records, 20,150 "
    "purchase orders, 12,868 goods receipts, and 315 vendor master records across "
    "20 cost centres.", S["body"]))

story.append(Paragraph("2.2  Detection Framework", S["h2"]))
story.append(Paragraph(
    "The detection framework operates in three layers applied sequentially:", S["body"]))

layers = [
    ["Layer", "Approach", "Rules / Models", "Purpose"],
    ["1 — Rule Engine", "Deterministic",
     "9 control rules", "Detect known, named violations"],
    ["2 — ML Models", "Unsupervised",
     "Isolation Forest + Autoencoder", "Detect statistical anomalies"],
    ["3 — Risk Scoring", "Composite",
     "Weighted ensemble (40/35/25)",
     "Prioritise by ML score, rule severity, and value"],
]
story.append(finding_table(
    layers[0], layers[1:],
    col_widths=[3.5*cm, 3.5*cm, 5*cm, 5*cm]
))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("2.3  Rule-Based Detection", S["h2"]))
rule_rows = [
    ["Rule", "Control Tested", "Threshold"],
    ["Exact Duplicate Invoice",    "Same vendor, same amount, same ref within 30 days", "Exact match"],
    ["Near-Duplicate Invoice",     "Same vendor, amount within 2%, within 30 days",     "2% tolerance"],
    ["Split Purchase Order",       "Multiple POs below threshold summing above it",      "₹1,00,000"],
    ["Three-Way Match Failure",    "Invoice vs PO deviation outside tolerance",          "±5%"],
    ["Maverick Buying",            "Invoice with no corresponding Purchase Order",       "Any"],
    ["After-Hours Transaction",    "Posting outside 08:00–19:00 or on weekends",         "Business hours"],
    ["Budget Overrun",             "Actual spend exceeds approved cost centre budget",   "100%"],
    ["Round Number Invoice",       "Amount divisible by ₹1,000 or higher modulo",       "Statistical"],
    ["New Vendor Fast Payment",    "Vendor created within 7 days of first invoice",      "7 days"],
]
story.append(finding_table(
    rule_rows[0], rule_rows[1:],
    col_widths=[4.5*cm, 8.5*cm, 4*cm]
))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("2.4  Machine Learning Models", S["h2"]))
story.append(Paragraph(
    "<b>Isolation Forest</b> (scikit-learn, 200 estimators, contamination=0.05): "
    "An unsupervised tree-based ensemble that isolates anomalies by randomly "
    "partitioning the feature space. Anomalies require fewer splits to isolate "
    "and receive lower anomaly scores. Applied to 10 engineered features including "
    "amount z-score, invoice-to-PO ratio, vendor age at transaction date, "
    "after-hours indicator, and Benford's Law first-digit deviation.",
    S["body"]))
story.append(Paragraph(
    "<b>Autoencoder</b> (Keras/TensorFlow, architecture: 10→16→8→3→8→16→10): "
    "A neural network trained exclusively on normal transactions to learn the "
    "compression-reconstruction pattern of legitimate P2P activity. Anomalous "
    "transactions produce high reconstruction error because the model has not "
    "learned their pattern. Threshold set at the 95th percentile of normal "
    "transaction reconstruction error.",
    S["body"]))
story.append(Paragraph(
    "<b>Ensemble</b>: Isolation Forest (55%) and Autoencoder (45%) scores were "
    "min-max normalised and combined into a single ensemble anomaly score. The "
    "composite 0-100 risk score weights the ensemble score (40%), rule-based "
    "severity (35%), and log-scaled transaction value (25%).",
    S["body"]))

story.append(Paragraph("2.5  Limitations and Assumptions", S["h2"]))
limits = [
    "Analysis is based on transaction data only. Physical inspection of goods, "
    "vendor site visits, and bank account verification were outside scope.",
    "Ghost vendor detection relies on temporal proximity signals; shell companies "
    "with longer setup periods may not be captured by the 7-day threshold.",
    "Unsupervised ML models flag statistical outliers — not all flagged transactions "
    "will represent actual fraud. Human investigation is required to confirm findings.",
    "Budget overrun analysis is limited to cost-centre level; sub-account and "
    "project-level budget violations are outside scope.",
]
for l in limits:
    story.append(Paragraph(f"• {l}", S["bullet"]))

story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DETAILED FINDINGS
# ─────────────────────────────────────────────────────────────────────────────
story += section_header("3. Detailed Findings")

# Each finding follows the same structure:
# Description → Transactions flagged → Financial exposure →
# Business impact → Root cause hypothesis → Recommended action

findings = [
    {
        "id":      "F-01",
        "title":   "Three-Way Match Failures",
        "tier":    "Critical",
        "count":   match_fail_count,
        "exposure": df[df["anomaly_type"]=="Three-Way Match Failure"]["invoice_amount"].sum(),
        "description": (
            "Three-way match validation compares the invoice amount against the "
            "originating Purchase Order and Goods Receipt within a ±5% tolerance. "
            f"The review identified {match_fail_count} invoices where the invoice "
            "amount deviated beyond this threshold, with deviations ranging from "
            "-34.7% to +34.9% of PO value. SAP correctly blocked these invoices "
            "for payment, however the volume of failures indicates a systemic issue "
            "requiring root cause investigation."
        ),
        "impact": (
            "Invoices above PO value represent potential vendor overbilling. "
            "Invoices significantly below PO value may indicate partial delivery "
            "being billed at full value, or price manipulation to remain close "
            "to the approval tolerance boundary."
        ),
        "root_cause": (
            "Most probable causes: (1) vendor invoicing at incorrect contracted "
            "rate, (2) quantity discrepancy between delivery and invoice, "
            "(3) deliberate amount manipulation to stay below investigation "
            "thresholds. The concentration of deviations between 5-15% suggests "
            "pattern (3) in a subset of cases."
        ),
        "recommendation": (
            "AP Manager to review and clear each blocked invoice individually "
            "with written vendor explanation. Cases with deviation >15% to be "
            "escalated to Internal Audit for full investigation. Implement SAP "
            "workflow routing for match failures above 10% deviation directly "
            "to the Finance Manager rather than AP Clerk level."
        ),
    },
    {
        "id":      "F-02",
        "title":   "Ghost Vendor Invoices",
        "tier":    "Critical",
        "count":   ghost_vendor_inv,
        "exposure": df[df["anomaly_type"]=="Ghost Vendor Invoice"]["invoice_amount"].sum(),
        "description": (
            f"{ghost_vendor_inv} invoices were identified from 15 vendors created "
            "within 3-22 days of their first invoice submission. These vendors "
            "exhibit multiple ghost vendor indicators: non-corporate contact email "
            "addresses, residential-pattern registered addresses, and payment terms "
            "of 15 days (significantly shorter than the standard 30-90 day terms). "
            "All ghost vendor invoices were paid, with no corresponding Purchase "
            "Order or Goods Receipt."
        ),
        "impact": (
            "Ghost vendor fraud represents the highest per-transaction financial "
            "exposure in this engagement. Payments made to fraudulent vendors "
            "result in direct financial loss with no goods or services received."
        ),
        "root_cause": (
            "The vendor master creation workflow does not enforce a mandatory "
            "cooling-off period before a new vendor can receive payment. "
            "Additionally, Segregation of Duties analysis indicates that in "
            "several instances, the user who created the vendor record also had "
            "access to the invoice approval workflow — a critical SoD violation."
        ),
        "recommendation": (
            "Immediately freeze all 15 identified ghost vendors in the SAP "
            "vendor master (transaction MK06). Initiate bank account verification "
            "for all 15 vendors via independent channel. Implement a mandatory "
            "30-day new vendor review period in SAP GRC Access Control before "
            "invoice processing is permitted. Investigate SoD violations for "
            "users with combined vendor creation and payment approval access."
        ),
    },
    {
        "id":      "F-03",
        "title":   "Maverick Buying — Invoices Without Purchase Orders",
        "tier":    "High",
        "count":   maverick_count,
        "exposure": df[df["anomaly_type"]=="Maverick Buying"]["invoice_amount"].sum(),
        "description": (
            f"{maverick_count} invoices ({maverick_count/total_transactions*100:.1f}% "
            "of total) were processed and paid without a corresponding Purchase "
            "Order reference. These transactions completely bypassed the procurement "
            "control framework — no vendor selection process, no price negotiation, "
            "no approval workflow, and no three-way match capability."
        ),
        "impact": (
            "Maverick buying exposes the organisation to overpayment risk, "
            "prevents leverage in vendor negotiations, and creates audit trail "
            "gaps. Industry benchmarks suggest maverick buying rates above 10% "
            "indicate a procurement compliance culture issue requiring policy "
            "intervention, not just individual transaction remediation."
        ),
        "root_cause": (
            "SAP MM is not configured with a hard block on PO-less invoice "
            "posting. AP Clerks have the system access to post invoices without "
            "a PO reference, and the current AP Manager approval workflow does "
            "not flag or escalate PO-less invoices differently from matched ones."
        ),
        "recommendation": (
            "Configure SAP MM to require a valid PO reference for all invoices "
            "above ₹10,000. Implement a separate AP workflow queue for PO-less "
            "invoices requiring Finance Manager sign-off and business justification. "
            "Issue procurement policy reminder to all department heads with "
            "mandatory PO-first training for budget holders."
        ),
    },
    {
        "id":      "F-04",
        "title":   "Duplicate and Near-Duplicate Invoices",
        "tier":    "High",
        "count":   duplicate_count + near_dup_count,
        "exposure": df[df["anomaly_type"].isin(
            ["Duplicate Invoice","Near-Duplicate Invoice"])]["invoice_amount"].sum(),
        "description": (
            f"The review identified {duplicate_count} exact duplicate invoices "
            f"and {near_dup_count} near-duplicate invoices (same vendor, amount "
            "differing by 0.5-2%, within 30 days). Exact duplicates share the "
            "same vendor invoice reference number — a primary indicator of "
            "inadvertent re-submission or deliberate double-billing. Near-duplicates "
            "with slightly altered amounts suggest sophisticated evasion of "
            "standard exact-match duplicate detection controls."
        ),
        "impact": (
            "Each duplicate invoice pair represents a potential double-payment "
            "to the vendor. Near-duplicate patterns, particularly where the "
            "amount difference is consistently less than 2%, warrant investigation "
            "for deliberate manipulation."
        ),
        "root_cause": (
            "SAP's standard duplicate invoice check (transaction MIRO) is "
            "configured for exact matching on vendor, amount, and reference "
            "number. Near-duplicate patterns with slightly altered amounts or "
            "reference numbers evade this check. Manual AP Clerk review is "
            "the only current control for near-duplicates — insufficient at "
            "current transaction volumes."
        ),
        "recommendation": (
            "Implement fuzzy matching logic in the AP invoice processing "
            "workflow to flag invoices within 5% of a recent same-vendor "
            "invoice for manual review."
            f"All {duplicate_count} exact duplicates "
            "should be confirmed with vendors that only one payment is due. "
            "Recovery procedures should be initiated for any duplicates "
            "already paid."
        ),
    },
    {
        "id":      "F-05",
        "title":   "After-Hours Transaction Postings",
        "tier":    "Medium",
        "count":   after_hours_count,
        "exposure": df[df["anomaly_type"]=="After-Hours Transaction"]["invoice_amount"].sum(),
        "description": (
            f"{after_hours_count} invoice postings were recorded outside normal "
            "business hours (08:00–19:00 Monday to Friday). Legitimate AP "
            "operations are conducted by office-based staff during business hours. "
            "After-hours postings may indicate unauthorised system access, "
            "credential sharing, or deliberate timing of fraudulent transactions "
            "to reduce oversight probability."
        ),
        "impact": (
            "While not all after-hours postings represent fraud, the combination "
            "of after-hours timing with other risk factors (high value, new vendor, "
            "no PO) significantly elevates the risk profile of individual transactions."
        ),
        "root_cause": (
            "SAP does not have time-based access restrictions configured for "
            "AP Clerk and AP Manager roles. Users retain full transaction "
            "access 24 hours a day, seven days a week."
        ),
        "recommendation": (
            "Implement SAP logon time restrictions for AP roles via transaction "
            "SU01 / SU10, limiting system access to 07:00–21:00 on business days. "
            "After-hours access should require explicit Finance Manager authorisation "
            "via a temporary elevated access request."
            f"All {after_hours_count} "
            "after-hours transactions should be reviewed for business justification."
        ),
    },
    {
        "id":      "F-06",
        "title":   "Budget Overruns",
        "tier":    "Medium",
        "count":   budget_overruns,
        "exposure": budget_df[budget_df["is_overrun"]==True]["actual_spend"].sum()
                    - budget_df[budget_df["is_overrun"]==True]["approved_budget"].sum(),
        "description": (
            f"{budget_overruns} cost centres recorded actual expenditure exceeding "
            "their approved annual budget, with utilisation rates ranging from "
            "100.1% to 125.7%. SAP CO was not configured with hard budget checks "
            "(availability control) for these cost centres, allowing transactions "
            "to post beyond the approved limit without system intervention."
        ),
        "impact": (
            "Budget overruns indicate either inadequate budget planning, "
            "unauthorised expenditure, or deliberate circumvention of financial "
            "controls. Overruns that are not identified and approved in-period "
            "create cash flow risk and reporting inaccuracies."
        ),
        "root_cause": (
            "SAP CO availability control is configured in 'Warning' mode rather "
            "than 'Error' mode for the affected cost centres, allowing postings "
            "to proceed beyond budget with only a system warning that users "
            "can override."
        ),
        "recommendation": (
            "Switch SAP CO availability control to 'Error' mode for all cost "
            "centres above ₹50 lakh annual budget. Implement a supplementary "
            "budget approval workflow requiring CFO sign-off before overrun "
            "postings are permitted. Obtain retrospective CFO approval for "
            f"all {budget_overruns} identified overruns."
        ),
    },
]

for f in findings:
    story.append(KeepTogether([
        Paragraph(f"{f['id']} — {f['title']}", S["h2"]),
        hr(MID_GREY, 0.3),
    ]))

    # Finding summary strip
    summary_data = [
        [Paragraph("Transactions Flagged", S["kpi_label"]),
         Paragraph("Financial Exposure", S["kpi_label"]),
         Paragraph("Risk Rating", S["kpi_label"])],
        [Paragraph(f"{f['count']:,}", S["kpi_value"]),
         Paragraph(f"₹{f['exposure']/1e6:.1f}M", S["kpi_value"]),
         risk_badge(f["tier"])],
    ]
    summary_t = Table(summary_data, colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    summary_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), LIGHT_GREY),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("BOX",           (0,0), (-1,-1), 0.5, MID_GREY),
        ("LINEAFTER",     (0,0), (-2,-1), 0.5, MID_GREY),
    ]))
    story.append(summary_t)
    story.append(Spacer(1, 0.2*cm))

    for label, key in [
        ("Finding Description",  "description"),
        ("Business Impact",      "impact"),
        ("Root Cause Hypothesis","root_cause"),
        ("Recommended Action",   "recommendation"),
    ]:
        story.append(Paragraph(label, S["h3"]))
        story.append(Paragraph(f[key], S["body"]))

    story.append(Spacer(1, 0.4*cm))

story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — OVERALL RISK RATING
# ─────────────────────────────────────────────────────────────────────────────
story += section_header("4. Overall Procurement Control Environment Risk Rating")

rating_table = Table(
    [[Paragraph("HIGH RISK", ParagraphStyle("rating",
        fontSize=28, textColor=white, fontName="Helvetica-Bold",
        alignment=TA_CENTER))]],
    colWidths=[17*cm],
    style=TableStyle([
        ("BACKGROUND",    (0,0),(0,0), HIGH_ORANGE),
        ("TOPPADDING",    (0,0),(0,0), 16),
        ("BOTTOMPADDING", (0,0),(0,0), 16),
    ])
)
story.append(rating_table)
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph(
    "The organisation's procurement control environment is rated <b>HIGH RISK</b> "
    "based on the following assessment criteria:",
    S["body"]))

rating_criteria = [
    ["Risk Domain",                  "Assessment", "Rating"],
    ["Three-Way Match Controls",     "Operating at 98% pass rate — above benchmark. "
                                     "Failure volume requires investigation.",          "Medium"],
    ["Ghost Vendor / Vendor Master", "Critical SoD violations detected. Immediate "
                                     "remediation required.",                           "Critical"],
    ["Procurement Compliance",       f"{maverick_count/total_transactions*100:.1f}% "
                                     "maverick buying rate — systemic policy failure.", "High"],
    ["Duplicate Payment Controls",   "Near-duplicate evasion pattern detected — "
                                     "current controls inadequate.",                    "High"],
    ["Access & Timing Controls",     f"{after_hours_count} after-hours postings — "
                                     "no time-based access restrictions in place.",     "Medium"],
    ["Budget Management",            f"{budget_overruns} cost centres over budget — "
                                     "availability control in Warning mode only.",      "Medium"],
]
story.append(finding_table(
    rating_criteria[0], rating_criteria[1:],
    col_widths=[4.5*cm, 9*cm, 3.5*cm]
))

story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MANAGEMENT ACTION PLAN
# ─────────────────────────────────────────────────────────────────────────────
story += section_header("5. Management Action Plan")

story.append(Paragraph(
    "The following remediation actions are presented in priority order. "
    "Management is requested to confirm acceptance of findings and assign "
    "named owners for each action within 10 business days of report receipt.",
    S["body"]))

actions = [
    ["#", "Action", "Owner", "Timeline", "Priority"],
    ["1", "Freeze all 15 ghost vendors in SAP vendor master (MK06). "
          "Initiate independent bank account verification.",
     "AP Manager", "Immediate\n(1-3 days)", "Critical"],
    ["2", "Place payment hold on all Critical and High tier invoices "
          f"({critical_count + high_count} transactions, ₹{total_exposure/1e7:.1f} Cr).",
     "Finance Manager", "Immediate\n(1-3 days)", "Critical"],
    ["3", "Investigate SoD violations — users with combined vendor "
          "creation and payment approval access.",
     "Internal Audit", "5 business days", "Critical"],
    ["4", "Configure SAP MM hard block on PO-less invoice posting "
          "above ₹10,000.",
     "SAP Basis / MM", "2 weeks", "High"],
    ["5", "Implement 30-day new vendor cooling-off period in "
          "SAP GRC Access Control.",
     "SAP GRC Admin", "2 weeks", "High"],
    ["6", "Deploy fuzzy duplicate invoice matching in AP workflow "
          "to catch near-duplicate evasion.",
     "IT / SAP Dev", "4 weeks", "High"],
    ["7", "Switch SAP CO availability control to Error mode for "
          "all cost centres >₹50L budget.",
     "SAP CO / Finance", "2 weeks", "Medium"],
    ["8", "Implement time-based SAP logon restrictions for AP roles "
          "(07:00–21:00 business days).",
     "SAP Basis", "3 weeks", "Medium"],
    ["9", "Obtain retrospective CFO approval for all budget overruns "
          f"({budget_overruns} cost centres).",
     "CFO Office", "1 week", "Medium"],
    ["10", "Issue mandatory procurement compliance training for all "
           "budget holders and department heads.",
     "HR / Procurement", "4 weeks", "Low"],
]

action_table = Table(
    [[Paragraph(h, S["table_header"]) for h in actions[0]]] +
    [[Paragraph(str(c), S["table_cell_c"] if i in [0,4] else S["table_cell"])
      for i, c in enumerate(row)]
     for row in actions[1:]],
    colWidths=[0.8*cm, 7.5*cm, 3*cm, 2.5*cm, 2.2*cm],
    repeatRows=1,
)
action_table.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),  (-1,0),  DARK_SLATE),
    ("TEXTCOLOR",     (0,0),  (-1,0),  white),
    ("ROWBACKGROUNDS",(0,1),  (-1,-1), [white, LIGHT_GREY]),
    ("GRID",          (0,0),  (-1,-1), 0.3, MID_GREY),
    ("TOPPADDING",    (0,0),  (-1,-1), 5),
    ("BOTTOMPADDING", (0,0),  (-1,-1), 5),
    ("LEFTPADDING",   (0,0),  (-1,-1), 5),
    ("RIGHTPADDING",  (0,0),  (-1,-1), 5),
    ("VALIGN",        (0,0),  (-1,-1), "TOP"),
]))
story.append(action_table)
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph(
    "This report has been prepared for internal use only. Findings are based on "
    "transaction data analytics and statistical modelling; conclusions represent "
    "risk indicators requiring human investigation and should not be treated as "
    "confirmed fraud determinations. Distribution of this report is restricted "
    "to Internal Audit, the CFO, and named action owners.",
    S["footer_text"]))

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PDF
# ─────────────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"\n✓ Audit report generated: {OUTPUT_PATH}")
print("  Open the PDF and verify formatting before attaching to GitHub.")