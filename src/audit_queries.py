# audit_queries.py
# Replicates the SQL audit reports a forensic accounting team
# would run against SAP data exports.
# Each query maps to a specific internal control test.

import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:devi2608@localhost:5432/erp_anomaly_detection"

def run_query(engine, sql: str, label: str) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    print(f"\n{'='*60}")
    print(f"AUDIT TEST: {label}")
    print('='*60)
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    print(f"  → {len(df)} exceptions found.")
    if len(df) > 0:
        print(df.head(10).to_string(index=False))
    return df

def main():
    engine = create_engine(DB_URL, echo=False)

    # ── Query 1 ───────────────────────────────────────────────────────────────
    # Control test: Invoice amount should be within 5% of PO value.
    # Deviation above tolerance = possible price manipulation or GR shortfall.
    q1 = run_query(engine, """
        SELECT
            i.invoice_number,
            i.invoice_date,
            i.vendor_id,
            po.total_po_value                                    AS po_amount,
            i.invoice_amount,
            ROUND(
                ((i.invoice_amount - po.total_po_value)
                 / po.total_po_value) * 100, 2
            )                                                    AS deviation_pct,
            i.invoice_status,
            i.anomaly_type
        FROM invoices i
        JOIN purchase_orders po ON i.po_number = po.po_number
        WHERE po.total_po_value > 0
          AND ABS(i.invoice_amount - po.total_po_value)
              / po.total_po_value > 0.05      -- 5% tolerance threshold
        ORDER BY ABS(i.invoice_amount - po.total_po_value)
                 / po.total_po_value DESC
        LIMIT 50;
    """, "Invoice Amount vs PO Amount Deviation > 5%")

    # ── Query 2 ───────────────────────────────────────────────────────────────
    # Control test: Same vendor, same invoice amount, same month = duplicate risk.
    # A single vendor should not invoice the same amount twice in one month.
    q2 = run_query(engine, """
        SELECT
            vendor_id,
            invoice_amount,
            DATE_TRUNC('month', invoice_date)  AS invoice_month,
            COUNT(*)                            AS invoice_count,
            ARRAY_AGG(invoice_number)           AS invoice_numbers
        FROM invoices
        WHERE invoice_amount > 0
        GROUP BY vendor_id, invoice_amount,
                 DATE_TRUNC('month', invoice_date)
        HAVING COUNT(*) > 1
        ORDER BY invoice_count DESC
        LIMIT 30;
    """, "Duplicate Invoice Amounts — Same Vendor, Same Month")

    # ── Query 3 ───────────────────────────────────────────────────────────────
    # Control test: Split PO detection.
    # Multiple POs to the same vendor in same month, each below threshold,
    # but combined they exceed the approval threshold.
    q3 = run_query(engine, """
        SELECT
            vendor_id,
            DATE_TRUNC('month', po_date)   AS po_month,
            COUNT(*)                        AS po_count,
            SUM(total_po_value)             AS total_combined_value,
            MAX(total_po_value)             AS max_single_po,
            ARRAY_AGG(po_number)            AS po_numbers
        FROM purchase_orders
        WHERE total_po_value < 100000      -- each is below threshold
          AND approval_status = 'Approved'
        GROUP BY vendor_id, DATE_TRUNC('month', po_date)
        HAVING COUNT(*) > 1
           AND SUM(total_po_value) > 100000  -- but combined they exceed it
        ORDER BY total_combined_value DESC
        LIMIT 30;
    """, "Split PO Detection — Combined Value Exceeds Approval Threshold")

    # ── Query 4 ───────────────────────────────────────────────────────────────
    # Control test: Maverick buying — invoices with no PO reference.
    # Every legitimate purchase should start with a PO.
    q4 = run_query(engine, """
        SELECT
            i.invoice_number,
            i.invoice_date,
            i.vendor_id,
            v.vendor_name,
            v.vendor_category,
            i.invoice_amount,
            i.invoice_status,
            i.processed_by
        FROM invoices i
        JOIN vendors v ON i.vendor_id = v.vendor_id
        WHERE i.po_number IS NULL
        ORDER BY i.invoice_amount DESC
        LIMIT 50;
    """, "Maverick Buying — Invoices with No PO Reference")

    # ── Query 5 ───────────────────────────────────────────────────────────────
    # Control test: Transactions posted outside business hours.
    # Legitimate AP postings happen 8AM–7PM weekdays.
    q5 = run_query(engine, """
        SELECT
            invoice_number,
            invoice_date,
            EXTRACT(HOUR FROM invoice_date)     AS posting_hour,
            EXTRACT(DOW  FROM invoice_date)     AS day_of_week,
            -- 0=Sunday, 1=Monday ... 6=Saturday in PostgreSQL
            CASE EXTRACT(DOW FROM invoice_date)
                WHEN 0 THEN 'Sunday'
                WHEN 6 THEN 'Saturday'
                ELSE 'Weekday'
            END                                  AS day_type,
            vendor_id,
            invoice_amount,
            processed_by
        FROM invoices
        WHERE
            EXTRACT(DOW FROM invoice_date) IN (0, 6)   -- weekend
            OR EXTRACT(HOUR FROM invoice_date) < 8      -- before 8AM
            OR EXTRACT(HOUR FROM invoice_date) >= 19    -- after 7PM
        ORDER BY invoice_date
        LIMIT 50;
    """, "After-Hours Transactions — Outside Business Hours or Weekend")

    # ── Query 6 ───────────────────────────────────────────────────────────────
    # Control test: Budget overrun by cost centre.
    # Actual spend should not exceed approved budget.
    q6 = run_query(engine, """
        SELECT
            b.cost_centre,
            b.department,
            b.budget_period,
            b.approved_budget,
            b.actual_spend,
            ROUND(b.actual_spend - b.approved_budget, 2)    AS overrun_amount,
            ROUND(b.budget_utilisation_pct, 1)               AS utilisation_pct
        FROM budget b
        WHERE b.actual_spend > b.approved_budget
        ORDER BY (b.actual_spend - b.approved_budget) DESC;
    """, "Budget Overruns — Actual Spend Exceeds Approved Budget")

    # ── Query 7 ───────────────────────────────────────────────────────────────
    # Control test: Ghost vendor signal.
    # Vendor created and first used within 7 days = high risk.
    q7 = run_query(engine, """
        SELECT
            v.vendor_id,
            v.vendor_name,
            v.vendor_creation_date,
            v.created_by_user,
            v.contact_email,
            MIN(i.invoice_date)::date             AS first_invoice_date,
            MIN(i.invoice_date)::date
                - v.vendor_creation_date          AS days_to_first_invoice,
            SUM(i.invoice_amount)                 AS total_invoiced,
            COUNT(i.invoice_number)               AS invoice_count,
            v.is_ghost_vendor
        FROM vendors v
        JOIN invoices i ON v.vendor_id = i.vendor_id
        GROUP BY v.vendor_id, v.vendor_name, v.vendor_creation_date,
                 v.created_by_user, v.contact_email, v.is_ghost_vendor
        HAVING MIN(i.invoice_date)::date
               - v.vendor_creation_date <= 7
        ORDER BY days_to_first_invoice ASC
        LIMIT 30;
    """, "Ghost Vendor Signal — Created and First Used Within 7 Days")

    print("\n✓ All audit queries complete.")

if __name__ == "__main__":
    main()