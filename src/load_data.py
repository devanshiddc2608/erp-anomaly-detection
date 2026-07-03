# load_data.py
# Loads all six CSV files into PostgreSQL.
# Uses pandas .to_sql() with SQLAlchemy — the standard approach
# for loading DataFrames into a relational database.

import pandas as pd
from sqlalchemy import create_engine, text
import os, sys
from config import OUTPUT_DIR

DB_URL = "postgresql://postgres:devi2608@localhost:5432/erp_anomaly_detection"

def get_engine():
    engine = create_engine(DB_URL, echo=False)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine

def load_table(engine, filename: str, table_name: str,
               parse_dates: list = None) -> int:
    """Load a single CSV into a PostgreSQL table."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"  ✗ File not found: {path} — run generate_data.py first.")
        return 0

    df = pd.read_csv(path, parse_dates=parse_dates)

    # Replace Python NaN with None so Postgres sees NULL, not the string 'nan'
    df = df.where(pd.notnull(df), None)

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="append",   # append to existing empty table (schema already created)
        index=False,
        method="multi",       # batch inserts — much faster than row-by-row
        chunksize=1000,       # insert 1000 rows at a time
    )
    return len(df)

def verify_load(engine):
    """Run row count checks on every table after loading."""
    tables = ["users", "vendors", "budget",
              "purchase_orders", "goods_receipts", "invoices"]
    print("\nVerification — row counts:")
    with engine.connect() as conn:
        for t in tables:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {t}"))
            count  = result.scalar()
            print(f"  {t:<20} {count:>8,} rows")

def main():
    print("=" * 50)
    print("ERP Data Loader")
    print("=" * 50)
    engine = get_engine()
    print("✓ Connected to database.\n")

    # Load order matters — respect foreign key dependencies
    # Users and Vendors must exist before POs; POs before GRs; etc.
    load_order = [
        ("users.csv",           "users",           ["last_login_date"]),
        ("vendors.csv",         "vendors",          ["vendor_creation_date"]),
        ("budget.csv",          "budget",           None),
        ("purchase_orders.csv", "purchase_orders",  ["po_date"]),
        ("goods_receipts.csv",  "goods_receipts",   ["gr_date"]),
        ("invoices.csv",        "invoices",
            ["invoice_date", "due_date", "payment_date"]),
    ]

    for filename, table, date_cols in load_order:
        print(f"Loading {filename} → {table}...")
        n = load_table(engine, filename, table, parse_dates=date_cols)
        print(f"  ✓ {n:,} rows loaded.")

    verify_load(engine)
    print("\n✓ All data loaded successfully.")

if __name__ == "__main__":
    main()