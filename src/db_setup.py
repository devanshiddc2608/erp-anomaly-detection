# db_setup.py
# Creates all ERP tables in PostgreSQL with proper schema.
# Run once before loading data.
# In a real SAP project, this mirrors the DDL a BASIS consultant
# would run when setting up a reporting schema from SAP extracts.

from sqlalchemy import create_engine, text
import sys

# ── Connection string ─────────────────────────────────────────────────────────
# Format: postgresql://username:password@host:port/database_name
DB_URL = "postgresql://postgres:devi2608@localhost:5432/erp_anomaly_detection"

def get_engine():
    """Create and return SQLAlchemy engine."""
    try:
        engine = create_engine(DB_URL, echo=False)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Database connection successful.")
        return engine
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("  Check PostgreSQL is running and credentials are correct.")
        sys.exit(1)

# ── DDL statements — one per table ───────────────────────────────────────────
# DDL = Data Definition Language — the SQL that creates structure

DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    user_id          VARCHAR(10)  PRIMARY KEY,
    user_name        VARCHAR(100) NOT NULL,
    role             VARCHAR(50)  NOT NULL,
    department       VARCHAR(50),
    access_level     INTEGER      CHECK (access_level BETWEEN 1 AND 4),
    last_login_date  TIMESTAMP,
    is_active        BOOLEAN      DEFAULT TRUE
);
"""

DDL_VENDORS = """
CREATE TABLE IF NOT EXISTS vendors (
    vendor_id             VARCHAR(10)  PRIMARY KEY,
    vendor_name           VARCHAR(200) NOT NULL,
    vendor_creation_date  DATE         NOT NULL,
    vendor_category       VARCHAR(50),
    bank_account_number   VARCHAR(50),
    registered_address    TEXT,
    contact_email         VARCHAR(150),
    payment_method        VARCHAR(20),
    is_active             BOOLEAN      DEFAULT TRUE,
    created_by_user       VARCHAR(10)  REFERENCES users(user_id),
    is_ghost_vendor       BOOLEAN      DEFAULT FALSE,
    anomaly_flag          INTEGER      DEFAULT 0
);
"""

DDL_BUDGET = """
CREATE TABLE IF NOT EXISTS budget (
    budget_id               SERIAL       PRIMARY KEY,
    cost_centre             VARCHAR(10)  NOT NULL,
    department              VARCHAR(50),
    budget_period           VARCHAR(10),
    approved_budget         NUMERIC(15,2),
    committed_amount        NUMERIC(15,2),
    actual_spend            NUMERIC(15,2),
    remaining_budget        NUMERIC(15,2),
    budget_utilisation_pct  NUMERIC(6,2),
    is_overrun              BOOLEAN      DEFAULT FALSE,
    anomaly_flag            INTEGER      DEFAULT 0,
    UNIQUE (cost_centre, budget_period)
);
"""

DDL_PURCHASE_ORDERS = """
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_number        VARCHAR(10)   PRIMARY KEY,
    po_date          TIMESTAMP     NOT NULL,
    vendor_id        VARCHAR(10)   REFERENCES vendors(vendor_id),
    vendor_name      VARCHAR(200),
    item_description TEXT,
    quantity_ordered INTEGER,
    unit_price       NUMERIC(15,2),
    total_po_value   NUMERIC(15,2),
    department       VARCHAR(50),
    cost_centre      VARCHAR(10),
    budget_code      VARCHAR(10),
    approval_status  VARCHAR(20),
    approver_id      VARCHAR(10)   REFERENCES users(user_id),
    plant_location   VARCHAR(20),
    is_split_po      BOOLEAN       DEFAULT FALSE,
    anomaly_flag     INTEGER       DEFAULT 0
);
"""

DDL_GOODS_RECEIPTS = """
CREATE TABLE IF NOT EXISTS goods_receipts (
    gr_number          VARCHAR(10)  PRIMARY KEY,
    gr_date            TIMESTAMP    NOT NULL,
    po_number          VARCHAR(10)  REFERENCES purchase_orders(po_number),
    vendor_id          VARCHAR(10)  REFERENCES vendors(vendor_id),
    quantity_received  INTEGER,
    actual_unit_price  NUMERIC(15,2),
    receiving_location VARCHAR(20),
    received_by        VARCHAR(10)  REFERENCES users(user_id),
    anomaly_flag       INTEGER      DEFAULT 0
);
"""

DDL_INVOICES = """
CREATE TABLE IF NOT EXISTS invoices (
    invoice_number        VARCHAR(15)  PRIMARY KEY,
    invoice_date          TIMESTAMP    NOT NULL,
    vendor_invoice_number VARCHAR(20),
    vendor_id             VARCHAR(10)  REFERENCES vendors(vendor_id),
    po_number             VARCHAR(10),  -- nullable: maverick invoices have no PO
    gr_number             VARCHAR(10),  -- nullable: some invoices have no GR
    invoice_amount        NUMERIC(15,2),
    tax_amount            NUMERIC(15,2),
    total_amount          NUMERIC(15,2),
    payment_terms_days    INTEGER,
    due_date              DATE,
    payment_date          DATE,
    payment_amount        NUMERIC(15,2),
    invoice_status        VARCHAR(20),
    processed_by          VARCHAR(10)  REFERENCES users(user_id),
    approved_by           VARCHAR(10),  -- nullable: blocked invoices have no approver
    cost_centre           VARCHAR(10),
    is_duplicate          BOOLEAN      DEFAULT FALSE,
    is_near_duplicate     BOOLEAN      DEFAULT FALSE,
    is_match_failure      BOOLEAN      DEFAULT FALSE,
    is_maverick           BOOLEAN      DEFAULT FALSE,
    is_after_hours        BOOLEAN      DEFAULT FALSE,
    is_round_number       BOOLEAN      DEFAULT FALSE,
    is_ghost_vendor_inv   BOOLEAN      DEFAULT FALSE,
    anomaly_flag          INTEGER      DEFAULT 0,
    anomaly_type          VARCHAR(50)  DEFAULT 'Normal'
);
"""

# ── Indexes for query performance ─────────────────────────────────────────────
# Indexes speed up WHERE clauses and JOIN conditions on these columns.
# In SAP reporting environments, indexes on document date and vendor ID
# are standard because audit queries almost always filter on these.

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_invoices_vendor    ON invoices(vendor_id);",
    "CREATE INDEX IF NOT EXISTS idx_invoices_date      ON invoices(invoice_date);",
    "CREATE INDEX IF NOT EXISTS idx_invoices_po        ON invoices(po_number);",
    "CREATE INDEX IF NOT EXISTS idx_invoices_anomaly   ON invoices(anomaly_flag);",
    "CREATE INDEX IF NOT EXISTS idx_invoices_status    ON invoices(invoice_status);",
    "CREATE INDEX IF NOT EXISTS idx_po_vendor          ON purchase_orders(vendor_id);",
    "CREATE INDEX IF NOT EXISTS idx_po_date            ON purchase_orders(po_date);",
    "CREATE INDEX IF NOT EXISTS idx_gr_po              ON goods_receipts(po_number);",
    "CREATE INDEX IF NOT EXISTS idx_vendors_creation   ON vendors(vendor_creation_date);",
]

def create_tables(engine):
    """Drop existing tables and recreate — safe for development."""
    with engine.begin() as conn:
        # Drop in reverse FK dependency order
        print("Dropping existing tables (if any)...")
        conn.execute(text("""
            DROP TABLE IF EXISTS invoices, goods_receipts,
                                 purchase_orders, budget,
                                 vendors, users CASCADE;
        """))

        print("Creating tables...")
        for ddl in [DDL_USERS, DDL_VENDORS, DDL_BUDGET,
                    DDL_PURCHASE_ORDERS, DDL_GOODS_RECEIPTS, DDL_INVOICES]:
            conn.execute(text(ddl))

        print("Creating indexes...")
        for idx_sql in DDL_INDEXES:
            conn.execute(text(idx_sql))

    print("✓ All tables and indexes created successfully.")

if __name__ == "__main__":
    engine = get_engine()
    create_tables(engine)