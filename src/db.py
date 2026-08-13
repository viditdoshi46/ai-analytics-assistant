"""
DuckDB layer for the assistant: load the CSVs into an in-memory database,
expose the schema (so the model knows the tables/columns), and run queries
through a read-only guard.

Governance matters here: the assistant must NEVER be able to modify data.
run_sql() rejects anything that isn't a single read-only SELECT/WITH query.
"""
from __future__ import annotations
import re
from pathlib import Path
import duckdb
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

# Human-readable schema shown to the language model (and to users).
SCHEMA = """
customers(customer_id INT, customer_name TEXT, city TEXT, state TEXT,
          signup_date DATE, segment TEXT)   -- segment: Consumer/Small Business/Enterprise
products(product_id INT, product_name TEXT, category TEXT, unit_price DECIMAL)
orders(order_id INT, customer_id INT, order_date DATE, status TEXT)  -- status: Completed/Cancelled/Returned
order_items(order_id INT, product_id INT, quantity INT, unit_price DECIMAL)

Notes for writing SQL:
- Revenue = SUM(order_items.quantity * order_items.unit_price).
- Only 'Completed' orders count as realized revenue unless stated otherwise.
- Join order_items -> orders (order_id), orders -> customers (customer_id),
  order_items -> products (product_id).
- Use DATE_TRUNC('month', order_date) for monthly trends.
""".strip()

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|copy|"
    r"pragma|install|load|export|grant)\b", re.IGNORECASE)


def get_con() -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    for t in ("customers", "products", "orders", "order_items"):
        con.execute(
            f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{DATA/f'{t}.csv'}', header=true)")
    return con


def is_safe_select(sql: str) -> bool:
    s = sql.strip().strip(";")
    if ";" in s:                       # single statement only
        return False
    if not re.match(r"(?is)^\s*(select|with)\b", s):
        return False
    if _FORBIDDEN.search(s):
        return False
    return True


def run_sql(con, sql: str, limit: int = 1000) -> pd.DataFrame:
    if not is_safe_select(sql):
        raise ValueError("Only single read-only SELECT queries are allowed.")
    s = sql.strip().strip(";")
    if re.search(r"(?is)\blimit\b", s) is None:
        s = f"{s}\nLIMIT {limit}"
    return con.execute(s).df()


if __name__ == "__main__":
    con = get_con()
    print(run_sql(con, "SELECT COUNT(*) AS orders FROM orders"))
