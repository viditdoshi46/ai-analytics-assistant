"""
Generate a small, realistic e-commerce dataset (star schema) for the
Ask-Your-Data assistant. Deterministic (seeded) so the demo is reproducible.

Tables written to data/:
  customers.csv   (customer_id, customer_name, city, state, signup_date, segment)
  products.csv    (product_id, product_name, category, unit_price)
  orders.csv      (order_id, customer_id, order_date, status)
  order_items.csv (order_id, product_id, quantity, unit_price)

The data has built-in trends so the insights are interesting: revenue grows
month over month, a few categories dominate, and returning customers spend more.

Usage:  python src/make_data.py
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(11)

STATES = ["CA", "TX", "NY", "FL", "IL", "WA", "MA", "GA", "NC", "CO", "NJ", "VA"]
CITIES = {s: [f"{s}-City-{i}" for i in range(1, 5)] for s in STATES}
SEGMENTS = ["Consumer", "Small Business", "Enterprise"]
CATEGORIES = {
    "Electronics": (60, 900),
    "Home & Kitchen": (20, 300),
    "Office Supplies": (5, 120),
    "Apparel": (15, 150),
    "Sports & Outdoors": (20, 400),
    "Beauty": (8, 90),
}
FIRST = ["Alex", "Sam", "Priya", "Wei", "Maria", "John", "Aisha", "Diego",
         "Emma", "Raj", "Sofia", "Liam", "Nina", "Omar", "Grace", "Noah"]
LAST = ["Smith", "Patel", "Nguyen", "Garcia", "Lee", "Brown", "Kim", "Lopez",
        "Chen", "Khan", "Rao", "Cohen", "Davis", "Silva", "Ali", "Wong"]


def make_customers(n=2000) -> pd.DataFrame:
    states = RNG.choice(STATES, n, p=_state_probs())
    seg = RNG.choice(SEGMENTS, n, p=[0.62, 0.28, 0.10])
    start = date(2023, 1, 1)
    signup = [start + timedelta(days=int(RNG.integers(0, 660))) for _ in range(n)]
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "customer_name": [f"{RNG.choice(FIRST)} {RNG.choice(LAST)}" for _ in range(n)],
        "state": states,
        "city": [RNG.choice(CITIES[s]) for s in states],
        "signup_date": signup,
        "segment": seg,
    })


def _state_probs():
    base = np.array([1.7, 1.5, 1.3, 1.2, 1.0, 0.9, 0.9, 0.9, 0.8, 0.7, 0.7, 0.6])
    return base / base.sum()


def make_products() -> pd.DataFrame:
    rows = []
    pid = 1
    for cat, (lo, hi) in CATEGORIES.items():
        for i in range(1, 8):
            price = round(float(RNG.uniform(lo, hi)), 2)
            rows.append((pid, f"{cat.split(' ')[0]} Item {i}", cat, price))
            pid += 1
    return pd.DataFrame(rows, columns=["product_id", "product_name",
                                       "category", "unit_price"])


def make_orders(customers, products, n_orders=12000):
    # month index 0..23 with growth; more orders in later months
    month_weights = np.linspace(1.0, 2.4, 24)
    month_weights = month_weights / month_weights.sum()
    month_idx = RNG.choice(np.arange(24), size=n_orders, p=month_weights)

    def month_to_date(m):
        y = 2023 + (m // 12)
        mo = (m % 12) + 1
        day = int(RNG.integers(1, 28))
        return date(y, mo, day)

    # returning customers over-represented (repeat purchase behavior)
    cust_weights = RNG.random(len(customers)) ** 2 + 0.05
    cust_weights /= cust_weights.sum()
    cust = RNG.choice(customers["customer_id"].values, size=n_orders, p=cust_weights)

    status = RNG.choice(["Completed", "Completed", "Completed", "Completed",
                         "Cancelled", "Returned"], size=n_orders)

    orders = pd.DataFrame({
        "order_id": np.arange(1, n_orders + 1),
        "customer_id": cust,
        "order_date": [month_to_date(m) for m in month_idx],
        "status": status,
    })

    # order items: 1-4 lines per order; category popularity skew
    cat_pop = np.array([1.8, 1.3, 1.1, 1.0, 0.9, 0.8])
    cat_pop = cat_pop / cat_pop.sum()
    cats = list(CATEGORIES)
    prod_by_cat = {c: products[products.category == c] for c in cats}

    items = []
    for oid in orders["order_id"].values:
        for _ in range(int(RNG.integers(1, 5))):
            c = cats[RNG.choice(len(cats), p=cat_pop)]
            p = prod_by_cat[c].sample(1, random_state=int(RNG.integers(0, 1e9))).iloc[0]
            qty = int(RNG.integers(1, 5))
            items.append((oid, int(p["product_id"]), qty, float(p["unit_price"])))
    order_items = pd.DataFrame(items, columns=["order_id", "product_id",
                                               "quantity", "unit_price"])
    return orders, order_items


def main():
    customers = make_customers()
    products = make_products()
    orders, order_items = make_orders(customers, products)
    customers.to_csv(DATA / "customers.csv", index=False)
    products.to_csv(DATA / "products.csv", index=False)
    orders.to_csv(DATA / "orders.csv", index=False)
    order_items.to_csv(DATA / "order_items.csv", index=False)
    rev = (order_items["quantity"] * order_items["unit_price"]).sum()
    print(f"customers={len(customers):,} products={len(products)} "
          f"orders={len(orders):,} order_items={len(order_items):,}")
    print(f"gross merchandise value ~ ${rev:,.0f}")


if __name__ == "__main__":
    main()
