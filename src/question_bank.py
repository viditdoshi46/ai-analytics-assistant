"""
Curated business questions with vetted SQL.

Two jobs:
  1. Powers the free public demo (no API key needed) — these run live.
  2. Serves as few-shot examples in the prompt so the language model writes
     SQL in the same house style when a user asks their own question.

Each item: question, sql (DuckDB), chart type, and a summarize(df) that writes
the plain-English takeaway a business user actually wants.
"""
from __future__ import annotations
import pandas as pd


def _money(x) -> str:
    return f"${x:,.0f}"


BANK = [
    {
        "q": "What is the monthly revenue trend?",
        "chart": "line",
        "sql": """
SELECT DATE_TRUNC('month', o.order_date) AS month,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'Completed'
GROUP BY 1 ORDER BY 1""",
        "summarize": lambda d: (
            f"Revenue grew from {_money(d['revenue'].iloc[0])} to "
            f"{_money(d['revenue'].iloc[-1])} across {len(d)} months — "
            f"a {(d['revenue'].iloc[-1]/d['revenue'].iloc[0]-1)*100:.0f}% increase, "
            "with a clear upward trend."),
    },
    {
        "q": "Which product categories drive the most revenue?",
        "chart": "bar",
        "sql": """
SELECT p.category,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'Completed'
GROUP BY 1 ORDER BY revenue DESC""",
        "summarize": lambda d: (
            f"{d['category'].iloc[0]} is the top category at "
            f"{_money(d['revenue'].iloc[0])} "
            f"({d['revenue'].iloc[0]/d['revenue'].sum()*100:.0f}% of revenue); "
            f"the top 2 categories account for "
            f"{d['revenue'].iloc[:2].sum()/d['revenue'].sum()*100:.0f}%."),
    },
    {
        "q": "Who are the top 10 customers by total spend?",
        "chart": "bar",
        "sql": """
SELECT c.customer_name, c.state,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spend
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'Completed'
GROUP BY 1, 2 ORDER BY total_spend DESC LIMIT 10""",
        "summarize": lambda d: (
            f"Your top customer, {d['customer_name'].iloc[0]} ({d['state'].iloc[0]}), "
            f"has spent {_money(d['total_spend'].iloc[0])}. The top 10 together "
            f"account for {_money(d['total_spend'].sum())}."),
    },
    {
        "q": "What is the average order value by customer segment?",
        "chart": "bar",
        "sql": """
WITH order_rev AS (
  SELECT o.order_id, o.customer_id,
         SUM(oi.quantity * oi.unit_price) AS rev
  FROM order_items oi JOIN orders o ON oi.order_id = o.order_id
  WHERE o.status = 'Completed'
  GROUP BY 1, 2)
SELECT c.segment,
       ROUND(AVG(rev), 2) AS avg_order_value,
       COUNT(*) AS orders
FROM order_rev r JOIN customers c ON r.customer_id = c.customer_id
GROUP BY 1 ORDER BY avg_order_value DESC""",
        "summarize": lambda d: (
            f"{d['segment'].iloc[0]} customers have the highest average order "
            f"value at {_money(d['avg_order_value'].iloc[0])}, versus "
            f"{_money(d['avg_order_value'].iloc[-1])} for {d['segment'].iloc[-1]}."),
    },
    {
        "q": "Which states generate the most revenue?",
        "chart": "bar",
        "sql": """
SELECT c.state,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'Completed'
GROUP BY 1 ORDER BY revenue DESC LIMIT 10""",
        "summarize": lambda d: (
            f"{d['state'].iloc[0]} leads with {_money(d['revenue'].iloc[0])}; the "
            f"top 3 states ({', '.join(d['state'].head(3))}) drive "
            f"{d['revenue'].head(3).sum()/d['revenue'].sum()*100:.0f}% of the top-10 revenue."),
    },
    {
        "q": "How much revenue is lost to cancelled and returned orders?",
        "chart": "bar",
        "sql": """
SELECT o.status,
       COUNT(DISTINCT o.order_id) AS orders,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS gross_value
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
GROUP BY 1 ORDER BY gross_value DESC""",
        "summarize": lambda d: (
            f"Cancelled + returned orders represent "
            f"{_money(d.loc[d['status'].isin(['Cancelled','Returned']),'gross_value'].sum())} "
            f"({d.loc[d['status'].isin(['Cancelled','Returned']),'gross_value'].sum()/d['gross_value'].sum()*100:.0f}% "
            "of gross order value) — a concrete leakage to investigate."),
    },
    {
        "q": "What are the top 5 best-selling products by units sold?",
        "chart": "bar",
        "sql": """
SELECT p.product_name, p.category, SUM(oi.quantity) AS units
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'Completed'
GROUP BY 1, 2 ORDER BY units DESC LIMIT 5""",
        "summarize": lambda d: (
            f"{d['product_name'].iloc[0]} ({d['category'].iloc[0]}) is the "
            f"best seller at {int(d['units'].iloc[0]):,} units."),
    },
    {
        "q": "What is the month-over-month revenue growth rate?",
        "chart": "line",
        "sql": """
WITH m AS (
  SELECT DATE_TRUNC('month', o.order_date) AS month,
         SUM(oi.quantity * oi.unit_price) AS revenue
  FROM order_items oi JOIN orders o ON oi.order_id = o.order_id
  WHERE o.status = 'Completed'
  GROUP BY 1)
SELECT month, ROUND(revenue, 2) AS revenue,
       ROUND(100*(revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM m ORDER BY month""",
        "summarize": lambda d: (
            f"Month-over-month growth averages "
            f"{d['mom_growth_pct'].dropna().mean():.1f}% — steady expansion "
            "rather than one-off spikes."),
    },
    {
        "q": "How does revenue split between new and returning customers?",
        "chart": "bar",
        "sql": """
WITH first_order AS (
  SELECT customer_id, MIN(order_date) AS first_dt FROM orders GROUP BY 1)
SELECT CASE WHEN o.order_date = f.first_dt THEN 'New' ELSE 'Returning' END AS customer_type,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue,
       COUNT(DISTINCT o.order_id) AS orders
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN first_order f ON o.customer_id = f.customer_id
WHERE o.status = 'Completed'
GROUP BY 1 ORDER BY revenue DESC""",
        "summarize": lambda d: (
            f"Returning customers drive "
            f"{d.loc[d['customer_type']=='Returning','revenue'].sum()/d['revenue'].sum()*100:.0f}% "
            "of revenue — retention is doing real work here."),
    },
    {
        "q": "Which customer segment has the highest repeat-purchase rate?",
        "chart": "bar",
        "sql": """
WITH cust_orders AS (
  SELECT o.customer_id, COUNT(DISTINCT o.order_id) AS n FROM orders o GROUP BY 1)
SELECT c.segment,
       ROUND(100.0*AVG(CASE WHEN co.n > 1 THEN 1 ELSE 0 END), 1) AS repeat_rate_pct,
       COUNT(*) AS customers
FROM cust_orders co JOIN customers c ON co.customer_id = c.customer_id
GROUP BY 1 ORDER BY repeat_rate_pct DESC""",
        "summarize": lambda d: (
            f"{d['segment'].iloc[0]} customers repeat the most "
            f"({d['repeat_rate_pct'].iloc[0]:.0f}% place more than one order)."),
    },
]


def few_shot_examples(n: int = 5) -> str:
    """Compact Q -> SQL pairs to steer the model's SQL style."""
    lines = []
    for item in BANK[:n]:
        lines.append(f"Q: {item['q']}\nSQL:\n{item['sql'].strip()}")
    return "\n\n".join(lines)
