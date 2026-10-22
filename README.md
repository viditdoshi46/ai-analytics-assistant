<p align="center"><img src="assets/banner.svg" alt="Ask-Your-Data" width="100%"></p>

<p align="center">
  <a href="https://ai-analytics-assistant-viditdoshi46.streamlit.app"><img src="https://img.shields.io/badge/Live%20app-open-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live app"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/SQL-analysis-336791?style=for-the-badge&logo=postgresql&logoColor=white" alt="SQL">
</p>

<p align="center"><b><a href="https://ai-analytics-assistant-viditdoshi46.streamlit.app">Live app</a></b> &nbsp;·&nbsp; <a href="#">Back to top</a></p>

---

# Ask-Your-Data — Natural-Language Analytics Assistant

**The problem it solves:** analysts spend a big share of every week answering the same *"can you pull…?"* questions in SQL. It's a bottleneck for the business and low-leverage work for the analyst. This tool puts a **natural-language layer on top of the data** so a non-technical teammate can ask a question in plain English and get an answer — a chart plus a plain-English takeaway — without waiting in the analytics queue.

**What it does:** you type *"Which product categories drive the most revenue?"* → the app drafts the **SQL**, validates it's safe, runs it on the sales database, **charts** the result, and **explains** it in a sentence a stakeholder can act on.

> **Not an ML project.** No model is trained here. It uses a hosted language model through a prompt as the natural-language layer. The real work — and the part that makes it trustworthy — is the **data model, the SQL, and the governance** around what the tool is allowed to do.

**[▶ Live demo](#)** · the demo runs a curated set of business questions **live, with no API key**. Add your own key to ask anything.

---

## Why it's built the way it is

**Self-serve analytics, done responsibly.** Letting an LLM touch a database is only safe with guardrails, so:

- **Read-only by construction.** Every generated query is validated to be a **single `SELECT`** before it runs — no `INSERT/UPDATE/DELETE/DDL`, one statement, automatically row-limited. A model can *read* the business, never change it.
- **Schema-scoped, not data-exposed.** The model only ever sees the **table/column schema**, never the underlying rows.
- **Transparent.** The exact SQL that ran is shown for every answer — an analyst can verify or copy it. The tool assists; it doesn't hide the work.
- **Graceful without a key.** The public demo answers a **curated question bank** live (real SQL, real charts) so anyone can try it for free; those same questions double as few-shot examples that steer the model's SQL style when a key is present.

## The data

A small, realistic **e-commerce sales** database (a star schema you'd recognize in any BI tool):

```
customers(customer_id, customer_name, city, state, signup_date, segment)
products(product_id, product_name, category, unit_price)
orders(order_id, customer_id, order_date, status)   -- Completed / Cancelled / Returned
order_items(order_id, product_id, quantity, unit_price)
```

~2,000 customers, 12,000 orders, 30,000 order lines, with built-in trends so the insights are real: revenue grows month over month, a few categories dominate, and returning customers spend more.

## Example questions it answers

- "What is the monthly revenue trend?" → *Revenue grew 141% over 24 months.*
- "Which categories drive the most revenue?" → *Electronics is 64% of revenue.*
- "How much revenue is lost to cancelled/returned orders?" → *34% of gross order value.*
- "How does revenue split between new and returning customers?" → *Returning customers drive 84%.*

…plus top customers, average order value by segment, revenue by state, best-selling products, month-over-month growth, and repeat-purchase rate.

---

## Architecture

```
Plain-English question
        │
        ▼
 Language model (hosted, via prompt)  ──►  draft SQL
        │
        ▼
 Read-only SELECT guard  ──►  DuckDB (sales database)
        │
        ▼
 Chart + result table + plain-English takeaway   (Streamlit + Plotly)
```

- **SQL / DuckDB** — the analytical engine; the same queries run on any SQL warehouse.
- **Language model** — OpenAI, Anthropic, or Groq (free tier) — pluggable via one environment variable. Optional: without a key, the curated questions still run.
- **Streamlit + Plotly** — the self-serve front end.

## Run it

```bash
pip install -r requirements.txt
python src/make_data.py                 # build the sample database (once)
streamlit run app/streamlit_app.py      # launch the assistant
```

To enable free-form questions, set **one** key (env var or `.streamlit/secrets.toml`):
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GROQ_API_KEY` (free tier). Without one, the curated questions run live.

## Repo structure

```
ai-analytics-assistant/
├── README.md
├── requirements.txt
├── data/                       # sample sales database (CSV)
├── src/
│   ├── make_data.py            # generate the sample e-commerce data
│   ├── db.py                   # DuckDB loader + schema + read-only SQL guard
│   ├── question_bank.py        # curated questions/SQL (demo + few-shot)
│   └── llm.py                  # natural-language → SQL + summary (pluggable)
└── app/
    └── streamlit_app.py        # the assistant UI
```

## Skills demonstrated

SQL & data modeling (star schema, window functions) · turning business questions into queries · self-serve BI / stakeholder enablement · applied LLM prompting with **safety guardrails** (read-only, schema-scoped, transparent) · Streamlit/Plotly delivery.

---
*Built by Vidit Doshi.*
