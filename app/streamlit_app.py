"""
Ask-Your-Data — a natural-language analytics assistant.

A business user types a question in plain English; the app turns it into SQL,
runs it against the sales database, shows a chart, and writes the takeaway.
The goal is self-serve analytics: let non-technical teammates answer their own
data questions without waiting on the analytics queue.

Run:  streamlit run app/streamlit_app.py
(No API key needed for the demo — suggested questions run live. Add a key to
ask anything; see the sidebar.)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from db import get_con, run_sql, SCHEMA
from question_bank import BANK
import llm

st.set_page_config(page_title="Ask-Your-Data | Vidit Doshi",
                   layout="wide", page_icon="💬")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1150px;}
  .hero {background: linear-gradient(120deg, #0b6b5b 0%, #14b8a6 100%);
         color:#fff; padding:26px 30px; border-radius:14px; margin-bottom:8px;}
  .hero h1 {color:#fff; margin:0; font-size:1.85rem; font-weight:700;}
  .hero p {color:#d7f5ef; margin:6px 0 0 0; font-size:1.02rem;}
  .hero .by {color:#a9e4da; font-size:0.9rem; margin-top:10px;}
  /* subtle card only — no background/text overrides, so labels + values
     stay readable in BOTH light and dark themes */
  div[data-testid="stMetric"] {border:1px solid rgba(20,184,166,.30);
      border-radius:12px; padding:14px 16px;}
  .insight {background:#f2fbf9; border-left:4px solid #14b8a6;
            padding:14px 16px; border-radius:8px; font-size:1.05rem;}
</style>
""", unsafe_allow_html=True)

HOVER = dict(bgcolor="white", bordercolor="#cdece6",
             font=dict(color="#0b6b5b", size=13))


@st.cache_resource
def connection():
    return get_con()


con = connection()

# ---- credentials: a Streamlit secret you set, OR a visitor's own key ----
with st.sidebar:
    st.subheader("🔑 Ask your own questions")
    st.caption("The suggested questions run free with no key. To ask any "
               "free-form question, paste your own API key — it's kept only for "
               "this browser session and never stored or logged.")
    _prov = st.selectbox("Provider", ["OpenAI", "Groq (free tier)", "Anthropic"])
    _key = st.text_input("API key", type="password",
                         placeholder="stays in your session — never saved")

_PROV = {"OpenAI": "openai", "Groq (free tier)": "groq", "Anthropic": "anthropic"}
user_creds = llm.Creds(_PROV[_prov], _key.strip()) if _key.strip() else None
mode_llm = (user_creds is not None) or (llm.env_provider() is not None)

st.markdown("""
<div class="hero">
  <h1>💬 Ask-Your-Data — Natural-Language Analytics Assistant</h1>
  <p>Ask a question about the sales data in plain English. It writes the SQL,
     runs it, charts the answer, and explains what it means — so anyone on the
     team can self-serve without waiting on the analytics queue.</p>
  <div class="by">Built by Vidit Doshi · SQL · Python · LLM · Streamlit</div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️  About this project", expanded=True):
    st.markdown("""
**The problem.** Analysts spend a large share of their week answering the same
ad-hoc "can you pull…?" questions in SQL. That's a bottleneck for the business
and low-leverage work for the analyst.

**The idea.** A thin natural-language layer on top of the data warehouse: a
teammate asks in plain English, a language model drafts the SQL, and the app
runs it and explains the result. The analyst curates the data model and the
guardrails; the tool handles the routine questions.

**Guardrails (why this is safe).** Every generated query is validated to be a
**read-only SELECT** before it runs — no writes, one statement, row-limited.
The model only ever sees the schema, never the raw data.

*No model was trained here — this uses a hosted language model through a prompt.
The engineering that matters is the data model, the SQL, and the governance.*
""")

# ---------------- helpers ----------------
def pick_chart(df: pd.DataFrame, chart: str):
    if chart == "table" or df.shape[1] < 2 or len(df) < 2:
        return None
    x = df.columns[0]
    ycands = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
    if not ycands:
        return None
    y = ycands[0]
    if chart == "line":
        fig = px.line(df, x=x, y=y, markers=True)
    else:
        d = df.sort_values(y)
        fig = px.bar(d, x=y, y=x, orientation="h",
                     color=y, color_continuous_scale="Teal")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10),
                      coloraxis_showscale=False, hoverlabel=HOVER)
    return fig


def _generic_summary(df: pd.DataFrame) -> str:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num or len(df) == 0:
        return "Here are the results."
    y = num[0]
    top = df.sort_values(y, ascending=False).iloc[0]
    label = top[df.columns[0]]
    return f"Top result: {label} at {top[y]:,.0f} ({y})."


def answer(question, sql, chart, summarize_fn, source, creds=None):
    st.markdown(f"#### {question}")
    try:
        df = run_sql(con, sql)
    except Exception as exc:
        st.error(f"Could not run that safely: {exc}")
        return
    with st.expander("🔎  SQL that ran"):
        st.code(sql.strip(), language="sql")
    if len(df) == 0:
        st.warning("That query returned no rows — try a different question or "
                   "widen the time range.")
        return
    fig = pick_chart(df, chart)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, hide_index=True, use_container_width=True)
    try:
        text = summarize_fn(df) if summarize_fn is not None \
            else llm.summarize(question, df, creds=creds)
    except Exception:
        text = _generic_summary(df)
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)
    st.caption("Demo answer from the curated question set."
               if source == "demo" else "Answer generated live by the language model.")


def find_bank(q: str):
    ql = q.lower().strip()
    for item in BANK:
        if ql and ql in item["q"].lower():
            return item
    return None


tab_ask, tab_data = st.tabs(["💬  Ask the data", "🗂️  Dataset"])

# ================= ASK TAB =================
with tab_ask:
    kpis = run_sql(con, """
    SELECT ROUND(SUM(oi.quantity*oi.unit_price),0) AS revenue,
           COUNT(DISTINCT o.order_id) AS orders,
           COUNT(DISTINCT o.customer_id) AS customers
    FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
    WHERE o.status='Completed'""").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completed revenue", f"${kpis['revenue']:,.0f}")
    c2.metric("Orders", f"{int(kpis['orders']):,}")
    c3.metric("Customers", f"{int(kpis['customers']):,}")
    c4.metric("Mode", "Live LLM" if mode_llm else "Demo",
              help="Demo = curated questions run with no API key. "
                   "Live LLM = your key is set, so any question can be asked.")

    st.caption(f"{len(BANK)} suggested questions run instantly with no API key. "
               "New to the data? Open the **Dataset** tab first.")
    cols = st.columns(2)
    for i, item in enumerate(BANK):
        if cols[i % 2].button(item["q"], key=f"chip{i}", use_container_width=True):
            st.session_state["sel"] = i
            st.session_state["typed"] = ""

    typed = st.text_input(
        "…or ask your own question",
        key="typed",
        placeholder=("e.g., Which category grew the most in the last 12 months?"
                     if mode_llm else
                     "Free-form questions need an API key — see the sidebar. "
                     "Meanwhile, tap a suggested question above."))
    ask = st.button("Ask", type="primary")

    if ask and typed:
        if mode_llm:
            with st.spinner("Thinking…"):
                try:
                    sql = llm.nl_to_sql(typed, creds=user_creds)
                    answer(typed, sql, "bar", None, "llm", creds=user_creds)
                except Exception as exc:
                    st.error(f"Could not answer that: {exc}")
        else:
            m = find_bank(typed)
            if m:
                answer(m["q"], m["sql"], m["chart"], m["summarize"], "demo")
            else:
                st.info("This live demo runs the curated questions above without "
                        "an API key. To ask any free-form question, add your key "
                        "in the sidebar and the assistant will write the SQL.")
    elif "sel" in st.session_state:
        it = BANK[st.session_state["sel"]]
        answer(it["q"], it["sql"], it["chart"], it["summarize"], "demo")
    else:
        st.info("👆 Tap a suggested question to see the assistant write SQL, "
                "chart the answer, and explain it.")

# ================= DATASET TAB =================
with tab_data:
    st.markdown(
        "**A small synthetic e-commerce database** — four related tables. "
        "Here's a live peek at each so you know what you can ask about. "
        "(This is sample data; the design works on any SQL warehouse.)")
    st.code(SCHEMA, language="text")
    _tables = [("customers", "Customers"), ("products", "Products"),
               ("orders", "Orders"), ("order_items", "Order items (line items)")]
    for tbl, lbl in _tables:
        n = int(run_sql(con, f"SELECT COUNT(*) AS n FROM {tbl}").iloc[0]["n"])
        st.markdown(f"**{lbl}** — {n:,} rows  ·  showing first 8")
        st.dataframe(run_sql(con, f"SELECT * FROM {tbl} LIMIT 8"),
                     hide_index=True, use_container_width=True)

# ---------------- sidebar ----------------
with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "1. You ask in plain English.\n"
        "2. A language model drafts **SQL** from the schema.\n"
        "3. The query is **validated as read-only** and run on DuckDB.\n"
        "4. You get a **chart + plain-English takeaway**.")
    st.caption("Groq offers a free API key (console.groq.com) if you want to try "
               "free-form questions at no cost.")
    st.subheader("The data model")
    st.code(SCHEMA, language="text")

st.caption("Built by Vidit Doshi · Natural-language analytics · SQL · Python · "
           "LLM (hosted) · Streamlit · read-only query governance")
