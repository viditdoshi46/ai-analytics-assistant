"""
Ask-Your-Data — a natural-language analytics assistant (chat interface).

A business user types a question in plain English; the app turns it into SQL,
runs it against the sales database, shows a chart, and writes the takeaway.
The goal is self-serve analytics: let non-technical teammates answer their own
data questions without waiting on the analytics queue.

Run:  streamlit run app/streamlit_app.py
(No API key needed for the demo — the example questions run live. Add a key to
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
                   layout="centered", page_icon="💬")

# ---------------------------------------------------------------- styling ----
st.markdown("""
<style>
  .block-container {max-width: 820px; padding-top: 1.6rem; padding-bottom: 6rem;}

  /* compact product-style header (no big banner — this is a chat app) */
  .brand {display:flex; align-items:center; gap:12px; margin-bottom:2px;}
  .brand .logo {
     width:46px; height:46px; border-radius:12px; flex:0 0 46px;
     display:flex; align-items:center; justify-content:center;
     background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);}
  .brand .logo svg {display:block;}
  .brand h1 {margin:0; font-size:1.4rem; font-weight:700; line-height:1.15;}
  .brand .pill {font-size:.72rem; font-weight:600; padding:2px 9px; border-radius:999px;
     margin-left:auto; white-space:nowrap;}
  .pill-demo {background:#eef2ff; color:#4f46e5;}
  .pill-live {background:#dcfce7; color:#15803d;}
  .subtitle {color:#94a3b8; font-size:.98rem; margin:8px 0 2px 3px;}

  /* the plain-English takeaway under each answer */
  /* fixed light bg + fixed dark text so it's readable in BOTH themes */
  .insight {background:#eef2ff; border-left:4px solid #6366f1; color:#1e1b4b;
            padding:12px 15px; border-radius:8px; font-size:1.02rem; margin-top:6px;}

  /* welcome mini-stats */
  .stat {border:1px solid rgba(99,102,241,.28); border-radius:12px;
         padding:12px 14px; text-align:center;}
  .stat .n {font-size:1.35rem; font-weight:700; line-height:1.1;}
  .stat .l {font-size:.78rem; color:#64748b; text-transform:uppercase;
            letter-spacing:.03em; margin-top:2px;}

  /* starter suggestion buttons: quieter, chip-like */
  div[data-testid="stButton"] > button {
     text-align:left; border-radius:10px; font-weight:500;}
</style>
""", unsafe_allow_html=True)

HOVER = dict(bgcolor="white", bordercolor="#d6d9f5",
             font=dict(color="#4338ca", size=13))


@st.cache_resource
def connection():
    return get_con()


con = connection()

# --------------------------------------------------- credentials & sidebar ----
with st.sidebar:
    st.subheader("🔑 Ask your own questions")
    st.caption("The example questions run free with no key. To ask any "
               "free-form question, paste your own API key — it's kept only for "
               "this browser session and never stored or logged.")
    _prov = st.selectbox("Provider", ["OpenAI", "Groq (free tier)", "Anthropic"])
    _key = st.text_input("API key", type="password",
                         placeholder="stays in your session — never saved")
    st.caption("Groq offers a free key (console.groq.com) to try free-form "
               "questions at no cost.")

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. You ask in plain English.\n"
        "2. A language model drafts **SQL** from the schema.\n"
        "3. The query is **validated as read-only** and run on DuckDB.\n"
        "4. You get a **chart + plain-English takeaway**.")

    with st.expander("📚 The data model"):
        st.code(SCHEMA, language="text")

    with st.expander("ℹ️ About this project"):
        st.markdown("""
**Problem.** Analysts spend much of the week on repetitive "can you pull…?"
SQL requests — a bottleneck for the business, low-leverage for the analyst.

**Idea.** A thin natural-language layer over the warehouse: a teammate asks in
plain English, a model drafts the SQL, the app runs it and explains the result.

**Guardrails.** Every generated query is validated as a **read-only SELECT**
before it runs — one statement, row-limited. The model only sees the schema,
never the raw data.

*No model was trained here — this uses a hosted model through a prompt. The
engineering that matters is the data model, the SQL, and the governance.*
""")

    if st.session_state.get("history"):
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

_PROV = {"OpenAI": "openai", "Groq (free tier)": "groq", "Anthropic": "anthropic"}
user_creds = llm.Creds(_PROV[_prov], _key.strip()) if _key.strip() else None
mode_llm = (user_creds is not None) or (llm.env_provider() is not None)

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------------------------ header ----
_pill = ('<span class="pill pill-live">● Live LLM</span>' if mode_llm
         else '<span class="pill pill-demo">● Demo mode</span>')
_LOGO_SVG = (
    '<svg width="26" height="26" viewBox="0 0 24 24" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<path d="M3.5 5.5A2 2 0 0 1 5.5 3.5h13A2 2 0 0 1 20.5 5.5v8'
    'A2 2 0 0 1 18.5 15.5H10l-4.2 3.8V15.5H5.5A2 2 0 0 1 3.5 13.5z" '
    'fill="none" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"/>'
    '<rect x="7.3" y="10.2" width="2.1" height="3.2" rx="0.5" fill="#fff"/>'
    '<rect x="10.9" y="7.8" width="2.1" height="5.6" rx="0.5" fill="#fff"/>'
    '<rect x="14.5" y="9.2" width="2.1" height="4.2" rx="0.5" fill="#fff"/>'
    '</svg>')
st.markdown(f"""
<div class="brand">
  <div class="logo">{_LOGO_SVG}</div>
  <h1>Ask-Your-Data</h1>
  {_pill}
</div>
<div class="subtitle">A natural-language analytics assistant — ask about the
sales data in plain English and get SQL, a chart, and the takeaway.</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers ----
def pick_chart(df: pd.DataFrame, chart: str):
    if chart == "table" or df.shape[1] < 2 or len(df) < 2:
        return None
    x = df.columns[0]
    ycands = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
    if not ycands:
        return None
    y = ycands[0]
    if chart == "line":
        fig = px.line(df, x=x, y=y, markers=True,
                      color_discrete_sequence=["#6366f1"])
    else:
        d = df.sort_values(y)
        fig = px.bar(d, x=y, y=x, orientation="h")
        # solid indigo fill — a continuous scale fades to white on close
        # values and vanishes on a dark background
        fig.update_traces(marker_color="#6366f1")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                      hoverlabel=HOVER)
    return fig


def _generic_summary(df: pd.DataFrame) -> str:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num or len(df) == 0:
        return "Here are the results."
    y = num[0]
    top = df.sort_values(y, ascending=False).iloc[0]
    return f"Top result: {top[df.columns[0]]} at {top[y]:,.0f} ({y})."


def find_bank(q: str):
    ql = q.lower().strip()
    for i, item in enumerate(BANK):
        if ql and ql in item["q"].lower():
            return i
    return None


def compute_turn(question: str, bank_idx=None) -> dict:
    """Run a question and return a self-contained turn dict to render later.
    Heavy work (SQL + any LLM call) happens once, here — not on every rerun."""
    # resolve to a curated item or a free-form LLM query
    if bank_idx is None and not mode_llm:
        bank_idx = find_bank(question)          # let demo users type an example

    if bank_idx is not None:
        item = BANK[bank_idx]
        question, sql, chart = item["q"], item["sql"], item["chart"]
        summarize_fn, source = item["summarize"], "demo"
    elif not mode_llm:
        return {"q": question, "info": (
            "This live demo runs the curated example questions without an API "
            "key. To ask any free-form question, add your key in the sidebar and "
            "the assistant will write the SQL for you.")}
    else:
        chart, summarize_fn, source = "bar", None, "llm"
        try:
            sql = llm.nl_to_sql(question, creds=user_creds)
        except Exception as exc:
            return {"q": question, "error": f"Could not answer that: {exc}"}

    try:
        df = run_sql(con, sql)
    except Exception as exc:
        return {"q": question, "sql": sql, "error": f"Could not run that safely: {exc}"}
    if len(df) == 0:
        return {"q": question, "sql": sql,
                "error": "That query returned no rows — try a different question "
                         "or widen the time range."}
    try:
        text = summarize_fn(df) if summarize_fn is not None \
            else llm.summarize(question, df, creds=user_creds)
    except Exception:
        text = _generic_summary(df)
    return {"q": question, "sql": sql, "df": df, "chart": chart,
            "text": text, "source": source}


def render_turn(t: dict, idx: int):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(t["q"])
    with st.chat_message("assistant", avatar="💬"):
        if t.get("error"):
            st.warning(t["error"])
        elif t.get("info"):
            st.info(t["info"])
        else:
            with st.expander("🔎 SQL that ran"):
                st.code(t["sql"].strip(), language="sql")
            fig = pick_chart(t["df"], t["chart"])
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"fig{idx}")
            st.dataframe(t["df"], hide_index=True, use_container_width=True)
            st.markdown(f'<div class="insight">💡 {t["text"]}</div>',
                        unsafe_allow_html=True)
            st.caption("Curated demo answer." if t["source"] == "demo"
                       else "Generated live by the language model.")


def queue(question: str, bank_idx=None):
    st.session_state.pending = (question, bank_idx)


def starter_grid(prefix: str):
    """Render the curated questions as clickable suggestions (two columns)."""
    cols = st.columns(2)
    for i, item in enumerate(BANK):
        cols[i % 2].button(item["q"], key=f"{prefix}{i}", use_container_width=True,
                           on_click=queue, args=(item["q"], i))


# ------------------------------------------------- process any queued input ----
if "pending" in st.session_state:
    q, idx = st.session_state.pop("pending")
    st.session_state.history.append(compute_turn(q, idx))

# ------------------------------------------------------------ welcome / thread ----
if not st.session_state.history:
    kpis = run_sql(con, """
    SELECT ROUND(SUM(oi.quantity*oi.unit_price),0) AS revenue,
           COUNT(DISTINCT o.order_id) AS orders,
           COUNT(DISTINCT o.customer_id) AS customers
    FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
    WHERE o.status='Completed'""").iloc[0]
    s1, s2, s3 = st.columns(3)
    s1.markdown(f'<div class="stat"><div class="n">${kpis["revenue"]:,.0f}</div>'
                f'<div class="l">Completed revenue</div></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="stat"><div class="n">{int(kpis["orders"]):,}</div>'
                f'<div class="l">Orders</div></div>', unsafe_allow_html=True)
    s3.markdown(f'<div class="stat"><div class="n">{int(kpis["customers"]):,}</div>'
                f'<div class="l">Customers</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f"**Try one of {len(BANK)} example questions** — they run "
                "instantly, no API key needed:")
    starter_grid("start")

    with st.expander("🗂️ New here? Peek at the data first"):
        st.caption("A small synthetic e-commerce database — four related tables. "
                   "This is sample data; the design works on any SQL warehouse.")
        st.code(SCHEMA, language="text")
        for tbl, lbl in [("customers", "Customers"), ("products", "Products"),
                         ("orders", "Orders"), ("order_items", "Order items")]:
            n = int(run_sql(con, f"SELECT COUNT(*) AS n FROM {tbl}").iloc[0]["n"])
            st.markdown(f"**{lbl}** — {n:,} rows · first 5")
            st.dataframe(run_sql(con, f"SELECT * FROM {tbl} LIMIT 5"),
                         hide_index=True, use_container_width=True)
else:
    for i, t in enumerate(st.session_state.history):
        render_turn(t, i)

    with st.expander("💡 More example questions"):
        starter_grid("more")

# ---------------------------------------------------------------- chat input ----
placeholder = ("Ask anything about the sales data…" if mode_llm
               else "Tap an example above, or add an API key to ask your own…")
prompt = st.chat_input(placeholder)
if prompt:
    queue(prompt, None)
    st.rerun()

st.caption("Built by Vidit Doshi · Natural-language analytics · SQL · Python · "
           "LLM (hosted) · Streamlit · read-only query governance")
