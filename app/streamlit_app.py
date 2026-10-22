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
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap');
  html, body, [class*="css"], .stMarkdown {font-family:'Inter', sans-serif;}
  code, pre {font-family:'Fira Code', monospace !important;}
  .stApp {background: radial-gradient(800px 420px at 85% -8%, rgba(99,102,241,.28), transparent 60%),
                      radial-gradient(640px 360px at -5% 12%, rgba(34,211,238,.13), transparent 60%), #0b1020;}
  /* generous top padding so nothing sits under Streamlit's fixed top toolbar */
  .block-container {max-width: 820px; padding-top: 4.5rem; padding-bottom: 6rem;}
  @keyframes rise {from {opacity:0; transform:translateY(12px);} to {opacity:1; transform:none;}}
  @keyframes glow {0%,100% {box-shadow:0 0 0 0 rgba(129,140,248,.0);} 50% {box-shadow:0 0 0 6px rgba(129,140,248,.14);}}

  .brand {display:flex; align-items:center; gap:14px; margin-bottom:2px; animation: rise .5s both;}
  .brand img.logo {width:48px; height:48px; border-radius:14px; display:block; flex:0 0 48px;
     box-shadow:0 10px 26px -8px rgba(99,102,241,.8);}
  .brand h1 {margin:0; font-family:'Sora', sans-serif; font-size:1.65rem; font-weight:800; letter-spacing:-.03em; line-height:1.1;
     background:linear-gradient(90deg,#fff,#a5b4fc 60%,#67e8f9); -webkit-background-clip:text; background-clip:text; color:transparent;}
  .brand .pill {font-size:.72rem; font-weight:600; padding:4px 11px; border-radius:999px;
     margin-left:auto; white-space:nowrap;}
  .pill-demo {background:rgba(129,140,248,.16); color:#c7d2fe; border:1px solid rgba(129,140,248,.4); animation: glow 2.6s infinite;}
  .pill-live {background:rgba(34,197,94,.15); color:#86efac; border:1px solid rgba(34,197,94,.4);}
  .subtitle {color:#9aa4d0; font-size:1rem; margin:10px 0 6px 2px; line-height:1.55; animation: rise .6s both;}

  .insight {background:linear-gradient(135deg, rgba(99,102,241,.22), rgba(34,211,238,.10));
            border:1px solid rgba(129,140,248,.35); border-left:4px solid #818cf8; color:#eef0ff;
            padding:13px 16px; border-radius:12px; font-size:1.02rem; margin-top:8px; line-height:1.55;}

  .stat {background:linear-gradient(160deg, #171e3d, #11172f); border:1px solid rgba(129,140,248,.25); border-radius:16px;
         padding:16px 14px; text-align:center; animation: rise .6s both; transition: transform .25s, border-color .25s;}
  .stat:hover {transform:translateY(-4px); border-color:rgba(129,140,248,.7);}
  .stat .n {font-family:'Sora', sans-serif; font-size:1.55rem; font-weight:700; line-height:1.1;
            background:linear-gradient(90deg,#c7d2fe,#67e8f9); -webkit-background-clip:text; background-clip:text; color:transparent;}
  .stat .l {font-size:.74rem; color:#8c97c7; text-transform:uppercase; letter-spacing:.1em; margin-top:6px; font-weight:600;}

  .lead {font-family:'Sora', sans-serif; font-weight:600; font-size:1.02rem; color:#dfe3ff; margin:18px 0 8px;}

  /* starter suggestions: chip-like cards */
  div[data-testid="stButton"] > button {
     text-align:left; border-radius:12px; font-weight:500; background:#141a33;
     border:1px solid rgba(129,140,248,.25); color:#dfe3ff; transition: all .2s;}
  div[data-testid="stButton"] > button:hover {border-color:#818cf8; background:#1b2347; transform:translateY(-2px);
     box-shadow:0 12px 24px -16px rgba(99,102,241,.9); color:#fff;}
  div[data-testid="stFormSubmitButton"] > button {background:linear-gradient(90deg,#6366f1,#8b5cf6); border:0; color:#fff; font-weight:600;}
  [data-testid="stChatMessage"] {background:rgba(20,26,51,.72); border:1px solid rgba(129,140,248,.16); border-radius:16px; animation: rise .45s both;}
  [data-testid="stForm"] {background:#11172f; border:1px solid rgba(129,140,248,.3); border-radius:16px;}
  section[data-testid="stSidebar"] {background:#0e1429; border-right:1px solid rgba(129,140,248,.18);}
</style>
""", unsafe_allow_html=True)

HOVER = dict(bgcolor="#141a33", bordercolor="#818cf8",
             font=dict(color="#e6e9ff", size=13))


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
# Logo: an inline data-URI <img> in one markdown call. It loads instantly (no
# network/reflow, so it doesn't trigger a scroll jump) and <img> is allowed by
# Streamlit's HTML sanitizer. Generous top padding keeps it clear of the toolbar.
import base64
_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" '
    'viewBox="0 0 46 46">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#4f46e5"/>'
    '<stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>'
    '<rect width="46" height="46" rx="12" fill="url(#g)"/>'
    '<g transform="translate(11,11)">'
    '<path d="M3.5 5.5A2 2 0 0 1 5.5 3.5h13A2 2 0 0 1 20.5 5.5v8'
    'A2 2 0 0 1 18.5 15.5H10l-4.2 3.8V15.5H5.5A2 2 0 0 1 3.5 13.5z" '
    'fill="none" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"/>'
    '<rect x="7.3" y="10.2" width="2.1" height="3.2" rx="0.5" fill="#fff"/>'
    '<rect x="10.9" y="7.8" width="2.1" height="5.6" rx="0.5" fill="#fff"/>'
    '<rect x="14.5" y="9.2" width="2.1" height="4.2" rx="0.5" fill="#fff"/>'
    '</g></svg>')
_LOGO_URI = ("data:image/svg+xml;base64,"
             + base64.b64encode(_LOGO_SVG.encode()).decode())
st.markdown(f"""
<div class="brand">
  <img class="logo" src="{_LOGO_URI}" alt="Ask-Your-Data logo">
  <h1>Ask-Your-Data</h1>
  {_pill}
</div>
<div class="subtitle">A natural-language analytics assistant. Ask about the
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
                      color_discrete_sequence=["#67e8f9"])
    else:
        d = df.sort_values(y)
        fig = px.bar(d, x=y, y=x, orientation="h")
        # solid indigo fill — a continuous scale fades to white on close
        # values and vanishes on a dark background
        fig.update_traces(marker_color="#818cf8", marker_cornerradius=6)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                      hoverlabel=HOVER, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color="#c7cdf0"),
                      xaxis=dict(gridcolor="rgba(129,140,248,.14)"),
                      yaxis=dict(gridcolor="rgba(129,140,248,.14)"))
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

    st.markdown(f'<div class="lead">Try one of {len(BANK)} example questions. '
                'They run instantly, no API key needed.</div>', unsafe_allow_html=True)
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

# ---------------------------------------------------------------- ask box ----
# A form (not st.chat_input): chat_input forces the page to auto-scroll to the
# bottom on every load — a known Streamlit behavior — which pushed the header
# off-screen. A form has no such side effect and clears itself on submit.
placeholder = ("Ask anything about the sales data…" if mode_llm
               else "Tap an example above, or add an API key to ask your own…")
with st.form("ask", clear_on_submit=True, border=True):
    _q = st.text_input("Ask the data", placeholder=placeholder,
                       label_visibility="collapsed")
    _sent = st.form_submit_button("Ask  ➤", type="primary")
if _sent and _q.strip():
    queue(_q.strip(), None)
    st.rerun()

st.caption("Built by Vidit Doshi · Natural-language analytics · SQL · Python · "
           "LLM (hosted) · Streamlit · read-only query governance")
