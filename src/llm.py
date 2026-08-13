"""
The natural-language layer.

This turns a plain-English question into (a) a safe SQL query and (b) a short
business summary of the result. It is intentionally thin: a hosted model does
the language work via a prompt — there is no model training or fine-tuning here.

A key can come from two places:
  * an environment variable / Streamlit secret (OPENAI_API_KEY,
    ANTHROPIC_API_KEY, or GROQ_API_KEY), or
  * passed in at call time (creds=...) — this is how the app supports a visitor
    pasting their OWN key, which is kept only for their browser session and
    never stored.

If no key is available, the app falls back to the curated question bank so the
public demo still runs with zero cost or keys.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass

from db import SCHEMA
from question_bank import few_shot_examples


class NoLLM(Exception):
    """Raised when no provider key is available."""


@dataclass
class Creds:
    provider: str          # 'openai' | 'anthropic' | 'groq'
    key: str


def env_provider() -> str | None:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return None


def guess_provider(key: str) -> str:
    k = key.strip()
    if k.startswith("sk-ant"):
        return "anthropic"
    if k.startswith("gsk_"):
        return "groq"
    return "openai"          # OpenAI keys start with sk-


def _resolve(creds: Creds | None) -> Creds:
    """Prefer a caller-supplied key (visitor's own); else fall back to env."""
    if creds and creds.key:
        return creds
    p = env_provider()
    if p == "openai":
        return Creds("openai", os.getenv("OPENAI_API_KEY"))
    if p == "anthropic":
        return Creds("anthropic", os.getenv("ANTHROPIC_API_KEY"))
    if p == "groq":
        return Creds("groq", os.getenv("GROQ_API_KEY"))
    raise NoLLM("No LLM key available.")


SQL_SYSTEM = (
    "You are a careful analytics engineer. Given a database schema and a "
    "business question, return ONE DuckDB SQL SELECT query that answers it. "
    "Rules: read-only SELECT only (never INSERT/UPDATE/DELETE/DDL); realized "
    "revenue = SUM(order_items.quantity*order_items.unit_price) on Completed "
    "orders unless told otherwise; add a sensible LIMIT for 'top N'. "
    "Return ONLY the SQL, no prose, no code fences."
)


def _chat(messages, creds: Creds | None = None, max_tokens=500) -> str:
    c = _resolve(creds)
    if c.provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=c.key)
        r = client.chat.completions.create(model="gpt-4o-mini", messages=messages,
                                            temperature=0, max_tokens=max_tokens)
        return r.choices[0].message.content
    if c.provider == "groq":
        from openai import OpenAI          # Groq is OpenAI-compatible
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=c.key)
        r = client.chat.completions.create(model="llama-3.1-8b-instant",
                                           messages=messages, temperature=0,
                                           max_tokens=max_tokens)
        return r.choices[0].message.content
    import anthropic
    client = anthropic.Anthropic(api_key=c.key)
    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")
    r = client.messages.create(model="claude-3-5-haiku-20241022",
                               max_tokens=max_tokens, system=sys,
                               messages=[{"role": "user", "content": user}])
    return r.content[0].text


def _strip_sql(text: str) -> str:
    t = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
    return t.strip().strip(";")


def nl_to_sql(question: str, creds: Creds | None = None) -> str:
    prompt = (f"Schema:\n{SCHEMA}\n\n"
              f"Examples:\n{few_shot_examples()}\n\n"
              f"Q: {question}\nSQL:")
    out = _chat([{"role": "system", "content": SQL_SYSTEM},
                 {"role": "user", "content": prompt}], creds=creds)
    return _strip_sql(out)


def summarize(question: str, df, creds: Creds | None = None) -> str:
    preview = df.head(15).to_csv(index=False)
    prompt = (f"Question: {question}\n\nResult (CSV):\n{preview}\n\n"
              "Write 1-2 sentences a business stakeholder would want: the key "
              "number(s) and what they imply. No preamble.")
    return _chat([{"role": "system",
                   "content": "You explain analytics results plainly and briefly."},
                  {"role": "user", "content": prompt}],
                 creds=creds, max_tokens=160).strip()
