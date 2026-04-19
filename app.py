"""
Advisor Copilot — Lightweight version for Render free tier (512MB RAM)
Uses TF-IDF instead of sentence-transformers to stay under memory limit.
"""

import os, json, math, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# ── Configuration ──
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
USE_LLM = False

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        USE_LLM = True
        print("Gemini API connected.")
    except Exception as e:
        print(f"Gemini setup failed: {e}")

# ── Load Data ──
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(DATA_DIR, "clients.csv")):
    DATA_DIR = os.path.join(DATA_DIR, "data")
clients = pd.read_csv(os.path.join(DATA_DIR, "clients.csv"))
holdings = pd.read_csv(os.path.join(DATA_DIR, "holdings.csv"))
transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
print(f"Data loaded: {len(clients)} clients, {len(holdings)} holdings")

# ── Lightweight RAG with TF-IDF (uses ~5MB instead of ~500MB) ──
KB_FILES = [
    ("Suitability Policy", os.path.join(DATA_DIR, "kb_suitability_policy.txt")),
    ("Model Portfolios", os.path.join(DATA_DIR, "kb_model_portfolios.txt")),
    ("Research Notes", os.path.join(DATA_DIR, "kb_research_notes.txt")),
    ("Product Factsheets", os.path.join(DATA_DIR, "kb_product_factsheets.txt")),
]

def chunk_text(text, size=400, overlap=80):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

docs = []
for source, path in KB_FILES:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    for j, ch in enumerate(chunk_text(raw)):
        docs.append({"source": source, "chunk_id": j, "text": ch})

# TF-IDF vectorizer (lightweight alternative to sentence-transformers)
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform([d["text"] for d in docs])
print(f"TF-IDF index built: {len(docs)} chunks")

def retrieve(query, k=4):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idxs = scores.argsort()[-k:][::-1]
    return [{"source": docs[i]["source"], "text": docs[i]["text"], "score": float(scores[i])}
            for i in top_idxs if scores[i] > 0]

def format_context(hits):
    return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)

# ── Portfolio Analytics ──
def portfolio_snapshot(client_id):
    c = clients[clients.client_id == client_id].iloc[0].to_dict()
    h = holdings[holdings.client_id == client_id].copy()
    h["market_value"] = h.quantity * h.current_price
    h["cost_basis"] = h.quantity * h.avg_cost
    h["pnl"] = h.market_value - h.cost_basis
    h["pnl_pct"] = (h.pnl / h.cost_basis) * 100
    total_mv = h.market_value.sum()
    h["weight_pct"] = (h.market_value / total_mv) * 100
    by_class = (h.groupby("asset_class")["market_value"].sum()
                .apply(lambda v: round(v / total_mv * 100, 1)).to_dict())
    targets = {
        "Conservative_30_70": {"Equity": 15, "Fixed Income": 65, "Real Estate": 10, "Cash": 10},
        "Balanced_60_40": {"Equity": 55, "Fixed Income": 35, "Real Estate": 5, "Cash": 5},
        "Growth_90_10": {"Equity": 85, "Fixed Income": 10, "Real Estate": 0, "Cash": 5},
    }
    target = targets.get(c["model_portfolio"], {})
    drift = {k: round(by_class.get(k, 0) - target.get(k, 0), 1) for k in target}
    recent_tx = (transactions[transactions.client_id == client_id]
                 .sort_values("date").tail(5).to_dict(orient="records"))
    limits = {"Conservative": 10, "Balanced": 15, "Aggressive": 20}
    limit = limits.get(c["risk_profile"], 15)
    conc_flags = [f"{r['ticker']} is {r['weight_pct']:.1f}% (limit: {limit}%)"
                  for _, r in h.iterrows() if r["weight_pct"] > limit]
    return {
        "client": c, "total_market_value": round(total_mv, 2),
        "total_pnl": round(h.pnl.sum(), 2),
        "total_pnl_pct": round(h.pnl.sum() / h.cost_basis.sum() * 100, 2),
        "allocation_pct_by_class": by_class, "target_allocation_pct": target,
        "drift_vs_target_pp": drift, "concentration_flags": conc_flags,
        "positions": h[["ticker","asset_class","sector","quantity","avg_cost",
                        "current_price","market_value","weight_pct","pnl","pnl_pct"]]
                     .round(2).to_dict(orient="records"),
        "recent_transactions": recent_tx,
    }

# ── LLM ──
SYSTEM_PROMPT = """You are Advisor Copilot, an AI assistant for licensed Relationship Managers at a wealth management firm.

RULES:
1. Only use the CLIENT DATA and RETRIEVED CONTEXT provided. Never invent numbers.
2. If a fact is not in the context, say "not available in the provided sources."
3. Cite sources inline like [Source: Document Name].
4. Never make forward-looking guarantees about returns.
5. Be specific with numbers.
6. For trading strategies, reference the firm's house view and suitability policy.
7. End client-facing drafts with the mandatory disclaimer.
"""

def call_llm(system, user):
    if USE_LLM:
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(
                f"SYSTEM INSTRUCTIONS:\n{system}\n\nUSER REQUEST:\n{user}",
                generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=2048))
            return resp.text.strip()
        except Exception as e:
            return f"[API error: {e}]\n\n" + _fallback(user)
    return _fallback(user)

def _fallback(user):
    return ("**DEMO MODE** (no API key)\n\n"
            "In production, the Gemini LLM generates a grounded response here.\n\n"
            "To enable: set GOOGLE_API_KEY environment variable with your free Gemini key.")

# ── Chatbot Logic ──
def detect_client(msg):
    msg = msg.lower()
    for cid in clients.client_id:
        if cid.lower() in msg:
            return cid
    for _, row in clients.iterrows():
        for part in row['name'].lower().split():
            if len(part) > 2 and part in msg:
                return row['client_id']
    return None

def detect_intent(msg):
    msg = msg.lower()
    if any(w in msg for w in ['summary','overview','tell me about','how is','performance','snapshot']): return 'summary'
    if any(w in msg for w in ['suitable','suitability','compliance','breach','aligned']): return 'suitability'
    if any(w in msg for w in ['rebalance','trade','trading','buy','sell','strategy','recommendation']): return 'rebalance'
    if any(w in msg for w in ['email','draft','letter','communication','write to']): return 'client_email'
    if any(w in msg for w in ['all client','book','batch','everyone']): return 'batch_review'
    if any(w in msg for w in ['policy','rule','limit','concentration']): return 'policy_question'
    if any(w in msg for w in ['market','outlook','house view','macro']): return 'market_view'
    return 'general'

def build_book_overview():
    rows = []
    for cid in clients.client_id:
        s = portfolio_snapshot(cid)
        mx = max((abs(v) for v in s["drift_vs_target_pp"].values()), default=0)
        rows.append({"Client": f"{s['client']['name']} ({cid})", "Risk": s['client']['risk_profile'],
                      "Value": f"${s['total_market_value']:,.0f}", "P&L": f"{s['total_pnl_pct']:+.1f}%",
                      "Max Drift": f"{mx:.1f}pp", "Review": "YES" if mx > 5 else "OK"})
    return f"**Full Book Overview**\n\n```\n{pd.DataFrame(rows).to_string(index=False)}\n```"

def process_message(message, history):
    if message.strip().lower() in ['help', '/help']:
        return ("**Advisor Copilot** — Ask about any client by name or ID:\n"
                "- 'Summary of Emily Tan'\n- 'Is C002 compliant?'\n"
                "- 'Trading strategy for Sofia'\n- 'Draft email for Daniel Lee'\n"
                "- 'Show all clients'\n\n"
                "**Clients:** Emily Tan (C001), Rajesh Kumar (C002), Sofia Martinez (C003), "
                "Daniel Lee (C004), Aisha Rahman (C005)")

    if message.strip().lower() in ['clients','list clients','show all clients','all clients','book']:
        return build_book_overview()

    intent = detect_intent(message)
    client_id = detect_client(message)

    if client_id:
        snap = portfolio_snapshot(client_id)
        name = snap['client']['name']
        if intent == 'summary':
            hits = retrieve(f"performance review {snap['client']['risk_profile']}", k=3)
            prompt = f"TASK: Write a portfolio summary.\n\nCLIENT DATA:\n{json.dumps({k: snap[k] for k in ['client','total_market_value','total_pnl','total_pnl_pct','allocation_pct_by_class','drift_vs_target_pp','concentration_flags','recent_transactions']}, indent=2, default=str)}\n\nPOSITIONS:\n{json.dumps(snap['positions'], indent=2, default=str)}\n\nCONTEXT:\n{format_context(hits)}"
        elif intent == 'suitability':
            hits = retrieve(f"suitability policy concentration {snap['client']['risk_profile']}", k=4)
            prompt = f"TASK: Suitability check — PASS/FAIL with numbers.\n\nCLIENT DATA:\n{json.dumps({k: snap[k] for k in ['client','allocation_pct_by_class','target_allocation_pct','drift_vs_target_pp','concentration_flags','positions']}, indent=2, default=str)}\n\nCONTEXT:\n{format_context(hits)}"
        elif intent == 'rebalance':
            hits = retrieve(f"rebalancing {snap['client']['model_portfolio']} house view", k=4)
            prompt = f"TASK: Rebalancing plan with BUY/SELL actions and dollar amounts.\n\nCLIENT DATA:\n{json.dumps({k: snap[k] for k in ['client','total_market_value','allocation_pct_by_class','target_allocation_pct','drift_vs_target_pp','concentration_flags','positions']}, indent=2, default=str)}\n\nCONTEXT:\n{format_context(hits)}"
        elif intent == 'client_email':
            hits = retrieve("client communication guidance rebalancing", k=3)
            prompt = f"TASK: Draft a client email (max 200 words, include disclaimer).\n\nCLIENT DATA:\n{json.dumps({k: snap[k] for k in ['client','total_pnl_pct','allocation_pct_by_class','drift_vs_target_pp']}, indent=2, default=str)}\n\nCONTEXT:\n{format_context(hits)}"
        else:
            hits = retrieve(message, k=4)
            prompt = f"QUESTION: {message}\n\nCLIENT DATA:\n{json.dumps(snap, indent=2, default=str)}\n\nCONTEXT:\n{format_context(hits)}"
        return f"**{name} ({client_id})** — {intent.replace('_',' ').title()}\n\n{call_llm(SYSTEM_PROMPT, prompt)}"

    elif intent == 'batch_review':
        return build_book_overview()
    else:
        hits = retrieve(message, k=4)
        prompt = f"QUESTION: {message}\n\nCONTEXT:\n{format_context(hits)}"
        return call_llm(SYSTEM_PROMPT, prompt)

# ── Gradio App ──
EXAMPLES = [
    "Show all clients",
    "Give me a full summary of Emily Tan's portfolio",
    "Is Rajesh Kumar's portfolio compliant?",
    "What trading strategy for Sofia Martinez?",
    "Draft a client email for Daniel Lee",
    "What are the concentration limits?",
    "What is the current house view on equities?",
]

demo = gr.ChatInterface(
    fn=process_message,
    title="Advisor Copilot — RAG Portfolio Assistant",
    description="Ask about any client's portfolio, compliance, trading strategies, or firm policies. Type **help** for commands.",
    examples=EXAMPLES,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
