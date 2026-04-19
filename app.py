import os, json
import numpy as np
import pandas as pd
import gradio as gr

print("Starting Advisor Copilot...")
print(f"PORT env: {os.environ.get('PORT', 'not set')}")

# ── Find data files ──
BASE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(BASE, "clients.csv")):
    DATA_DIR = BASE
elif os.path.exists(os.path.join(BASE, "data", "clients.csv")):
    DATA_DIR = os.path.join(BASE, "data")
else:
    print("ERROR: Cannot find clients.csv")
    DATA_DIR = BASE

print(f"Data dir: {DATA_DIR}")

# ── Load Data ──
clients = pd.read_csv(os.path.join(DATA_DIR, "clients.csv"))
holdings = pd.read_csv(os.path.join(DATA_DIR, "holdings.csv"))
transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
print(f"Loaded {len(clients)} clients, {len(holdings)} holdings")

# ── Load knowledge base as plain text ──
kb_text = ""
for fname in ["kb_suitability_policy.txt", "kb_model_portfolios.txt", "kb_research_notes.txt", "kb_product_factsheets.txt"]:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            kb_text += f.read() + "\n\n"
print(f"Knowledge base loaded: {len(kb_text)} chars")

# ── Gemini LLM ──
USE_LLM = False
try:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if key:
        import google.generativeai as genai
        genai.configure(api_key=key)
        USE_LLM = True
        print("Gemini connected")
except Exception as e:
    print(f"Gemini failed: {e}")

SYSTEM = """You are Advisor Copilot, an AI for wealth management Relationship Managers.
Rules: Only use provided data/context. Cite sources like [Source: Doc Name]. Never invent numbers. No return guarantees. Be specific with dollar amounts and percentages."""

def call_llm(prompt):
    if USE_LLM:
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            r = model.generate_content(f"{SYSTEM}\n\n{prompt}",
                generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=2048))
            return r.text.strip()
        except Exception as e:
            return f"[API error: {e}]"
    return "**Demo mode** — add GOOGLE_API_KEY env var for live AI responses."

# ── Portfolio math ──
def snapshot(cid):
    c = clients[clients.client_id == cid].iloc[0].to_dict()
    h = holdings[holdings.client_id == cid].copy()
    h["mv"] = h.quantity * h.current_price
    h["pnl"] = h.mv - h.quantity * h.avg_cost
    total = h.mv.sum()
    h["wt"] = (h.mv / total * 100).round(1)
    by_class = h.groupby("asset_class")["mv"].sum().apply(lambda v: round(v/total*100,1)).to_dict()
    targets = {"Conservative_30_70":{"Equity":15,"Fixed Income":65,"Real Estate":10,"Cash":10},
               "Balanced_60_40":{"Equity":55,"Fixed Income":35,"Real Estate":5,"Cash":5},
               "Growth_90_10":{"Equity":85,"Fixed Income":10,"Real Estate":0,"Cash":5}}
    tgt = targets.get(c["model_portfolio"],{})
    drift = {k: round(by_class.get(k,0)-tgt.get(k,0),1) for k in tgt}
    return {"client":c,"mv":round(total,2),"pnl":round(h.pnl.sum(),2),
            "pnl_pct":round(h.pnl.sum()/(h.quantity*h.avg_cost).sum()*100,2),
            "alloc":by_class,"target":tgt,"drift":drift,
            "positions":h[["ticker","asset_class","quantity","current_price","mv","wt","pnl"]].round(2).to_dict("records"),
            "txns":transactions[transactions.client_id==cid].sort_values("date").tail(5).to_dict("records")}

def find_client(msg):
    msg = msg.lower()
    for cid in clients.client_id:
        if cid.lower() in msg: return cid
    for _,r in clients.iterrows():
        for p in r["name"].lower().split():
            if len(p)>2 and p in msg: return r["client_id"]
    return None

def book_overview():
    rows = []
    for cid in clients.client_id:
        s = snapshot(cid)
        mx = max((abs(v) for v in s["drift"].values()), default=0)
        rows.append({"Client":f"{s['client']['name']} ({cid})","Risk":s["client"]["risk_profile"],
                      "Value":f"${s['mv']:,.0f}","P&L":f"{s['pnl_pct']:+.1f}%",
                      "Drift":f"{mx:.1f}pp","Review":"YES" if mx>5 else "OK"})
    return f"**Full Book**\n```\n{pd.DataFrame(rows).to_string(index=False)}\n```"

# ── Chat handler ──
def chat(message, history):
    msg = message.strip().lower()
    if msg in ["help","/help"]:
        return ("**Advisor Copilot** — try:\n- 'Summary of Emily Tan'\n- 'Is C002 compliant?'\n"
                "- 'Trading strategy for Sofia'\n- 'Draft email for Daniel Lee'\n- 'Show all clients'\n\n"
                "Clients: Emily Tan (C001), Rajesh Kumar (C002), Sofia Martinez (C003), Daniel Lee (C004), Aisha Rahman (C005)")
    if msg in ["clients","all clients","show all clients","book"]:
        return book_overview()

    cid = find_client(message)
    if cid:
        s = snapshot(cid)
        data = json.dumps(s, indent=2, default=str)
        prompt = f"USER QUESTION: {message}\n\nCLIENT DATA:\n{data}\n\nKNOWLEDGE BASE:\n{kb_text[:3000]}"
        return f"**{s['client']['name']} ({cid})**\n\n{call_llm(prompt)}"
    else:
        prompt = f"USER QUESTION: {message}\n\nKNOWLEDGE BASE:\n{kb_text[:3000]}"
        return call_llm(prompt)

# ── Launch ──
print("Creating Gradio app...")
demo = gr.ChatInterface(
    fn=chat,
    title="Advisor Copilot — RAG Portfolio Assistant",
    description="Ask about any client's portfolio, compliance, or trading strategies. Type **help** for commands.",
    examples=["Show all clients","Summary of Emily Tan","Is C002 compliant?","Trading strategy for Sofia Martinez","Draft email for Daniel Lee"],
    theme=gr.themes.Soft(),
)

port = int(os.environ.get("PORT", 10000))
print(f"Launching on port {port}...")
demo.launch(server_name="0.0.0.0", server_port=port)
