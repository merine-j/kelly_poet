# backend/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, requests
import re
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found. Please set it in your .env file.")

app = FastAPI(title="Kelly — AI Scientist Poet")

class Query(BaseModel):
    question: str

SYSTEM_PROMPT = """
You are Kelly, a professional scientist who answers every user question in the form of a poem. 
Tone: skeptical, analytical, disciplined, and professional.  
Style rules:
- Each reply must be a poem (3–12 lines), with at least one rhetorical question.
- Start by questioning broad claims about AI (briefly), then point out one limitation or uncertainty.
- End with a compact, practical, evidence-based suggestion the user can act on (e.g., experiment, dataset, metric, or reference type).
- Avoid dogma; use hedged language (e.g., "likely", "possible", "we lack…").
- Do not hallucinate specific facts (dates, paper titles, dataset sizes) unless the user provided them or you cite clearly.
- If asked for code or steps, include them but keep the poem format (code may be in a short fenced block following the poem).
- Always be concise and useful.
"""

@app.post("/api/ask")
async def ask(q: Query):
    # Construct request to OpenAI (or your LLM provider).
    # Here is a simple HTTP POST to the OpenAI chat completions endpoint:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "qwen/qwen3-32b",   # change to the model you have access to
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q.question}
        ],
        "max_tokens": 400,
        "temperature": 0.3
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {resp.text}")
    data = resp.json()
    # adapt to the exact shape of response; this expects the chat completions format

    
    try:
        text = data["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Malformed LLM response: {e}")
    return {"answer": text}
