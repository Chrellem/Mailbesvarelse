"""
Hybrid Streamlit prototype (KB-first variant)

- Loads an Excel knowledge base (xlsx) with columns: ID, Question, Answer
- If "KB-only" is enabled, the model is instructed to use ONLY the KB content when answering.
- Robust OpenAI wrapper: tries openai.OpenAI client (v1+), otherwise REST fallback using OPENAI_API_KEY.
- Debug button in the sidebar to verify OpenAI connectivity.
- When decision leads to manual review, an editable draft is produced (based on KB if KB-only is set).

Place this file in repo root as streamlit_app_hybrid.py and run:
  streamlit run streamlit_app_hybrid.py

Make sure to set OPENAI_API_KEY via Streamlit Secrets or environment variable.
"""
import os
import re
import json
import traceback
import ssl
import urllib.request
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from jinja2 import Template
from dotenv import load_dotenv

# Optional local embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Try import openai (may be v0.x or v1.x); ok if missing (we have REST fallback)
try:
    import openai
except Exception:
    openai = None

load_dotenv()

# ---------- Configuration ----------
REQUIREMENTS_CSV = os.getenv("REQUIREMENTS_CSV", "sheets/requirements_suggestions.csv")
TEMPLATES_CSV = os.getenv("TEMPLATES_CSV", "sheets/templates_suggestions.csv")

# Excel KB defaults (user said KB has 3 columns: ID, Question, Answer)
EXCEL_KB_PATH = os.getenv("EXCEL_KB_PATH", "sheets/faq_kb.xlsx")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
TEST_MODE = os.getenv("TEST_MODE", "true").lower() in ("1", "true", "yes")

HIGH_THRESHOLD = float(os.getenv("HIGH_SIM_THRESHOLD", "0.80"))
MID_THRESHOLD = float(os.getenv("MID_SIM_THRESHOLD", "0.65"))
KEYWORD_BOOST = float(os.getenv("KEYWORD_BOOST", "0.06"))
SPECIFICITY_BOOST = float(os.getenv("SPECIFICITY_BOOST", "0.01"))

st.set_page_config(page_title="Mail‑besvarelse (KB-first)", layout="wide")
st.title("Mail‑besvarelse Prototype — KB‑first")

# ---------- Sidebar / Secrets ----------
st.sidebar.header("Konfiguration")

st.sidebar.text_input("Requirements CSV", value=REQUIREMENTS_CSV, key="req_path")
st.sidebar.text_input("Templates CSV", value=TEMPLATES_CSV, key="tpl_path")
st.sidebar.text_input("Excel KB path (xlsx)", value=EXCEL_KB_PATH, key="excel_kb_path")
use_excel_kb_only = st.sidebar.checkbox("Begræns model til kun Excel KB (KB-only)", value=True, key="use_excel_kb_only")
st.sidebar.text_input("OpenAI model", value=OPENAI_MODEL, key="openai_model")

# Read OPENAI key from st.secrets (Streamlit) or environment (.env)
OPENAI_API_KEY = None
try:
    if isinstance(st.secrets, dict) and "OPENAI_API_KEY" in st.secrets:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    else:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# sanitize/trim
if OPENAI_API_KEY:
    OPENAI_API_KEY = OPENAI_API_KEY.strip()

# Show only presence / safe info
st.sidebar.write("OPENAI API key tilgængelig:", bool(OPENAI_API_KEY))
st.sidebar.write("OPENAI API key length (safe):", len(OPENAI_API_KEY) if OPENAI_API_KEY else 0)

# If openai module exists, try to set openai.api_key for backwards compat where relevant
if openai is not None and OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception:
        pass

# ---------- OpenAI compatibility + REST fallback ----------
def create_chat_completion(messages: List[Dict[str, str]],
                           model: str = None,
                           max_tokens: int = 400,
                           temperature: float = 0.0) -> Tuple[str, Any]:
    """
    Returns (text, raw_response).
    Strategy:
      1) If openai.OpenAI client is present (v1+), use client.chat.completions.create(...)
      2) Else use REST POST to https://api.openai.com/v1/chat/completions with OPENAI_API_KEY
    Raises RuntimeError on missing credentials or network/HTTP failure.
    """
    model = model or OPENAI_MODEL

    # Try new client if available
    if openai is not None:
        try:
            OpenAI = getattr(openai, "OpenAI", None)
            if OpenAI:
                client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                # Try to extract assistant text
                try:
                    text = resp.choices[0].message.content
                except Exception:
                    try:
                        text = resp["choices"][0]["message"]["content"]
                    except Exception:
                        text = str(resp)
                return text, resp
        except Exception:
            # proceed to REST fallback
            pass

    # REST fallback
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not available for REST fallback. Please set the key in st.secrets or env.")

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # Try requests if available
    try:
        import requests
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        j = r.json()
    except Exception:
        # urllib fallback
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, context=ctx, timeout=30) as fh:
                j = json.load(fh)
        except Exception as e:
            raise RuntimeError(f"REST call to OpenAI failed: {e}")

    try:
        text = j["choices"][0]["message"]["content"]
    except Exception:
        text = json.dumps(j)[:2000]
    return text, j

# ---------- Debug button ----------
def openai_debug_test():
    st.sidebar.markdown("### Debug: OpenAI connection")
    if st.sidebar.button("Test OpenAI connection"):
        with st.spinner("Tester OpenAI‑forbindelse…"):
            try:
                st.sidebar.write("OPENAI_API_KEY set:", bool(OPENAI_API_KEY))
                if not OPENAI_API_KEY:
                    st.error("OPENAI_API_KEY ikke sat i dette miljø. Tilføj den i .streamlit/secrets.toml eller som env.")
                    return
                messages = [{"role": "user", "content": "Ping"}]
                try:
                    text, resp = create_chat_completion(messages, model="gpt-3.5-turbo", max_tokens=1, temperature=0.0)
                except Exception as e:
                    st.error(f"OpenAI test fejlede: {type(e).__name__}: {e}")
                    tb = traceback.format_exc()
                    st.sidebar.text_area("Traceback (debug)", value=tb, height=300)
                    return
                st.success("OpenAI test OK — modtog svar.")
                try:
                    if isinstance(resp, dict):
                        keys = list(resp.keys())
                    else:
                        keys = [k for k in dir(resp)[:60]]
                except Exception:
                    keys = ["(kunne ikke læse respons keys)"]
                st.sidebar.json({"response_keys": keys})
            except Exception as e:
                st.error(f"OpenAI test fejlede: {type(e).__name__}: {e}")
                tb = traceback.format_exc()
                st.sidebar.text_area("Traceback (debug)", value=tb, height=300)

openai_debug_test()

# ---------- KB loading & scoring ----------
@st.cache_data(ttl=300)
def load_excel_kb(path: str) -> List[Dict[str, str]]:
    """
    Load Excel KB. Expects columns (case-insensitive): id, question, answer.
    Returns list of dict rows.
    """
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_excel(path, sheet_name=0, dtype=str).fillna("")
    except Exception as e:
        st.error(f"Fejl ved indlæsning af Excel KB '{path}': {e}")
        return []
    # normalize columns to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    # Attempt to map common names
    id_col = None
    q_col = None
    a_col = None
    for c in df.columns:
        if c in ("id", "identifier"):
            id_col = c
        if c.startswith("q") or "question" in c:
            q_col = c
        if c.startswith("a") or "answer" in c or "svar" in c:
            a_col = c
    # fallback to first 3 columns
    cols = list(df.columns)
    if id_col is None and len(cols) >= 1:
        id_col = cols[0]
    if q_col is None and len(cols) >= 2:
        q_col = cols[1]
    if a_col is None and len(cols) >= 3:
        a_col = cols[2]
    out = []
    for _, r in df.iterrows():
        out.append({
            "id": str(r.get(id_col, "")).strip(),
            "question": str(r.get(q_col, "")).strip(),
            "answer": str(r.get(a_col, "")).strip()
        })
    return out

def score_kb_rows(kb_rows: List[Dict[str, str]], incoming_text: str) -> List[Dict[str, str]]:
    """
    Simple token/substring based scoring. Returns rows sorted by descending relevance.
    """
    t = (incoming_text or "").lower()
    scored = []
    for r in kb_rows:
        combined = f"{r.get('question','')} {r.get('answer','')}".lower()
        # count unique tokens from combined that appear in incoming text
        tokens = set(re.findall(r'\w{3,}', combined))
        matched = sum(1 for tok in tokens if tok in t)
        # boost if question exact substring appears in incoming text
        qtext = (r.get('question') or "").strip().lower()
        if qtext and qtext in t:
            matched += 5
        scored.append((r, matched))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    # return only items with positive score first; if none positive return top N
    positives = [r for r, s in scored_sorted if s > 0]
    if positives:
        return positives
    return [r for r, s in scored_sorted][:6]

# ---------- (Minimal) other helper functions used in matching flow ----------
def embed_texts(model, texts):
    if model is None:
        return None
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def cosine_sim(a, b):
    if a is None or b is None:
        return np.array([])
    return np.dot(b, a)

def split_keywords(cell):
    return [k.strip().lower() for k in (cell or "").split(",") if k.strip()]

def keyword_matches_text(text, kw):
    if not kw:
        return False
    try:
        return re.search(r'(?i)\b' + re.escape(kw) + r'\b', text) is not None
    except re.error:
        return kw in text

def path_specificity(path):
    toks = (path or "").split("/")
    return sum(1 for t in toks if t and t != "*")

# ---------- OpenAI-based extractors and draft generation (use wrapper) ----------
def call_openai_slot_extractor(mail_text: str, candidate_rows: List[Dict[str, str]]):
    if not OPENAI_API_KEY and openai is None:
        return None
    ctx = []
    for r in candidate_rows:
        ctx.append({
            "id": r.get("id", ""),
            "path": r.get("path", ""),
            "requirements_cache": (r.get("requirements_cache","") or "")[:400],
            "requirements_url": r.get("requirements_url","")
        })
    system = (
        "Du er en extractor der udtrækker: country_code (ISO2), intent (one token), program (eller empty), "
        "applicant_name (if present), missing_info (list). Returner kun gyldig JSON."
    )
    user = (
        f"Mail content:\n'''{mail_text}'''\n\n"
        f"Candidate contexts (id/path/short):\n{json.dumps(ctx, ensure_ascii=False, indent=2)}\n\n"
        "Returner et enkelt JSON-objekt med felterne: country_code, intent, program, applicant_name, missing_info (array), confidence (0-1)."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        out_text, resp = create_chat_completion(messages, model=OPENAI_MODEL, max_tokens=300, temperature=0.0)
        try:
            return json.loads(out_text)
        except Exception:
            m = re.search(r'(\{[\s\S]*\})', out_text)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    return None
            return None
    except Exception as e:
        st.error(f"OpenAI call failed: {e}")
        return None

def generate_draft_reply(mail_text: str,
                         top_candidates: List[Tuple[Dict[str, str], float]],
                         templates_dict: Dict[str, Any],
                         openai_model: str = OPENAI_MODEL,
                         excel_kb: List[Dict[str, str]] = None,
                         kb_only: bool = False) -> Dict[str, Any]:
    """
    Generate a suggested reply.
    If kb_only==True and excel_kb present, instruct model to use ONLY KB content.
    Returns dict with keys: subject, body, confidence, notes
    """
    if not OPENAI_API_KEY and openai is None:
        return None

    ctx_lines = []
    for c in (top_candidates or [])[:4]:
        row = c[0] if isinstance(c, (list, tuple)) else c
        ctx_lines.append(f"- id:{row.get('id','')}, path:{row.get('path','')}, topic:{row.get('topic','')}, url:{row.get('requirements_url','')}")
    ctx_text = "\n".join(ctx_lines)

    kb_block = ""
    if kb_only and excel_kb:
        matched = score_kb_rows(excel_kb, mail_text)
        if not matched:
            matched = excel_kb[:6]
        kb_lines = []
        for r in matched[:8]:
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            # keep snippet length moderate
            if len(a) > 600:
                a = a[:600].rstrip() + " ..."
            kb_lines.append(f"KB id:{r.get('id','')} | Q: {q} | A: {a}")
        kb_block = "\n".join(kb_lines)

    base_system = (
        "Du er en hjælpsom assistent der skriver et kort, professionelt svar på en studenterhenvendelse. "
        "Returner KUN et JSON-objekt med felterne: subject, body, confidence (0-1), notes."
    )

    messages = []
    if kb_only:
        kb_sys = (
            "VIGTIGT: Brug KUN informationen i de medfølgende KB-poster (nedenfor). "
            "Du må IKKE tilføje, antage eller opfinde information ud over hvad KB indeholder. "
            "Hvis du ikke kan besvare spørgsmålet ud fra KB, returner subject empty and body: "
            "'Jeg kender ikke svaret ud fra de tilgængelige oplysninger.'"
        )
        messages.append({"role": "system", "content": kb_sys})

    messages.append({"role": "system", "content": base_system})

    if kb_block:
        messages.append({"role": "system", "content": f"KB entries for this query (use these, do not invent):\n{kb_block}"})

    user_text = f"Mail content:\n'''{mail_text}'''\n\n"
    if not kb_only and ctx_text:
        user_text += f"Top candidate contexts:\n{ctx_text}\n\n"
    user_text += (
        "Skriv et forslag til SUBJECT (max 80 chars) og BODY (kort, ~100-400 words) som et JSON objekt med felterne: "
        "subject, body, confidence, notes. BODY skal være venlig, indeholde references til KB entries hvis relevant."
    )
    messages.append({"role": "user", "content": user_text})

    try:
        out_text, resp = create_chat_completion(messages, model=openai_model, max_tokens=600, temperature=0.2)
        try:
            parsed = json.loads(out_text)
            # Ensure keys
            parsed.setdefault("subject", parsed.get("subject", "") or "")
            parsed.setdefault("body", parsed.get("body", "") or "")
            parsed.setdefault("confidence", float(parsed.get("confidence", 0) or 0))
            parsed.setdefault("notes", parsed.get("notes", "") or "")
            return parsed
        except Exception:
            m = re.search(r'(\{[\s\S]*\})', out_text)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                    return parsed
                except Exception:
                    return {"subject": "", "body": out_text, "confidence": 0.0, "notes": "Kunne ikke parse JSON; se body."}
            return {"subject": "", "body": out_text, "confidence": 0.0, "notes": "Ingen JSON fundet i modeloutput."}
    except Exception as e:
        return {"subject": "", "body": f"Fejl ved opkald til OpenAI: {e}", "confidence": 0.0, "notes": "OpenAI-fejl"}

# ---------- Load other CSVs (optional) ----------
@st.cache_data(ttl=120)
def load_requirements(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str).fillna("")
    return df

@st.cache_data(ttl=120)
def load_templates(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if "allowed_path_patterns" in df.columns:
        df["allowed_path_patterns"] = df["allowed_path_patterns"].apply(
            lambda s: [p.strip() for p in s.split(",") if p.strip()] if s else []
        )
    else:
        df["allowed_path_patterns"] = [[] for _ in range(len(df))]
    return df.set_index("template_id").to_dict(orient="index")

# ---------- Main UI flow ----------
req_path = st.session_state.get("req_path", REQUIREMENTS_CSV)
tpl_path = st.session_state.get("tpl_path", TEMPLATES_CSV)
excel_path = st.session_state.get("excel_kb_path", EXCEL_KB_PATH)

req_df = load_requirements(req_path)
templates = load_templates(tpl_path)
excel_kb = load_excel_kb(excel_path)

if req_df.empty:
    st.warning(f"Requirements CSV ikke fundet eller tom: {req_path}")
if not templates:
    st.warning(f"Templates CSV ikke fundet eller tom: {tpl_path}")
if not excel_kb:
    st.info(f"Excel KB ikke fundet eller tom: {excel_path} — app kan stadig bruge fallback flow.")

model = None
if SentenceTransformer is not None:
    try:
        model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception:
        model = None

st.subheader("Test input (hybrid, KB-first)")
with st.form("email_form"):
    subject = st.text_input("Subject", value="")
    body = st.text_area("Body", value="", height=200)
    applicant_name = st.text_input("Applicant name (valgfri)", value="")
    program = st.text_input("Program (valgfri)", value="")
    use_openai_checkbox = st.checkbox("Use OpenAI for slot extraction / draft", value=bool(OPENAI_API_KEY))
    submitted = st.form_submit_button("Preview svar")
if not submitted:
    st.info("Indtast subject/body og klik Preview.")
else:
    combined = f"{subject}\n\n{body}"
    st.markdown(f"**KB-only mode:** `{use_excel_kb_only}` — Excel KB path: `{excel_path}`")
    # Basic semantic matching optional (kept minimal)
    query_emb = None
    rows_index, index_embs = ([], None)
    if model is not None and not req_df.empty:
        try:
            query_emb = embed_texts(model, [combined])[0]
            texts = [" ".join([str(r.get("requirements_cache","")), str(r.get("path","")), str(r.get("question_keywords",""))]) for _, r in req_df.iterrows()]
            embs = embed_texts(model, texts)
            # simple semantic ranking
            sims = cosine_sim(query_emb, embs)
            candidates = []
            for i, score in enumerate(sims):
                r = req_df.iloc[i].to_dict()
                candidates.append((r, float(score)))
            candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
            topk = candidates_sorted[:6]
        except Exception:
            topk = []
    else:
        topk = []  # we'll still provide KB-based draft if KB-only

    # If there are no semantic candidates, we still continue to KB draft
    st.markdown("### Genereret forslag (hvis manual_review / no auto)")
    # Here we directly produce a draft when manual review desired
    # For prototype: always attempt to produce a draft using KB if KB-only or if no strong semantic candidate
    try:
        draft = None
        if use_openai_checkbox and (OPENAI_API_KEY or openai is not None):
            draft = generate_draft_reply(combined, topk, templates, openai_model=OPENAI_MODEL, excel_kb=excel_kb, kb_only=use_excel_kb_only)
        if draft:
            st.markdown("### Forslag (auto‑genereret udkast — rediger før afsendelse)")
            subj_suggestion = draft.get("subject", "") or "Vedr. din henvendelse"
            body_suggestion = draft.get("body", "") or ""
            st.text_input("Suggested subject (edit before sending):", value=subj_suggestion, key="manual_subj")
            st.text_area("Suggested body (edit before sending):", value=body_suggestion, height=300, key="manual_body")
            st.markdown(f"**Confidence:** {draft.get('confidence', 0):.2f}")
            if draft.get("notes"):
                st.info(f"Notes: {draft.get('notes')}")
            st.success("Rediger forslaget og kopier/brug det i din mailklient. App'en sender ikke automatisk.")
        else:
            # fallback: build a simple KB-based reply if KB is present
            if use_excel_kb_only and excel_kb:
                matched = score_kb_rows(excel_kb, combined)
                lines = []
                for r in matched[:4]:
                    lines.append(f"- {r.get('question','')}: {r.get('answer','')}")
                fallback_body = "Kære ansøger,\n\n" + "Se venligst nedenstående information fra vores KB:\n\n" + "\n".join(lines) + "\n\nHvis du har brug for yderligere hjælp, svar venligst på denne mail.\n\nMed venlig hilsen\nStudiekontoret"
                st.text_input("Suggested subject (edit):", value="Vedr. din henvendelse", key="fallback_subj")
                st.text_area("Suggested body (edit):", value=fallback_body, height=300, key="fallback_body")
            else:
                st.info("Ingen OpenAI‑udkast genereret eller OpenAI ikke konfigureret. Brug KB eller skriv manuelt.")
    except Exception as e:
        st.error(f"Fejl ved generering af udkast: {e}")
        tb = traceback.format_exc()
        st.text_area("Traceback", value=tb, height=300)

st.write("---")
st.caption("KB-first prototype: Excel KB (ID/Question/Answer) + OpenAI (REST fallback). Husk at sætte OPENAI_API_KEY i st.secrets eller som miljøvariabel.")
