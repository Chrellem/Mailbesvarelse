"""
streamlit_app_hybrid.py

KB-first prototype (Programmes.xlsx)
- Loads Programmes.xlsx (sheets/Programmes.xlsx) with columns: ID, Programme, Admission requiements
- Automatically detects programme from mail text and uses that programme's admission text as KB (KB-only)
- Robust OpenAI wrapper: tries openai.OpenAI client (v1+), otherwise REST fallback using OPENAI_API_KEY
- Debug button in sidebar to verify OpenAI connectivity
- Generates editable draft replies constrained to KB when requested
"""
from typing import List, Dict, Any, Tuple
import os
import re
import json
import ssl
import urllib.request
import traceback

import numpy as np
import pandas as pd
import streamlit as st
from jinja2 import Template
from dotenv import load_dotenv

# Optional local embeddings (kept minimal for prototype)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Try import openai (may be absent or v0.x/v1.x). We have REST fallback.
try:
    import openai
except Exception:
    openai = None

load_dotenv()

# ---------------- Configuration ----------------
REQUIREMENTS_CSV = os.getenv("REQUIREMENTS_CSV", "sheets/requirements_suggestions.csv")
TEMPLATES_CSV = os.getenv("TEMPLATES_CSV", "sheets/templates_suggestions.csv")
EXCEL_KB_PATH = os.getenv("EXCEL_KB_PATH", "sheets/faq_kb.xlsx")
PROGRAMS_KB_PATH = os.getenv("PROGRAMS_KB_PATH", "sheets/Programmes.xlsx")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

HIGH_THRESHOLD = float(os.getenv("HIGH_SIM_THRESHOLD", "0.80"))
MID_THRESHOLD = float(os.getenv("MID_SIM_THRESHOLD", "0.65"))
KEYWORD_BOOST = float(os.getenv("KEYWORD_BOOST", "0.06"))
SPECIFICITY_BOOST = float(os.getenv("SPECIFICITY_BOOST", "0.01"))

st.set_page_config(page_title="Mail‑besvarelse (KB-first)", layout="wide")
st.title("Mail‑besvarelse Prototype — KB‑first")

# ---------------- Sidebar / Secrets ----------------
st.sidebar.header("Konfiguration")
st.sidebar.text_input("Requirements CSV", value=REQUIREMENTS_CSV, key="req_path")
st.sidebar.text_input("Templates CSV", value=TEMPLATES_CSV, key="tpl_path")
st.sidebar.text_input("Excel KB path (faq KB)", value=EXCEL_KB_PATH, key="excel_kb_path")
st.sidebar.text_input("Programmes KB path", value=PROGRAMS_KB_PATH, key="programs_kb_path")
use_excel_kb_only = st.sidebar.checkbox("Begræns model til kun Excel KB (KB-only)", value=True, key="use_excel_kb_only")
st.sidebar.text_input("OpenAI model", value=OPENAI_MODEL, key="openai_model")

# Read OPENAI key from st.secrets or environment
OPENAI_API_KEY = None
try:
    if isinstance(st.secrets, dict) and "OPENAI_API_KEY" in st.secrets:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    else:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if OPENAI_API_KEY:
    OPENAI_API_KEY = OPENAI_API_KEY.strip()

st.sidebar.write("OPENAI API key tilgængelig:", bool(OPENAI_API_KEY))
st.sidebar.write("OPENAI API key length (safe):", len(OPENAI_API_KEY) if OPENAI_API_KEY else 0)

if openai is not None and OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception:
        pass

# ---------------- OpenAI compatibility + REST fallback ----------------
def create_chat_completion(messages: List[Dict[str, str]],
                           model: str = None,
                           max_tokens: int = 400,
                           temperature: float = 0.0) -> Tuple[str, Any]:
    """
    Returns (text, raw_response).
    Strategy:
      1) If openai.OpenAI client present, use client.chat.completions.create(...)
      2) Otherwise REST POST to /v1/chat/completions using OPENAI_API_KEY
    Raises RuntimeError on missing credentials or network/HTTP failure.
    """
    model = model or OPENAI_MODEL

    # Try new OpenAI client if available
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
                try:
                    text = resp.choices[0].message.content
                except Exception:
                    try:
                        text = resp["choices"][0]["message"]["content"]
                    except Exception:
                        text = str(resp)
                return text, resp
        except Exception:
            # fallthrough to REST fallback
            pass

    # REST fallback
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not available for REST fallback. Set OPENAI_API_KEY in st.secrets or env.")

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

    # requests if available
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

# ---------------- Debug button ----------------
def openai_debug_test():
    st.sidebar.markdown("### Debug: OpenAI connection")
    if st.sidebar.button("Test OpenAI connection"):
        with st.spinner("Tester OpenAI‑forbindelse…"):
            try:
                st.sidebar.write("OPENAI_API_KEY set:", bool(OPENAI_API_KEY))
                if not OPENAI_API_KEY:
                    st.error("OPENAI_API_KEY ikke sat i dette miljø. Tilføj i st.secrets eller env.")
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

# ---------------- KB loaders & program helpers ----------------
@st.cache_data(ttl=300)
def load_excel_kb(path: str) -> List[Dict[str, str]]:
    """Load a generic KB xlsx/csv (ID, Question, Answer) - falls back to CSV if openpyxl missing."""
    if not os.path.exists(path):
        return []
    try:
        if path.lower().endswith(".xlsx"):
            df = pd.read_excel(path, sheet_name=0, dtype=str, engine="openpyxl").fillna("")
        else:
            df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        # fallback csv
        base, _ = os.path.splitext(path)
        csv_path = base + ".csv"
        st.error(f"Fejl ved indlæsning af Excel KB '{path}': {e}")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, dtype=str).fillna("")
                st.info(f"Bruger fallback CSV: {csv_path}")
            except Exception as e2:
                st.error(f"Fallback til CSV fejlede: {e2}")
                return []
        else:
            return []
    df.columns = [c.strip().lower() for c in df.columns]
    # Map to id/question/answer
    id_col = None
    q_col = None
    a_col = None
    cols = list(df.columns)
    for c in df.columns:
        if c in ("id",):
            id_col = c
        if "question" in c or c.startswith("q"):
            q_col = c
        if "answer" in c or "svar" in c or c.startswith("a"):
            a_col = c
    if id_col is None and len(cols) >= 1:
        id_col = cols[0]
    if q_col is None and len(cols) >= 2:
        q_col = cols[1]
    if a_col is None and len(cols) >= 3:
        a_col = cols[2]
    out = []
    for _, r in df.iterrows():
        out.append({
            "id": str(r.get(id_col, "")).strip() if id_col else "",
            "question": str(r.get(q_col, "")).strip() if q_col else "",
            "answer": str(r.get(a_col, "")).strip() if a_col else ""
        })
    return out

@st.cache_data(ttl=300)
def load_programs_kb(path: str) -> List[Dict[str, str]]:
    """
    Load Programmes.xlsx (columns: ID, Programme, Admission requiements).
    Returns list of dicts with keys: id, programme, admission.
    """
    if not os.path.exists(path):
        return []
    try:
        if path.lower().endswith(".xlsx"):
            df = pd.read_excel(path, sheet_name=0, dtype=str, engine="openpyxl").fillna("")
        else:
            df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        st.error(f"Fejl ved indlæsning af programs KB '{path}': {e}")
        return []
    df.columns = [c.strip().lower() for c in df.columns]
    colmap = {c: orig for c, orig in zip(df.columns, df.columns)}
    # find best columns
    id_col = colmap.get("id") or (df.columns[0] if len(df.columns) >= 1 else None)
    prog_col = colmap.get("programme") or colmap.get("program") or (df.columns[1] if len(df.columns) >= 2 else None)
    admission_col = None
    for lc, orig in zip(df.columns, df.columns):
        if "admiss" in lc or "requi" in lc or "require" in lc:
            admission_col = orig
            break
    if not admission_col and len(df.columns) >= 3:
        admission_col = df.columns[2]
    out = []
    for _, r in df.iterrows():
        out.append({
            "id": str(r.get(id_col, "")).strip() if id_col else "",
            "programme": str(r.get(prog_col, "")).strip() if prog_col else "",
            "admission": str(r.get(admission_col, "")).strip() if admission_col else ""
        })
    return out

def detect_program_from_mail_simple(mail_text: str, programs_kb: List[Dict[str,str]], top_n: int = 3) -> List[Tuple[Dict[str,str], float]]:
    """Simple token/substr based detection. Returns list of (row, score)."""
    t = (mail_text or "").lower()
    if not t or not programs_kb:
        return []
    scored = []
    for r in programs_kb:
        pname = (r.get("programme") or "").lower()
        if not pname:
            continue
        score = 0.0
        if pname in t:
            score += 6.0
        tokens = set(re.findall(r'\w{3,}', pname))
        score += sum(1.0 for tok in tokens if tok in t)
        adm = (r.get("admission") or "").lower()
        if adm:
            adm_tokens = set(re.findall(r'\w{4,}', adm))
            score += sum(0.5 for tok in adm_tokens if tok in t)
        if score > 0:
            scored.append((r, float(score)))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    return scored_sorted[:top_n]

def select_program_rows(programme_name_or_id: str, programs_kb: List[Dict[str,str]]) -> List[Dict[str,str]]:
    """Return rows matching programme name or ID (case-insensitive)."""
    if not programme_name_or_id or not programs_kb:
        return []
    target = programme_name_or_id.strip().lower()
    matches = [r for r in programs_kb if (r.get("programme","").strip().lower() == target) or (r.get("id","").strip().lower() == target)]
    if not matches:
        matches = [r for r in programs_kb if target in (r.get("programme","").strip().lower()) or target in (r.get("id","").strip().lower())]
    return matches

# ---------------- Minimal helper functions reused ----------------
def embed_texts(model, texts):
    if model is None:
        return None
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def cosine_sim(a, b):
    if a is None or b is None:
        return np.array([])
    return np.dot(b, a)

# ---------------- OpenAI-based extractors & draft generation ----------------
def call_openai_slot_extractor(mail_text: str, candidate_rows: List[Dict[str,str]]):
    if not OPENAI_API_KEY and openai is None:
        return None
    ctx = []
    for r in candidate_rows or []:
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
                         top_candidates: List[Any],
                         templates_dict: Dict[str, Any],
                         openai_model: str = OPENAI_MODEL,
                         excel_kb: List[Dict[str,str]] = None,
                         kb_only: bool = False) -> Dict[str, Any]:
    """
    Generate a suggested reply.
    If kb_only==True and excel_kb present, instruct model to use ONLY KB content (admission text).
    excel_kb: list of rows (each with 'id','programme','admission' or generic KB rows)
    """
    if not OPENAI_API_KEY and openai is None:
        return None

    ctx_lines = []
    for c in (top_candidates or [])[:4]:
        row = c[0] if isinstance(c, (list, tuple)) else c
        ctx_lines.append(f"- id:{row.get('id','')}, path:{row.get('path','')}, topic:{row.get('topic','')}, url:{row.get('requirements_url','')}")
    ctx_text = "\n".join(ctx_lines)

    # If KB-only and excel_kb present, prepare KB block (admission text or answer)
    kb_block = ""
    if kb_only and excel_kb:
        kb_lines = []
        for r in excel_kb[:8]:
            if "admission" in r:
                admission_text = (r.get("admission") or "").strip()
                prog = (r.get("programme") or r.get("question") or "").strip()
                snippet = admission_text if len(admission_text) <= 600 else admission_text[:600].rstrip() + " ..."
                kb_lines.append(f"KB id:{r.get('id','')} | Programme: {prog} | Admission: {snippet}")
            else:
                a = (r.get("answer") or "").strip()
                q = (r.get("question") or "").strip()
                snippet = a if len(a) <= 600 else a[:600].rstrip() + " ..."
                kb_lines.append(f"KB id:{r.get('id','')} | Q: {q} | A: {snippet}")
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
            "Hvis du ikke kan besvare spørgsmålet ud fra KB, returner subject empty og body: 'Jeg kender ikke svaret ud fra de tilgængelige oplysninger.'"
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
        "subject, body, confidence, notes. BODY skal være venlig og indeholde references til KB entries hvis relevant."
    )
    messages.append({"role": "user", "content": user_text})

    try:
        out_text, resp = create_chat_completion(messages, model=openai_model, max_tokens=600, temperature=0.2)
        try:
            parsed = json.loads(out_text)
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

# ---------------- Load other CSVs (optional) ----------------
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

# ---------------- Main UI flow ----------------
req_path = st.session_state.get("req_path", REQUIREMENTS_CSV)
tpl_path = st.session_state.get("tpl_path", TEMPLATES_CSV)
excel_path = st.session_state.get("excel_kb_path", EXCEL_KB_PATH)
programs_kb_path = st.session_state.get("programs_kb_path", PROGRAMS_KB_PATH)

req_df = load_requirements(req_path)
templates = load_templates(tpl_path)
excel_kb = load_excel_kb(excel_path)
programs_kb = load_programs_kb(programs_kb_path)

if req_df.empty:
    st.warning(f"Requirements CSV ikke fundet eller tom: {req_path}")
if not templates:
    st.warning(f"Templates CSV ikke fundet eller tom: {tpl_path}")
if not programs_kb:
    st.info(f"Programmes KB ikke fundet eller tom: {programs_kb_path} — programdetektion vil ikke være aktiv.")

model = None
if SentenceTransformer is not None:
    try:
        model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception:
        model = None

st.subheader("Test input (KB-first)")
with st.form("email_form"):
    subject = st.text_input("Subject", value="")
    body = st.text_area("Body", value="", height=200)
    applicant_name = st.text_input("Applicant name (valgfri)", value="")
    # free text manual override inside form
    st.text_input("Manuelt valgt program (valgfri)", value="", key="program_manual_input")
    use_openai_checkbox = st.checkbox("Use OpenAI for slot extraction / draft", value=bool(OPENAI_API_KEY))
    submitted = st.form_submit_button("Preview svar")
if not submitted:
    st.info("Indtast subject/body og klik Preview.")
else:
    combined = f"{subject}\n\n{body}"
    # store last mail text for other flows
    st.session_state["last_mail_text"] = combined

    # ---------- Program detection & auto-selection ----------
    program_rows = None

    # Manual free-text override from form (if user filled it)
    program_manual = st.session_state.get("program_manual_input", "") or None

    if programs_kb:
        candidates = detect_program_from_mail_simple(combined, programs_kb, top_n=4)
    else:
        candidates = []

    # Auto-select top candidate if found and no manual override
    if program_manual:
        # user provided manual program within form -> use it
        st.session_state["selected_program"] = {"programme": program_manual, "rows": select_program_rows(program_manual, programs_kb)}
    elif candidates:
        top_row, top_score = candidates[0]
        # always auto-set the selected_program to top candidate (no confirmation required)
        st.session_state["selected_program"] = {"programme": top_row.get("programme"), "rows": select_program_rows(top_row.get("programme"), programs_kb)}
        st.markdown(f"**Foreslået program (auto‑valgt):** {top_row.get('programme')}  (score {top_score:.1f})")
    else:
        # no candidate found -> do nothing (user can manual select further down)
        st.info("Ingen tydelige program‑matches i Programmes.xlsx.")

    # Manual override selectbox outside form (persistent)
    all_programs = sorted({(p.get("programme") or "") for p in programs_kb if p.get("programme")})
    manual_choice = st.selectbox("Eller vælg manuelt et program fra KB (valgfrit):", options=[""] + all_programs, key="manual_program_choice")
    if manual_choice:
        st.session_state["selected_program"] = {"programme": manual_choice, "rows": select_program_rows(manual_choice, programs_kb)}
        st.success(f"Manuelt valgt program: {manual_choice}")

    # Obtain program_rows for generator
    if st.session_state.get("selected_program"):
        program_rows = st.session_state["selected_program"].get("rows", None)
        st.markdown(f"**Aktuelt valgt program:** {st.session_state['selected_program'].get('programme')} (rækker: {len(program_rows) if program_rows else 0})")

    # ---------- Minimal semantic matching for requirements CSV (optional) ----------
    topk = []
    if model is not None and not req_df.empty:
        try:
            query_emb = embed_texts(model, [combined])[0]
            texts = [" ".join([str(r.get("requirements_cache","")), str(r.get("path","")), str(r.get("question_keywords",""))]) for _, r in req_df.iterrows()]
            embs = embed_texts(model, texts)
            sims = cosine_sim(query_emb, embs)
            candidates_sem = []
            for i, score in enumerate(sims):
                r = req_df.iloc[i].to_dict()
                candidates_sem.append((r, float(score)))
            topk = sorted(candidates_sem, key=lambda x: x[1], reverse=True)[:6]
        except Exception:
            topk = []
    else:
        topk = []

    # ---------- Generate draft (uses program_rows if present) ----------
    st.markdown("### Genereret forslag (manual / preview)")
    try:
        draft = None
        if use_openai_checkbox and (OPENAI_API_KEY or openai is not None):
            if program_rows:
                draft = generate_draft_reply(combined, topk, templates, openai_model=OPENAI_MODEL, excel_kb=program_rows, kb_only=True)
            else:
                excel_kb_global = load_excel_kb(st.session_state.get("excel_kb_path", EXCEL_KB_PATH))
                draft = generate_draft_reply(combined, topk, templates, openai_model=OPENAI_MODEL, excel_kb=excel_kb_global, kb_only=use_excel_kb_only)
        else:
            draft = None

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
            # Fallback: show program KB snippets if program selected or KB-only selected
            if program_rows:
                lines = []
                for r in program_rows[:4]:
                    lines.append(f"- {r.get('programme')}: {r.get('admission')[:400]}")
                fallback_body = "Kære ansøger,\n\nSe venligst nedenstående information fra vores KB:\n\n" + "\n".join(lines) + "\n\nHvis du har brug for yderligere hjælp, svar venligst på denne mail.\n\nMed venlig hilsen\nStudiekontoret"
                st.text_input("Suggested subject (edit):", value="Vedr. din henvendelse", key="fallback_subj")
                st.text_area("Suggested body (edit):", value=fallback_body, height=300, key="fallback_body")
            elif use_excel_kb_only and excel_kb:
                matched = excel_kb[:4]
                lines = []
                for r in matched:
                    q = r.get("question") or ""
                    a = r.get("answer") or ""
                    lines.append(f"- {q}: {a[:300]}")
                fallback_body = "Kære ansøger,\n\nSe venligst nedenstående information fra vores KB:\n\n" + "\n".join(lines) + "\n\nMed venlig hilsen\nStudiekontoret"
                st.text_input("Suggested subject (edit):", value="Vedr. din henvendelse", key="fallback_subj2")
                st.text_area("Suggested body (edit):", value=fallback_body, height=300, key="fallback_body2")
            else:
                st.info("Ingen OpenAI‑udkast genereret eller OpenAI ikke konfigureret. Brug KB eller skriv manuelt.")
    except Exception as e:
        st.error(f"Fejl ved generering af udkast: {e}")
        tb = traceback.format_exc()
        st.text_area("Traceback", value=tb, height=300)

st.write("---")
st.caption("KB-first prototype: Programmes.xlsx (ID/Programme/Admission requiements) + OpenAI (REST fallback). Husk at sætte OPENAI_API_KEY i st.secrets eller som miljøvariabel.")
