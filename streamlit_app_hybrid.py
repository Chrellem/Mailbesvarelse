"""
streamlit_app_hybrid.py

Complete Streamlit app with:
- KB loaders that read Excel/CSV (semikolon CSV support) for programmes and FAQ.
- Improved retrieval (fuzzy/TF-IDF fallback) and language detection.
- Split-into-subquestions strategy: answer each sub-question from KB when possible,
  return aggregated reply + provenance for each part.
- Feedback saved locally to feedback/feedback.csv (XLSX best-effort).
- Sidebar: reset feedback, download CSV/XLSX, reload assistant config.

Notes:
- If you previously installed rapidfuzz and scikit-learn, retrieval quality improves.
- Put OPENAI_API_KEY (if used) in environment or st.secrets.
- Keep Programmes.xlsx and faq_kb.xlsx in sheets/ or update sidebar paths.
"""
from typing import List, Dict, Any, Tuple
import os
import re
import json
import ssl
import urllib.request
import time
import hashlib
from datetime import datetime, timezone, timedelta
from collections import OrderedDict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# optional libs
try:
    import yaml
except Exception:
    yaml = None

try:
    import openai
except Exception:
    openai = None

# Optional retrieval libs
try:
    from rapidfuzz import fuzz
    RF_AVAILABLE = True
except Exception:
    RF_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKL_AVAILABLE = True
except Exception:
    SKL_AVAILABLE = False

load_dotenv()

# ---------------- Defaults / config ----------------
PROGRAMS_KB_DEFAULT = "sheets/Programmes.xlsx"
FAQ_KB_DEFAULT = "sheets/faq_kb.xlsx"
ASSISTANT_CONFIG_DEFAULT = "config/assistant_config.yaml"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
PROGRAM_DETECT_THRESHOLD = float(os.getenv("PROGRAM_DETECT_THRESHOLD", "2.5"))
FAQ_MATCH_TOPK = int(os.getenv("FAQ_MATCH_TOPK", "3"))
KB_SNIPPET_MAX_CHARS = 800

FEEDBACK_DIR = "feedback"
FEEDBACK_CSV = os.path.join(FEEDBACK_DIR, "feedback.csv")
FEEDBACK_XLSX = os.path.join(FEEDBACK_DIR, "feedback.xlsx")

# Exact feedback header/order
LEADING_COLUMNS = [
    "Mail", "Svar", "feedback_text", "rating",
    "assistant_instruction", "generated_subject", "program_detected", "timestamp_utc"
]
METADATA_COLUMNS = ["model", "temperature", "top_p", "confidence"]

st.set_page_config(page_title="Mailbesvarelse — KB-driven", layout="wide")
st.title("Mail‑besvarelse — KB‑driven (Program + FAQ)")

# ---------------- Sidebar: config UI ----------------
st.sidebar.header("Konfiguration")
st.sidebar.text_input("Programmes KB path", value=PROGRAMS_KB_DEFAULT, key="programs_kb_path")
st.sidebar.text_input("FAQ KB path", value=FAQ_KB_DEFAULT, key="faq_kb_path")
st.sidebar.text_input("Assistant config path (YAML)", value=ASSISTANT_CONFIG_DEFAULT, key="assistant_config_path")
st.sidebar.text_input("OpenAI model (env default)", value=DEFAULT_MODEL, key="openai_model")
st.sidebar.write("Program detection threshold:", PROGRAM_DETECT_THRESHOLD)

st.sidebar.markdown("### Assistant prompt (fallback)")
assistant_prompt_ui = st.sidebar.text_area(
    "Assistant prompt (fallback). Bruges hvis YAML ikke findes eller er tom.",
    value=st.session_state.get("assistant_prompt", ""),
    height=120,
    key="assistant_prompt_input"
)
if assistant_prompt_ui is not None:
    st.session_state["assistant_prompt"] = assistant_prompt_ui

# Reset feedback storage
if st.sidebar.button("Reset feedback storage (create new CSV with correct header)"):
    try:
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        cols = LEADING_COLUMNS + METADATA_COLUMNS
        pd.DataFrame(columns=cols).to_csv(FEEDBACK_CSV, index=False, encoding="utf-8")
        try:
            pd.DataFrame(columns=cols).to_excel(FEEDBACK_XLSX, index=False, engine="openpyxl")
        except Exception:
            pass
        st.sidebar.success(f"Feedback storage reset. New CSV at {FEEDBACK_CSV}")
    except Exception as e:
        st.sidebar.error(f"Could not reset feedback storage: {e}")

# Download buttons
if os.path.exists(FEEDBACK_CSV):
    try:
        with open(FEEDBACK_CSV, "rb") as fh:
            csv_bytes = fh.read()
        st.sidebar.download_button("Download feedback CSV", data=csv_bytes, file_name=os.path.basename(FEEDBACK_CSV), mime="text/csv")
    except Exception as e:
        st.sidebar.error(f"Kunne ikke forberede CSV download: {e}")
else:
    st.sidebar.write("Ingen feedback CSV fundet endnu.")

if os.path.exists(FEEDBACK_XLSX):
    try:
        with open(FEEDBACK_XLSX, "rb") as fh:
            xlsx_bytes = fh.read()
        st.sidebar.download_button("Download feedback XLSX", data=xlsx_bytes, file_name=os.path.basename(FEEDBACK_XLSX), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.sidebar.error(f"Kunne ikke forberede XLSX download: {e}")

if st.sidebar.button("Reload assistant config now"):
    try:
        load_assistant_config_with_mtime.clear()
        st.sidebar.success("Config cache ryddet.")
    except Exception:
        try:
            st.cache_data.clear()
            st.sidebar.success("st.cache_data cleared.")
        except Exception:
            st.sidebar.info("Genstart app for fuld reload.")

# ---------------- Read OPENAI key ----------------
OPENAI_API_KEY = None
try:
    if isinstance(st.secrets, dict):
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    else:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if OPENAI_API_KEY:
    OPENAI_API_KEY = OPENAI_API_KEY.strip()

st.sidebar.write("OPENAI key set:", bool(OPENAI_API_KEY))
if openai is not None and OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception:
        pass

# ---------------- YAML loader (mtime-aware) ----------------
@st.cache_data(ttl=3600)
def load_assistant_config_with_mtime(path: str, mtime: float) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    if yaml is None:
        return {"_parse_error": "PyYAML not installed"}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            docs = list(yaml.safe_load_all(fh))
    except Exception as e:
        return {"_parse_error": str(e)}
    merged: Dict[str, Any] = {}
    for d in docs:
        if not d:
            continue
        if not isinstance(d, dict):
            continue
        merged.update(d)
    out: Dict[str, Any] = {}
    instr = merged.get("instruction")
    out["instruction"] = str(instr).strip() if instr is not None else None
    temp = merged.get("temperature")
    try:
        out["temperature"] = float(temp) if temp is not None else None
    except Exception:
        out["temperature"] = None
    top_p = merged.get("top_p")
    try:
        out["top_p"] = float(top_p) if top_p is not None else None
    except Exception:
        out["top_p"] = None
    model = merged.get("model")
    out["model"] = str(model) if model is not None else None
    return out

def load_assistant_config(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = time.time()
    return load_assistant_config_with_mtime(path, mtime)

# ---------------- KB loaders (Excel/CSV tolerant) ----------------
def _read_table_file(path: str, sheet_name: Any = 0) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
            df = pd.read_excel(path, sheet_name=sheet_name, dtype=str, engine="openpyxl").fillna("")
        else:
            df = pd.read_csv(path, sep=";", dtype=str, engine="python", quotechar='"', keep_default_na=False).fillna("")
    except Exception:
        df = pd.read_csv(path, sep=";", dtype=str, engine="python", encoding="utf-8", quotechar='"', keep_default_na=False).fillna("")
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(ttl=300)
def load_programs_kb(path: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not path or not os.path.exists(path):
        return out
    df = _read_table_file(path)
    if df.empty:
        return out
    cols = {c.lower(): c for c in df.columns}
    def _col(name_variants):
        for n in name_variants:
            k = n.strip().lower()
            if k in cols:
                return df[cols[k]].astype(str).fillna("").tolist()
        return [""] * len(df)
    ids = _col(["id"])
    programmes = _col(["programme","program","title","name"])
    admissions = _col(["admission requiements","admission requirements","admission","admission_requirements"])
    keywords = _col(["keywords","keyword","tags"])
    short_desc = _col(["short description","short_description","short","description"])
    ects = _col(["ects","ects_credits"])
    degree = _col(["degree_type","degree","degree type"])
    duration = _col(["duration_years","duration"])
    location = _col(["location","locations","campus"])
    n = len(df)
    for i in range(n):
        out.append({
            "id": ids[i].strip(),
            "programme": programmes[i].strip(),
            "admission": admissions[i].strip(),
            "keywords": keywords[i].strip(),
            "short_description": short_desc[i].strip(),
            "ects": ects[i].strip(),
            "degree_type": degree[i].strip(),
            "duration_years": duration[i].strip(),
            "location": location[i].strip()
        })
    return out

@st.cache_data(ttl=300)
def load_faq_kb(path: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not path or not os.path.exists(path):
        return out
    df = _read_table_file(path)
    if df.empty:
        return out
    cols = {c.lower(): c for c in df.columns}
    def _val(i, variants):
        for v in variants:
            k = v.strip().lower()
            if k in cols:
                return str(df.iloc[i][cols[k]] or "")
        return ""
    for i in range(len(df)):
        out.append({
            "id": _val(i, ["id"]),
            "question": _val(i, ["question","question_text"]),
            "category": _val(i, ["category"]),
            "short_answer": _val(i, ["short_answer","short answer","short_answer"]),
            "answer": _val(i, ["full _answer","full_answer","full answer","answer"])
        })
    return out

# ---------------- Language detection ----------------
def detect_language(text: str) -> str:
    if not text:
        return "da"
    t = (text or "").lower()
    danish_tokens = [
        "hej", "tak", "venlig", "venligst", "hvad", "hvordan", "jeg", "ikke",
        "årsag", "bemærk", "svare", "svaret", "afdeling", "uddannelse", "studie",
        "tilmelding", "optagelse", "ansøg", "oplysninger", "dato", "tidspunkt"
    ]
    english_tokens = [
        "hello", "thanks", "please", "how", "what", "i", "not",
        "application", "admission", "program", "degree", "deadline",
        "when", "where", "please", "thank", "regards", "best"
    ]
    dan = sum(1 for tok in danish_tokens if tok in t)
    eng = sum(1 for tok in english_tokens if tok in t)
    return "en" if eng > dan else "da"

# ---------------- Retrieval helpers ----------------
def _word_tokens(s: str) -> List[str]:
    return re.findall(r'\w{3,}', (s or "").lower())

def token_overlap_ratio(a: str, b: str) -> float:
    atoks = set(_word_tokens(a))
    btoks = set(_word_tokens(b))
    if not atoks or not btoks:
        return 0.0
    return len(atoks & btoks) / max(1, len(atoks))

# ---------------- Improved detect_programs ----------------
def detect_programs(mail_text: str, programs_kb: List[Dict[str,str]], top_n: int = 3) -> List[Tuple[Dict[str,str], float]]:
    t = (mail_text or "").lower()
    scored: List[Tuple[Dict[str,str], float]] = []
    if not programs_kb:
        return scored
    for row in programs_kb:
        name = (row.get("programme") or row.get("program") or "").strip()
        kw = (row.get("keywords") or "").strip()
        short = (row.get("short_description") or "").strip()
        adm = (row.get("admission") or "").strip()
        score = 0.0
        if name and name.lower() in t:
            score += 8.0
        if RF_AVAILABLE and name:
            try:
                f = fuzz.token_set_ratio(name.lower(), t) / 100.0
                score += 6.0 * f
            except Exception:
                score += 3.0 * token_overlap_ratio(name, t)
        else:
            score += 3.0 * token_overlap_ratio(name, t)
        if RF_AVAILABLE and kw:
            try:
                f2 = fuzz.token_set_ratio(kw.lower(), t) / 100.0
                score += 3.0 * f2
            except Exception:
                score += 1.5 * token_overlap_ratio(kw, t)
        else:
            score += 1.5 * token_overlap_ratio(kw, t)
        score += 0.8 * token_overlap_ratio(short, t)
        score += 0.4 * token_overlap_ratio(adm, t)
        if score > 0:
            scored.append((row, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

# ---------------- Improved match_faq ----------------
def match_faq(mail_text: str, faq_kb: List[Dict[str,str]], top_k: int = 3) -> List[Tuple[Dict[str,str], float]]:
    t = (mail_text or "").strip()
    if not faq_kb:
        return []
    candidates = []
    for row in faq_kb:
        q = row.get("question", "") or ""
        short = row.get("short_answer", "") or ""
        full = row.get("answer", "") or ""
        candidates.append(" ".join([q, short, full]))
    scores = []
    if SKL_AVAILABLE and len(candidates) >= 1:
        try:
            vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
            X = vect.fit_transform(candidates)
            vq = vect.transform([t])
            sims = cosine_similarity(vq, X)[0]
            for idx, sim in enumerate(sims):
                if sim > 0:
                    scores.append((faq_kb[idx], float(sim)))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
        except Exception:
            pass
    for row in faq_kb:
        q = row.get("question", "") or ""
        full = row.get("answer", "") or ""
        score = 0.0
        score += 2.0 * token_overlap_ratio(q, t)
        score += 0.7 * token_overlap_ratio(full, t)
        cat = (row.get("category") or "").lower()
        if cat and cat in t:
            score += 0.8
        if RF_AVAILABLE and q:
            try:
                f = fuzz.token_set_ratio(q.lower(), t) / 100.0
                score += 1.5 * f
            except Exception:
                pass
        if score > 0:
            scores.append((row, float(score)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# ---------------- OpenAI wrapper (same as before) ----------------
def create_chat_completion(messages: List[Dict[str, str]],
                           model: str = None,
                           max_tokens: int = 400,
                           temperature: float = 0.0,
                           top_p: float = 1.0) -> Tuple[str, Any]:
    model = model or DEFAULT_MODEL
    if openai is not None:
        try:
            OpenAI = getattr(openai, "OpenAI", None)
            if OpenAI:
                client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                try:
                    text = resp.choices[0].message.content
                except Exception:
                    text = str(resp)
                return text, resp
        except Exception:
            pass
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not available for REST fallback.")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    try:
        import requests
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        j = r.json()
    except Exception:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, context=ctx, timeout=30) as fh:
                j = json.load(fh)
        except Exception as e:
            raise RuntimeError(f"REST call to OpenAI chat completions failed: {e}")
    try:
        text = j["choices"][0]["message"]["content"]
    except Exception:
        text = json.dumps(j)[:2000]
    return text, j

# ---------------- Split + per-part KB helpers (new) ----------------
# Tunables
MIN_KB_SCORE = 0.1
KB_MAX_CHARS_PER_PART = 1600
SUBQUESTION_MIN_LEN = 20

def split_into_subquestions(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    t = text.strip()
    parts = []
    if "?" in t:
        raw = re.split(r'(\?)', t)
        for i in range(0, len(raw), 2):
            sentence = raw[i].strip()
            qmark = raw[i+1] if i+1 < len(raw) else ""
            candidate = (sentence + qmark).strip()
            if candidate:
                parts.append(candidate)
    else:
        raw = re.split(r'\n+|•|- |;{1,2}', t)
        for r in raw:
            s = r.strip()
            if s:
                parts.append(s)
        if len(parts) <= 1:
            raw2 = re.split(r'(?<=[\.\!\?])\s+', t)
            parts = [p.strip() for p in raw2 if p.strip()]
    merged = []
    for p in parts:
        if len(p) < SUBQUESTION_MIN_LEN and merged:
            merged[-1] = (merged[-1] + " " + p).strip()
        else:
            merged.append(p)
    unique = []
    seen = set()
    for p in merged:
        q = " ".join(p.split())
        if len(q) >= 5 and q.lower() not in seen:
            unique.append(q)
            seen.add(q.lower())
    return unique

def build_kb_for_subquestion(subq: str,
                             programs_kb: List[Dict[str,str]],
                             faq_kb: List[Dict[str,str]],
                             top_prog: int = 2,
                             top_faq: int = 3) -> Tuple[str, List[Tuple[str,str]]]:
    kb_concat = ""
    used = []
    prog_hits = []
    try:
        prog_hits = detect_programs(subq, programs_kb, top_n=top_prog) if programs_kb else []
    except Exception:
        prog_hits = []
    for (row, score) in prog_hits:
        if score < MIN_KB_SCORE:
            continue
        rid = str(row.get("id") or row.get("ID") or (row.get("programme") or "")[:40])
        title = row.get("programme") or row.get("program") or ""
        adm = row.get("admission") or ""
        snippet = adm.strip()[:800]
        block = f"PROGRAM|{rid}|{title}\nADMISSION: {snippet}"
        if (rid, snippet) not in used:
            used.append((rid, snippet))
            kb_concat += f"\n---\nID:{rid}\n{block}\n"
        if len(kb_concat) > KB_MAX_CHARS_PER_PART:
            break
    faq_hits = []
    try:
        faq_hits = match_faq(subq, faq_kb, top_k=top_faq) if faq_kb else []
    except Exception:
        faq_hits = []
    for (row, score) in faq_hits:
        if score < MIN_KB_SCORE:
            continue
        rid = str(row.get("id") or row.get("ID") or "")
        q = (row.get("question") or "")[:200]
        a = (row.get("answer") or "")[:800]
        block = f"FAQ|{rid}|Q: {q}\nA: {a}"
        if (rid, a) not in used:
            used.append((rid, a))
            kb_concat += f"\n---\nID:{rid}\n{block}\n"
        if len(kb_concat) > KB_MAX_CHARS_PER_PART:
            break
    return kb_concat.strip(), used

def map_used_kb_ids_to_sources(used_ids: List[str],
                               programs_kb: List[Dict[str,str]],
                               faq_kb: List[Dict[str,str]]) -> List[Dict[str,str]]:
    sources = []
    if not used_ids:
        return sources
    prog_map = {}
    for r in (programs_kb or []):
        key = str(r.get("id") or r.get("ID") or r.get("programme","")).strip()
        if key:
            prog_map[key] = r
    faq_map = {}
    for r in (faq_kb or []):
        key = str(r.get("id") or r.get("ID") or "").strip()
        if key:
            faq_map[key] = r
    for uid in used_ids:
        u = str(uid)
        if u in prog_map:
            r = prog_map[u]
            sources.append({"id": u, "type": "program", "title": r.get("programme",""), "snippet": (r.get("admission") or "")[:300]})
        elif u in faq_map:
            r = faq_map[u]
            sources.append({"id": u, "type": "faq", "title": r.get("question",""), "snippet": (r.get("answer") or "")[:300]})
        else:
            sources.append({"id": u, "type": "unknown", "title": "", "snippet": ""})
    return sources

# ---------------- Split-aware generation (replaces previous single-shot generator) ----------------
def generate_combined_reply_split_parts(mail_text: str,
                                        program_rows: List[Dict[str,str]] = None,
                                        faq_rows: List[Dict[str,str]] = None,
                                        language: str = "da",
                                        openai_model: str = None,
                                        assistant_instruction: str = None,
                                        temperature: float = 0.0,
                                        top_p: float = 1.0) -> Dict[str, Any]:
    model = openai_model or DEFAULT_MODEL
    subqs = split_into_subquestions(mail_text)
    if not subqs:
        subqs = [mail_text.strip()] if mail_text else []
    parts_out = []
    confidences = []
    aggregate_subjects = []
    for idx, subq in enumerate(subqs, start=1):
        part = {"question": subq, "answer": "", "confidence": 0.0, "used_kb_ids": [], "used_sources": [], "status": "no_kb"}
        kb_concat, used_pairs = build_kb_for_subquestion(subq, program_rows or [], faq_rows or [])
        if not kb_concat:
            part["answer"] = "Jeg har ikke nok dokumenteret information i den tilgængelige vidensbase til at besvare dette delspørgsmål. Kan du give flere oplysninger?"
            part["status"] = "no_kb"
            part["confidence"] = 0.0
            parts_out.append(part)
            continue
        kb_ids = [str(u[0]) for u in used_pairs]
        lang_name = "dansk" if language == "da" else "english"
        base_system = (
            f"You are a strict KB-grounded assistant answering in {lang_name}. "
            "YOU MUST ONLY use facts present in the 'KB block' provided below. "
            "If the KB does not contain enough information to answer, respond with one short clarification question. "
            "You MUST return exactly one JSON object and nothing else, with keys: "
            "\"subject\" (string), \"body\" (string), \"confidence\" (0-1), \"notes\" (string), and \"used_kb_ids\" (array of ids). "
            "The 'body' must be fully supported by the KB and must not include invented content."
        )
        messages = [{"role": "system", "content": base_system}]
        if assistant_instruction:
            messages.append({"role": "assistant", "content": assistant_instruction})
        user_msg = f"Subquestion:\n'''{subq}'''\n\nKB block (only use information from here):\n{kb_concat}\n\nReturn only the JSON object specified."
        messages.append({"role": "user", "content": user_msg})
        try:
            out_text, resp = create_chat_completion(messages, model=model, max_tokens=600, temperature=temperature, top_p=top_p)
        except Exception as e:
            part["answer"] = f"Fejl ved modelkald: {e}"
            part["confidence"] = 0.0
            part["status"] = "error"
            parts_out.append(part)
            continue
        parsed = None
        try:
            parsed = json.loads(out_text)
        except Exception:
            m = re.search(r'(\{[\s\S]*\})', out_text)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    parsed = None
        if not parsed or not isinstance(parsed, dict):
            part["answer"] = "Modelen svarede ikke i det forventede format; kan du præcisere delspørgsmålet?"
            part["confidence"] = 0.0
            part["status"] = "clarify_needed"
            parts_out.append(part)
            continue
        used_ids = parsed.get("used_kb_ids", [])
        if not isinstance(used_ids, list):
            used_ids = []
        used_sources = map_used_kb_ids_to_sources(used_ids, program_rows or [], faq_rows or [])
        if not used_sources:
            part["answer"] = "Modelen angav ingen anvendte kilder fra vidensbasen; kan du uddybe?"
            part["confidence"] = 0.0
            part["status"] = "clarify_needed"
            parts_out.append(part)
            continue
        body_text = parsed.get("body", "") or ""
        if not body_text.strip():
            part["answer"] = "Modelen producerede tomt svar; kan du præcisere?"
            part["confidence"] = 0.0
            part["status"] = "clarify_needed"
            parts_out.append(part)
            continue
        kb_union_text = " ".join([s for (_id, s) in used_pairs])
        body_tokens = set(_word_tokens(body_text))
        kb_tokens = set(_word_tokens(kb_union_text))
        grounding_ratio = 0.0
        if body_tokens:
            grounding_ratio = len(body_tokens & kb_tokens) / len(body_tokens)
        if grounding_ratio < 0.35:
            part["answer"] = "Svaret ser ikke ud til primært at være baseret på den tilgængelige vidensbase. Kan du give mere kontekst?"
            part["confidence"] = 0.0
            part["status"] = "clarify_needed"
            parts_out.append(part)
            continue
        part["answer"] = body_text
        try:
            conf_val = float(parsed.get("confidence", 0.0))
            conf_val = max(0.0, min(1.0, conf_val))
        except Exception:
            conf_val = 0.0
        part["confidence"] = conf_val
        part["used_kb_ids"] = used_ids
        part["used_sources"] = used_sources
        part["status"] = "answered"
        parts_out.append(part)
        confidences.append(conf_val)
        subj = parsed.get("subject", "") or ""
        if subj:
            aggregate_subjects.append(subj)
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
    else:
        avg_conf = 0.0
    body_lines = []
    for i, p in enumerate(parts_out, start=1):
        heading = f"Del {i}: {p['question']}"
        body_lines.append(heading)
        body_lines.append("- Svar:")
        body_lines.append(p["answer"])
        if p.get("used_sources"):
            body_lines.append("- Kilder:")
            for s in p["used_sources"]:
                typ = s.get("type", "unknown")
                title = s.get("title", "")
                snippet = s.get("snippet", "")
                body_lines.append(f"  - [{typ}] {title} (id={s.get('id')}) — \"{snippet[:160]}\"")
        body_lines.append("")
    aggregated_subject = " | ".join([s for s in aggregate_subjects if s]) or (parts_out[0]["question"][:80] if parts_out else "")
    out = {
        "subject": aggregated_subject,
        "body": "\n".join(body_lines).strip(),
        "confidence": round(avg_conf, 3),
        "notes": f"{len(parts_out)} parts processed; {sum(1 for p in parts_out if p['status']=='answered')} answered from KB.",
        "parts": parts_out
    }
    return out

# ---------------- build_kb_block for context display ----------------
def build_kb_block(program_rows: List[Dict[str,str]] = None, faq_rows: List[Dict[str,str]] = None) -> str:
    parts: List[str] = []
    if program_rows:
        parts.append("Program information (admission requirements):")
        for r in program_rows:
            prog = r.get("programme") or ""
            adm = r.get("admission") or ""
            snippet = adm if len(adm) <= KB_SNIPPET_MAX_CHARS else adm[:KB_SNIPPET_MAX_CHARS].rstrip() + " ..."
            parts.append(f"- {prog}: {snippet}")
    if faq_rows:
        parts.append("Supplementary FAQ answers:")
        for r in faq_rows:
            q = r.get("question") or ""
            a = r.get("answer") or ""
            snippet = a if len(a) <= KB_SNIPPET_MAX_CHARS else a[:KB_SNIPPET_MAX_CHARS].rstrip() + " ..."
            parts.append(f"- Q: {q}\n  A: {snippet}")
    return "\n".join(parts)

# ---------------- Save feedback local ----------------
def save_feedback_row_local(row: Dict[str, Any], csv_path: str = FEEDBACK_CSV, xlsx_path: str = FEEDBACK_XLSX) -> Tuple[bool, str]:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    try:
        if not os.path.exists(csv_path):
            cols = LEADING_COLUMNS + METADATA_COLUMNS
            pd.DataFrame(columns=cols).to_csv(csv_path, index=False, encoding="utf-8")
        file_cols = LEADING_COLUMNS + METADATA_COLUMNS
        df_row = pd.DataFrame([row])
        df_row = df_row.reindex(columns=file_cols)
        df_row.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8")
        try:
            if os.path.exists(xlsx_path):
                existing = pd.read_excel(xlsx_path, engine="openpyxl")
                df = pd.concat([existing, df_row], ignore_index=True)
            else:
                df = df_row
            df.to_excel(xlsx_path, index=False, engine="openpyxl")
        except Exception:
            pass
        return True, f"Saved locally to {csv_path}"
    except Exception as e:
        return False, f"Failed to save locally: {e}"

def save_feedback(row_dict: Dict[str, Any]) -> Tuple[bool, str]:
    return save_feedback_row_local(row_dict)

# ---------------- UI: main flow ----------------
programs_kb = load_programs_kb(st.session_state.get("programs_kb_path", PROGRAMS_KB_DEFAULT))
faq_kb = load_faq_kb(st.session_state.get("faq_kb_path", FAQ_KB_DEFAULT))

st.subheader("Indtast mail til analyse")
with st.form("email_form"):
    body = st.text_area("Mail (indsæt hele beskeden her)", value="", height=300)
    submitted = st.form_submit_button("Preview svar")

# clear pending feedback clear
if st.session_state.get("feedback_clear_pending"):
    st.session_state["feedback_text_input"] = ""
    st.session_state["feedback_rating"] = ""
    try:
        del st.session_state["feedback_clear_pending"]
    except Exception:
        pass

if submitted:
    for k in ["last_messages_sent", "assistant_raw_response"]:
        if k in st.session_state:
            del st.session_state[k]
    combined = body or ""
    st.session_state["last_mail_text"] = combined
    language = detect_language(combined)
    program_candidates = detect_programs(combined, programs_kb, top_n=4) if programs_kb else []
    uddannelses_relateret = False
    selected_program_rows = None
    if program_candidates and program_candidates[0][1] >= PROGRAM_DETECT_THRESHOLD:
        uddannelses_relateret = True
        top_row = program_candidates[0][0]
        selected_program_rows = [r for r in programs_kb if r.get("programme","").strip().lower() == top_row.get("programme","").strip().lower()]
    faq_matches = match_faq(combined, faq_kb, top_k=FAQ_MATCH_TOPK) if faq_kb else []
    faq_to_use = [r for r, s in faq_matches][:FAQ_MATCH_TOPK] if faq_matches else []
    assistant_instruction_to_use = None
    assistant_config = load_assistant_config(st.session_state.get("assistant_config_path", ASSISTANT_CONFIG_DEFAULT))
    if assistant_config and isinstance(assistant_config, dict):
        assistant_instruction_to_use = assistant_config.get("instruction")
    if not assistant_instruction_to_use:
        assistant_instruction_to_use = st.session_state.get("assistant_prompt", None)
    used_temp = float(assistant_config.get("temperature")) if assistant_config and assistant_config.get("temperature") is not None else 0.0
    used_top_p = float(assistant_config.get("top_p")) if assistant_config and assistant_config.get("top_p") is not None else 1.0
    used_model = assistant_config.get("model") or DEFAULT_MODEL
    # Use split-aware generator (stricter KB grounding)
    reply = generate_combined_reply_split_parts(combined,
                                               program_rows=selected_program_rows if uddannelses_relateret else None,
                                               faq_rows=faq_to_use,
                                               language=language,
                                               openai_model=used_model,
                                               assistant_instruction=assistant_instruction_to_use,
                                               temperature=used_temp if used_temp is not None else 0.0,
                                               top_p=used_top_p if used_top_p is not None else 1.0)
    st.session_state["previewed"] = True
    st.session_state["last_generated_reply"] = reply
    st.session_state["last_generated_meta"] = {
        "mail_body": combined,
        "program_detected": selected_program_rows[0]['programme'] if selected_program_rows else "",
        "model": used_model,
        "temperature": used_temp,
        "top_p": used_top_p,
        "assistant_instruction": (assistant_instruction_to_use or "")[:4000],
        "generated_subject": reply.get("subject", "") if isinstance(reply, dict) else ""
    }
    st.session_state["feedback_text_input"] = st.session_state.get("feedback_text_input", "")
    st.session_state["feedback_rating"] = st.session_state.get("feedback_rating", "")

# Display preview + feedback UI
if st.session_state.get("previewed"):
    reply = st.session_state.get("last_generated_reply", {})
    meta = st.session_state.get("last_generated_meta", {})
    program_name = meta.get("program_detected", "")
    if program_name:
        st.markdown(f"**Uddannelse relateret: JA — valgt program:** {program_name}")
    else:
        st.markdown("**Uddannelse relateret: NEJ**")
    generated_subject = meta.get("generated_subject", "") or reply.get("subject", "")
    st.text_input("Genereret emne (edit)", value=generated_subject, key="reply_subj")
    generated_body = reply.get("body") or ""
    st.markdown("**Genereret svar (rediger hvis nødvendigt):**")
    st.text_area("Genereret svar (edit)", value=generated_body, height=360, key="reply_body")
    st.markdown(f"**Generation params:** model={meta.get('model')}, temperature={meta.get('temperature')}, top_p={meta.get('top_p')}")
    st.markdown(f"**Confidence (model estimate):** {reply.get('confidence', 0):.2f}")
    if reply.get("notes"):
        st.info(f"Notes: {reply.get('notes')}")
    # Show parts provenance if present
    parts = reply.get("parts", [])
    if parts:
        st.markdown("### Parts and provenance (debug)")
        try:
            st.text_area("parts", value=json.dumps(parts, ensure_ascii=False, indent=2), height=300)
        except Exception:
            st.text_area("parts (repr)", value=str(parts)[:4096], height=300)
    st.markdown("### Feedback på dette svar")
    feedback_text = st.text_area("Skriv din feedback her", value=st.session_state.get("feedback_text_input", ""), height=160, key="feedback_text_input")
    rating = st.selectbox("Vurder svar (valgfrit)", options=["", "1 - meget utilfreds", "2", "3", "4", "5 - meget tilfreds"], index=0, key="feedback_rating")
    if st.button("Gem feedback", key="save_feedback_btn"):
        mail_body_orig = meta.get("mail_body", "")
        generated_body_current = st.session_state.get("reply_body", generated_body)
        generated_subject_current = st.session_state.get("reply_subj", generated_subject)
        confidence = reply.get("confidence") if isinstance(reply.get("confidence"), (int, float)) else reply.get("confidence", "")
        program_name = meta.get("program_detected", "")
        now_utc1 = datetime.now(timezone.utc) + timedelta(hours=1)
        timestamp = now_utc1.isoformat()
        od = OrderedDict()
        od["Mail"] = mail_body_orig
        od["Svar"] = generated_body_current
        od["feedback_text"] = feedback_text or ""
        od["rating"] = rating or ""
        od["assistant_instruction"] = meta.get("assistant_instruction", "")
        od["generated_subject"] = generated_subject_current
        od["program_detected"] = program_name
        od["timestamp_utc"] = timestamp
        od["model"] = meta.get("model", "")
        od["temperature"] = meta.get("temperature", "")
        od["top_p"] = meta.get("top_p", "")
        od["confidence"] = confidence
        ok, msg = save_feedback(dict(od))
        if ok:
            st.success("Feedback gemt. " + msg)
            st.session_state["feedback_clear_pending"] = True
            try:
                st.experimental_rerun()
            except Exception:
                st.stop()
        else:
            st.error(msg)
    messages_debug = st.session_state.get("last_messages_sent")
    if messages_debug:
        st.markdown("**Messages sent to model (debug)**")
        try:
            st.text_area("messages", value=json.dumps(messages_debug, ensure_ascii=False, indent=2), height=200)
        except Exception:
            st.text_area("messages (repr)", value=str(messages_debug)[:4096], height=200)
else:
    st.info("Indtast mail i 'Mail' feltet og klik Preview svar for automatisk svarudkast.")

st.write("---")
st.caption("Feedback saved to feedback/feedback.csv (XLSX best-effort). Use 'Reset feedback storage' (sidebar) to start from a new document with the specified columns.")
