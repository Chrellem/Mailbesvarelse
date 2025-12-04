"""
streamlit_app_hybrid.py

Stable app version with:
- KB loaders for Excel/CSV (Programmes + FAQ).
- detect_language heuristic.
- Single-shot generator (uses YAML instruction if present; optional checkbox to enforce YAML-only).
- Fixed session_state usage and safe cache reload behavior (avoids the StreamlitAPIException).
- Local feedback saving and sidebar controls.
"""
from typing import List, Dict, Any, Tuple
import os
import re
import json
import ssl
import urllib.request
import time
from datetime import datetime, timezone, timedelta
from collections import OrderedDict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Optional libs
try:
    import yaml
except Exception:
    yaml = None

try:
    import openai
except Exception:
    openai = None

load_dotenv()

# ---------------- Defaults / Config ----------------
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

# Leading columns — kept consistent
LEADING_COLUMNS = [
    "Mail", "Svar", "feedback_text", "rating",
    "assistant_instruction", "generated_subject", "program_detected", "timestamp_utc"
]
METADATA_COLUMNS = ["model", "temperature", "top_p", "confidence"]

st.set_page_config(page_title="Mailbesvarelse — KB-driven", layout="wide")
st.title("Mail‑besvarelse — KB‑driven (Program + FAQ)")

# ---------------- Sidebar: config + UI ----------------
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

# New: option to force YAML-only prompt (no fallback).
# IMPORTANT: do NOT assign into session_state again after calling checkbox;
# st.sidebar.checkbox already writes into session_state under the given key.
use_yaml_only = st.sidebar.checkbox(
    "Use YAML only for assistant prompt (no fallback)",
    value=st.session_state.get("use_yaml_only", False),
    key="use_yaml_only"
)
# NOTE: do NOT do st.session_state["use_yaml_only"] = use_yaml_only here (Streamlit disallows
# certain direct mutations at this point and the checkbox already set the key).

# ---------------- YAML loader (mtime aware) ----------------
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

# ---------------- Helper: read CSV/XLSX robustly ----------------
def _read_table_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
            df = pd.read_excel(path, sheet_name=0, dtype=str, engine="openpyxl").fillna("")
        else:
            df = pd.read_csv(path, sep=";", dtype=str, engine="python", quotechar='"', keep_default_na=False).fillna("")
    except Exception:
        df = pd.read_csv(path, sep=";", dtype=str, engine="python", encoding="utf-8", quotechar='"', keep_default_na=False).fillna("")
    df.columns = [c.strip() for c in df.columns]
    return df

# ---------------- KB loaders ----------------
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

# ---------------- Language detection (heuristic) ----------------
def detect_language(text: str) -> str:
    """
    Heuristic offline detector returning 'da' or 'en'.
    Conservative: returns 'en' only if english signals > danish signals; otherwise 'da'.
    """
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

# ---------------- Matching / detection (simple overlap) ----------------
def normalize_text(s: str) -> str:
    return (s or "").strip().lower()

def score_text_overlap(a: str, b: str) -> float:
    a_tokens = set(re.findall(r'\w{3,}', (a or "").lower()))
    b_tokens = set(re.findall(r'\w{3,}', (b or "").lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return float(len(a_tokens.intersection(b_tokens)))

def detect_programs(mail_text: str, programs_kb: List[Dict[str,str]], top_n: int = 3) -> List[Tuple[Dict[str,str], float]]:
    t = normalize_text(mail_text)
    scored: List[Tuple[Dict[str,str], float]] = []
    if not programs_kb:
        return scored
    for row in programs_kb:
        pname = normalize_text(row.get("programme",""))
        adm = normalize_text(row.get("admission",""))
        kw = normalize_text(row.get("keywords",""))
        short = normalize_text(row.get("short_description",""))
        score = 0.0
        if pname and pname in t:
            score += 6.0
        score += 2.0 * score_text_overlap(pname, t)
        score += 1.5 * score_text_overlap(kw, t)
        score += 0.8 * score_text_overlap(short, t)
        score += 0.5 * score_text_overlap(adm, t)
        if score > 0:
            scored.append((row, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

def match_faq(mail_text: str, faq_kb: List[Dict[str,str]], top_k: int = 3) -> List[Tuple[Dict[str,str], float]]:
    t = normalize_text(mail_text)
    scored = []
    if not faq_kb:
        return scored
    for row in faq_kb:
        q = normalize_text(row.get("question",""))
        a = normalize_text(row.get("answer",""))
        score = score_text_overlap(t, q) * 2.0
        score += score_text_overlap(t, a) * 0.5
        if row.get("category") and row.get("category").lower() in t:
            score += 1.0
        if score > 0:
            scored.append((row, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# ---------------- OpenAI wrapper (SDK or REST fallback) ----------------
def create_chat_completion(messages: List[Dict[str, str]],
                           model: str = None,
                           max_tokens: int = 400,
                           temperature: float = 0.2,
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

# ---------------- Generation function (single-shot JSON expected) ----------------
def generate_combined_reply(mail_text: str,
                            program_rows: List[Dict[str,str]] = None,
                            faq_rows: List[Dict[str,str]] = None,
                            language: str = "da",
                            openai_model: str = None,
                            assistant_instruction: str = None,
                            temperature: float = 0.2,
                            top_p: float = 1.0) -> Dict[str, Any]:
    model = openai_model or DEFAULT_MODEL
    kb_block = build_kb_block(program_rows, faq_rows)
    lang_name = "dansk" if language == "da" else "english"
    base_system = (
        f"You are a helpful assistant that must answer a prospective student's question. "
        f"YOU MUST ONLY use the information provided in the 'KB block' below. Do NOT invent or assume anything. "
        f"If the KB does not contain enough information to answer, respond with a short clarification question asking for the missing info. "
        f"Reply in {lang_name}. Output must be a single JSON object with keys: subject, body, confidence (0-1), notes."
    )
    messages = [{"role": "system", "content": base_system}]
    instr = assistant_instruction or st.session_state.get("assistant_prompt", None)
    if instr:
        messages.append({"role": "assistant", "content": instr})
    user = f"Mail:\n'''{mail_text}'''\n\nKB block:\n{kb_block}\n\nInstructions:\n- Answer the most important question first.\n- If the mail refers to a specific programme, refer to it by name.\n- Keep the reply concise and practical.\n- If KB is insufficient, ask one clear follow-up question.\nReturn only JSON."
    messages.append({"role": "user", "content": user})
    try:
        st.session_state["last_messages_sent"] = messages
    except Exception:
        pass
    try:
        out_text, resp = create_chat_completion(messages, model=model, max_tokens=700, temperature=temperature, top_p=top_p)
    except Exception as e:
        return {"subject": "", "body": f"Fejl ved opkald til model: {e}", "confidence": 0.0, "notes": "LLM call failed"}
    try:
        parsed = json.loads(out_text)
        return parsed
    except Exception:
        m = re.search(r'(\{[\s\S]*\})', out_text)
        if m:
            try:
                parsed = json.loads(m.group(1))
                return parsed
            except Exception:
                pass
    return {"subject": "", "body": out_text, "confidence": 0.0, "notes": "Kunne ikke parse JSON fra model; se body"}

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

# clear pending session flags
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

    # detect language & KB matches
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

    # Assistant instruction selection respecting "use_yaml_only" checkbox
    assistant_instruction_to_use = None
    assistant_config = load_assistant_config(st.session_state.get("assistant_config_path", ASSISTANT_CONFIG_DEFAULT))
    if assistant_config and isinstance(assistant_config, dict):
        assistant_instruction_to_use = assistant_config.get("instruction")

    if not assistant_instruction_to_use:
        if st.session_state.get("use_yaml_only"):
            # YAML enforced but missing: show error and set to empty string to avoid fallback
            st.sidebar.error("YAML missing key 'instruction' eller fil ikke fundet. Tjek 'Assistant config path' eller slå 'Use YAML only' fra.")
            assistant_instruction_to_use = ""
        else:
            # legacy behavior: use session_state fallback
            assistant_instruction_to_use = st.session_state.get("assistant_prompt", None)

    used_temp = float(assistant_config.get("temperature")) if assistant_config and assistant_config.get("temperature") is not None else 0.2
    used_top_p = float(assistant_config.get("top_p")) if assistant_config and assistant_config.get("top_p") is not None else 1.0
    used_model = assistant_config.get("model") or DEFAULT_MODEL

    reply = generate_combined_reply(combined,
                                    program_rows=selected_program_rows if uddannelses_relateret else None,
                                    faq_rows=faq_to_use,
                                    language=language,
                                    openai_model=used_model,
                                    assistant_instruction=assistant_instruction_to_use,
                                    temperature=used_temp,
                                    top_p=used_top_p)

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
