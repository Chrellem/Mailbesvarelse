"""
streamlit_app_hybrid.py

KB-first mail-besvarelse app using a repository YAML file for assistant instruction and generation params.

Behavior:
- Load Programmes.xlsx and faq_kb.xlsx (sheets/)
- Detect if mail is programme-related and fetch matching programme rows
- Match mail against FAQ KB and pick top FAQ answers
- Build KB block (programme admission + FAQ snippets)
- Load assistant config from config/assistant_config.yaml and use:
    - instruction (assistant-role content)
    - temperature, top_p, model
  as defaults for generation. Sidebar has a fallback assistant prompt and debug toggles.
- Generate reply via chat completions (no Assistants API calls).
"""
from typing import List, Dict, Any, Tuple
import os
import re
import json
import ssl
import urllib.request
import traceback
import time

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# YAML support
try:
    import yaml
except Exception:
    yaml = None

# Optional local embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# OpenAI import (may be absent; we have REST fallback that needs OPENAI_API_KEY)
try:
    import openai
except Exception:
    openai = None

load_dotenv()

# ---------------- Config / defaults ----------------
PROGRAMS_KB_DEFAULT = "sheets/Programmes.xlsx"
FAQ_KB_DEFAULT = "sheets/faq_kb.xlsx"
ASSISTANT_CONFIG_DEFAULT = "config/assistant_config.yaml"

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
PROGRAM_DETECT_THRESHOLD = float(os.getenv("PROGRAM_DETECT_THRESHOLD", "2.5"))
FAQ_MATCH_TOPK = int(os.getenv("FAQ_MATCH_TOPK", "3"))
KB_SNIPPET_MAX_CHARS = 800

st.set_page_config(page_title="Mailbesvarelse — KB-driven", layout="wide")
st.title("Mail‑besvarelse — KB‑driven (Program + FAQ)")

# ---------------- Sidebar / secrets / config path ----------------
st.sidebar.header("Konfiguration")
st.sidebar.text_input("Programmes KB path", value=PROGRAMS_KB_DEFAULT, key="programs_kb_path")
st.sidebar.text_input("FAQ KB path", value=FAQ_KB_DEFAULT, key="faq_kb_path")
st.sidebar.text_input("Assistant config path (YAML)", value=ASSISTANT_CONFIG_DEFAULT, key="assistant_config_path")
st.sidebar.text_input("OpenAI model (env default)", value=DEFAULT_MODEL, key="openai_model")
st.sidebar.write("Program detection threshold:", PROGRAM_DETECT_THRESHOLD)

# Assistant prompt fallback (if YAML missing)
st.sidebar.markdown("### Assistant prompt (fallback)")
assistant_prompt_ui = st.sidebar.text_area(
    "Assistant prompt (fallback). Bruges hvis YAML ikke findes eller er tom.",
    value=st.session_state.get("assistant_prompt", ""),
    height=120,
    key="assistant_prompt_input"
)
if assistant_prompt_ui is not None:
    st.session_state["assistant_prompt"] = assistant_prompt_ui

# Read OPENAI key from st.secrets or environment
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

# ---------------- Helper: load assistant config YAML ----------------
@st.cache_data(ttl=300)
def load_assistant_config(path: str) -> Dict[str, Any]:
    """
    Load assistant config YAML from repo. Expected keys:
      instruction: str
      temperature: float
      top_p: float
      model: str
    """
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    if yaml is None:
        st.sidebar.warning("PyYAML ikke installeret — kan ikke læse assistant config YAML.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
            out = {}
            out["instruction"] = str(cfg.get("instruction", "")).strip() if cfg.get("instruction") else None
            out["temperature"] = float(cfg.get("temperature")) if cfg.get("temperature") is not None else None
            out["top_p"] = float(cfg.get("top_p")) if cfg.get("top_p") is not None else None
            out["model"] = str(cfg.get("model")) if cfg.get("model") else None
            return out
    except Exception as e:
        st.sidebar.error(f"Fejl ved læsning af assistant config: {e}")
        return {}

# ---------------- Helper: extract text from JSON response (robust) ----------------
def extract_text_from_response_json(j: Any) -> str:
    texts = []
    def recurse(o):
        if isinstance(o, str):
            if len(o.strip()) > 2:
                texts.append(o.strip())
        elif isinstance(o, dict):
            for v in o.values():
                recurse(v)
        elif isinstance(o, list):
            for item in o:
                recurse(item)
    try:
        recurse(j)
    except Exception:
        pass
    if texts:
        texts_sorted = sorted(texts, key=lambda s: len(s), reverse=True)
        return texts_sorted[0]
    try:
        return json.dumps(j, ensure_ascii=False)[:4000]
    except Exception:
        return str(j)[:2000]

# ---------------- Chat completion wrapper (use temperature + top_p) ----------------
def create_chat_completion(messages: List[Dict[str, str]],
                           model: str = None,
                           max_tokens: int = 400,
                           temperature: float = 0.2,
                           top_p: float = 1.0) -> Tuple[str, Any]:
    model = model or DEFAULT_MODEL

    # Try using openai client if present
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
                    try:
                        text = resp["choices"][0]["message"]["content"]
                    except Exception:
                        text = str(resp)
                return text, resp
        except Exception:
            pass

    # REST fallback
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not available for REST fallback. Please set the key in st.secrets or env.")

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

# ---------------- KB loaders ----------------
@st.cache_data(ttl=300)
def load_programs_kb(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        if path.lower().endswith(".xlsx"):
            df = pd.read_excel(path, sheet_name=0, dtype=str, engine="openpyxl").fillna("")
        else:
            df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        st.warning(f"Failed loading programs KB '{path}': {e}")
        return []
    df.columns = [c.strip().lower() for c in df.columns]
    id_col = next((c for c in df.columns if c == "id"), df.columns[0] if len(df.columns) > 0 else None)
    prog_col = next((c for c in df.columns if "programme" in c or "program" in c), df.columns[1] if len(df.columns) > 1 else None)
    adm_col = next((c for c in df.columns if "admiss" in c or "requi" in c or "require" in c), df.columns[2] if len(df.columns) > 2 else None)
    out = []
    for _, r in df.iterrows():
        out.append({
            "id": str(r.get(id_col, "")).strip() if id_col else "",
            "programme": str(r.get(prog_col, "")).strip() if prog_col else "",
            "admission": str(r.get(adm_col, "")).strip() if adm_col else ""
        })
    return out

@st.cache_data(ttl=300)
def load_faq_kb(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        if path.lower().endswith(".xlsx"):
            df = pd.read_excel(path, sheet_name=0, dtype=str, engine="openpyxl").fillna("")
        else:
            df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        st.warning(f"Failed loading FAQ KB '{path}': {e}")
        return []
    df.columns = [c.strip().lower() for c in df.columns]
    id_col = next((c for c in df.columns if c == "id"), df.columns[0] if len(df.columns) > 0 else None)
    q_col = next((c for c in df.columns if "question" in c or c.startswith("q")), df.columns[1] if len(df.columns) > 1 else None)
    a_col = next((c for c in df.columns if "answer" in c or "svar" in c or c.startswith("a")), df.columns[2] if len(df.columns) > 2 else None)
    out = []
    for _, r in df.iterrows():
        out.append({
            "id": str(r.get(id_col, "")).strip() if id_col else "",
            "question": str(r.get(q_col, "")).strip() if q_col else "",
            "answer": str(r.get(a_col, "")).strip() if a_col else ""
        })
    return out

# ---------------- Utilities & matching ----------------
def detect_language(text: str) -> str:
    t = (text or "").lower()
    danish_tokens = ["hej", "tak", "venlig", "venligst", "hvordan", "hvad", "jeg", "ikke"]
    english_tokens = ["hello", "thank", "please", "how", "what", "i", "not"]
    dan = sum(1 for tok in danish_tokens if tok in t)
    eng = sum(1 for tok in english_tokens if tok in t)
    return "da" if dan >= eng else "en"

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
    scored = []
    for row in programs_kb:
        pname = normalize_text(row.get("programme", ""))
        if not pname:
            continue
        score = 0.0
        if pname in t:
            score += 6.0
        score += score_text_overlap(pname, t)
        adm = normalize_text(row.get("admission", ""))
        if adm:
            score += 0.5 * score_text_overlap(adm, t)
        if score > 0:
            scored.append((row, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

def match_faq(mail_text: str, faq_kb: List[Dict[str,str]], top_k: int = 3) -> List[Tuple[Dict[str,str], float]]:
    scored = []
    t = normalize_text(mail_text)
    for row in faq_kb:
        q = normalize_text(row.get("question", ""))
        a = normalize_text(row.get("answer", ""))
        score = score_text_overlap(t, q) * 2.0
        score += score_text_overlap(t, a) * 0.5
        if q and q in t:
            score += 3.0
        if score > 0:
            scored.append((row, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        for row in faq_kb:
            a = normalize_text(row.get("answer",""))
            s = score_text_overlap(t, a)
            if s > 0:
                scored.append((row, float(s)))
        scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def select_program_rows(programme_name_or_id: str, programs_kb: List[Dict[str,str]]) -> List[Dict[str,str]]:
    if not programme_name_or_id or not programs_kb:
        return []
    target = str(programme_name_or_id).strip().lower()
    matches = [r for r in programs_kb if (str(r.get("programme","")).strip().lower() == target) or (str(r.get("id","")).strip().lower() == target)]
    if not matches:
        matches = [r for r in programs_kb if target in (str(r.get("programme","")).strip().lower()) or target in (str(r.get("id","")).strip().lower())]
    return matches

def build_kb_block(program_rows: List[Dict[str,str]] = None, faq_rows: List[Dict[str,str]] = None) -> str:
    parts = []
    if program_rows:
        parts.append("Program information (admission requirements):")
        for r in program_rows:
            prog = r.get("programme") or r.get("question") or ""
            adm = r.get("admission") or r.get("answer") or ""
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

# ---------------- Generation using YAML instruction ----------------
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

    messages: List[Dict[str, str]] = [{"role": "system", "content": base_system}]
    instr = assistant_instruction or st.session_state.get("assistant_prompt", None)
    if instr:
        messages.append({"role": "assistant", "content": instr})

    user = f"Mail:\n'''{mail_text}'''\n\nKB block:\n{kb_block}\n\nInstructions:\n- Answer the most important question first.\n- If the mail refers to a specific programme, refer to it by name.\n- Keep the reply concise and practical.\n- If KB is insufficient, ask one clear follow-up question.\nReturn only JSON."
    messages.append({"role": "user", "content": user})

    # store messages for debugging
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

# ---------------- Debug: show YAML config and messages ----------------
st.sidebar.markdown("### Assistant config (YAML)")
assistant_config_path = st.session_state.get("assistant_config_path", ASSISTANT_CONFIG_DEFAULT)
assistant_config = load_assistant_config(assistant_config_path)
if not assistant_config:
    st.sidebar.info("Ingen assistant config fundet eller filen er tom.")
else:
    try:
        st.sidebar.text_area("assistant_config (parsed)", value=json.dumps(assistant_config, ensure_ascii=False, indent=2), height=200)
    except Exception:
        st.sidebar.text_area("assistant_config (repr)", value=str(assistant_config)[:4000], height=200)

# ---------------- Main UI / flow ----------------
programs_kb = load_programs_kb(st.session_state.get("programs_kb_path", PROGRAMS_KB_DEFAULT))
faq_kb = load_faq_kb(st.session_state.get("faq_kb_path", FAQ_KB_DEFAULT))

st.subheader("Indtast mail til analyse")

with st.form("email_form"):
    subject = st.text_input("Subject", value="")
    body = st.text_area("Body", value="", height=260)
    submitted = st.form_submit_button("Preview svar")

if not submitted:
    st.info("Indtast mail og klik Preview svar for automatisk svarudkast.")
else:
    combined = f"{subject}\n\n{body}"
    st.session_state["last_mail_text"] = combined

    # 1) language
    language = detect_language(combined)

    # 2) detect programme
    program_candidates = detect_programs(combined, programs_kb, top_n=4) if programs_kb else []
    uddannelses_relateret = False
    selected_program_rows = None

    if program_candidates and program_candidates[0][1] >= PROGRAM_DETECT_THRESHOLD:
        uddannelses_relateret = True
        top_row = program_candidates[0][0]
        selected_program_rows = select_program_rows(top_row.get("programme"), programs_kb)
        st.markdown(f"**Uddannelse relateret: JA — valgt program:** {top_row.get('programme')} (score {program_candidates[0][1]:.1f})")
    else:
        uddannelses_relateret = False
        st.markdown("**Uddannelse relateret: NEJ**")

    # 3) FAQ matches
    faq_matches = match_faq(combined, faq_kb, top_k=FAQ_MATCH_TOPK) if faq_kb else []
    faq_to_use = [r for r, s in faq_matches][:FAQ_MATCH_TOPK] if faq_matches else []

    # 4) choose instruction & params from YAML (or fallback)
    assistant_instruction_to_use = None
    if assistant_config and isinstance(assistant_config, dict):
        assistant_instruction_to_use = assistant_config.get("instruction")
    if not assistant_instruction_to_use:
        assistant_instruction_to_use = st.session_state.get("assistant_prompt", None)

    used_temp = float(assistant_config.get("temperature")) if assistant_config and assistant_config.get("temperature") is not None else 0.2
    used_top_p = float(assistant_config.get("top_p")) if assistant_config and assistant_config.get("top_p") is not None else 1.0
    used_model = assistant_config.get("model") or DEFAULT_MODEL

    st.markdown("**Assistant instruction (brugt til denne generering):**")
    st.text_area("Assistant instruction (used)", value=assistant_instruction_to_use or "<ingen>", height=120)
    st.markdown(f"**Generation params:** model={used_model}, temperature={used_temp}, top_p={used_top_p}")

    # 5) generate reply using YAML instruction
    if uddannelses_relateret and selected_program_rows:
        reply = generate_combined_reply(combined,
                                        program_rows=selected_program_rows,
                                        faq_rows=faq_to_use,
                                        language=language,
                                        openai_model=used_model,
                                        assistant_instruction=assistant_instruction_to_use,
                                        temperature=used_temp,
                                        top_p=used_top_p)
    else:
        reply = generate_combined_reply(combined,
                                        program_rows=None,
                                        faq_rows=faq_to_use,
                                        language=language,
                                        openai_model=used_model,
                                        assistant_instruction=assistant_instruction_to_use,
                                        temperature=used_temp,
                                        top_p=used_top_p)

    # show messages that were sent for debugging (if present)
    messages_debug = st.session_state.get("last_messages_sent")
    if messages_debug:
        st.markdown("**Messages sent to model (debug)**")
        try:
            st.text_area("messages", value=json.dumps(messages_debug, ensure_ascii=False, indent=2), height=200)
        except Exception:
            st.text_area("messages (repr)", value=str(messages_debug)[:4096], height=200)

    no_program_match = (not uddannelses_relateret)
    no_faq_match = len(faq_to_use) == 0

    if no_program_match and no_faq_match:
        st.warning("Meget lavt match i Programmes.xlsx og faq_kb.xlsx — genererer en kort afklaringsforespørgsel.")
        if language == "da":
            clarification_subject = "Opfølgning: Manglende oplysninger"
            clarification_body = ("Kære ansøger,\n\n"
                                  "Tak for din henvendelse. For at vi kan hjælpe dig bedst muligt, oplys venligst hvilken uddannelse henvendelsen gælder, samt relevante detaljer (fx uddannelsesbaggrund, karakterer, statsborgerskab).")
        else:
            clarification_subject = "Follow-up: Missing information"
            clarification_body = ("Dear applicant,\n\n"
                                  "Thanks for your message. To help you further, please tell us which programme you refer to and provide relevant details (e.g. prior education, grades, citizenship).")
        st.text_input("Suggested subject (edit):", value=clarification_subject, key="clarify_subj")
        st.text_area("Suggested body (edit):", value=clarification_body, height=260, key="clarify_body")
    else:
        subj = ""
        body_out = ""
        if reply:
            subj = reply.get("subject") or ""
            body_out = reply.get("body") or ""
        if not subj:
            subj = f"Vedr. din henvendelse om {selected_program_rows[0]['programme']}" if selected_program_rows else "Vedr. din henvendelse"
        st.text_input("Suggested subject (edit):", value=subj, key="reply_subj")
        st.text_area("Suggested body (edit):", value=body_out, height=360, key="reply_body")
        if reply:
            st.markdown(f"**Confidence (model estimate):** {reply.get('confidence', 0):.2f}")
            if reply.get("notes"):
                st.info(f"Notes: {reply.get('notes')}")

st.write("---")
st.caption("Logic: detect programme → lookup Programmes.xlsx → supplement with faq_kb.xlsx → load assistant config (YAML) → inject as assistant-role → final chat completion generation.")
