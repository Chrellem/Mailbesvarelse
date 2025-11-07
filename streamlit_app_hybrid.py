"""
Hybrid Streamlit prototype:
- sentence-transformers local embeddings for semantic search
- OpenAI ChatCompletion for draft generation and slot extraction/confirmation
Place this file in repo root as streamlit_app_hybrid.py.
"""
import os
import re
import json
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

# OpenAI client
try:
    import openai
except Exception:
    openai = None

load_dotenv()

# Config
REQUIREMENTS_CSV = os.getenv("REQUIREMENTS_CSV", "sheets/requirements_suggestions.csv")
TEMPLATES_CSV = os.getenv("TEMPLATES_CSV", "sheets/templates_suggestions.csv")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
TEST_MODE = os.getenv("TEST_MODE", "true").lower() in ("1", "true", "yes")

# Thresholds (tune as needed)
HIGH_THRESHOLD = float(os.getenv("HIGH_SIM_THRESHOLD", "0.80"))
MID_THRESHOLD = float(os.getenv("MID_SIM_THRESHOLD", "0.65"))
KEYWORD_BOOST = float(os.getenv("KEYWORD_BOOST", "0.06"))
SPECIFICITY_BOOST = float(os.getenv("SPECIFICITY_BOOST", "0.01"))

st.set_page_config(page_title="Mail‑besvarelse Prototype (Hybrid)", layout="wide")
st.title("Prototype: Mail‑besvarelse (hybrid semantic + keywords + LLM)")

st.sidebar.header("Konfiguration")
st.sidebar.text_input("Requirements CSV", value=REQUIREMENTS_CSV, key="req_path")
st.sidebar.text_input("Templates CSV", value=TEMPLATES_CSV, key="tpl_path")
st.sidebar.text_input("OpenAI model", value=OPENAI_MODEL, key="openai_model")
use_openai_default = bool(OPENAI_API_KEY and openai is not None)
st.sidebar.checkbox("Use OpenAI for slot extraction/confirmation", value=use_openai_default, key="use_openai")
st.sidebar.markdown(f"High threshold: {HIGH_THRESHOLD}  —  Mid threshold: {MID_THRESHOLD}")

if openai is not None and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

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

@st.cache_resource(ttl=3600)
def get_sentence_model(name=EMBED_MODEL_NAME):
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(name)
    except Exception:
        return None

def embed_texts(model, texts):
    if model is None:
        return None
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms

def cosine_sim(a, b):
    if a is None or b is None:
        return np.array([])
    return np.dot(b, a)

def normalize_text(s):
    return (s or "").strip().lower()

COUNTRY_KEYWORDS = {
    "DK": ["danmark", "danish", "denmark", "dk"],
    "SE": ["sverige", "sweden", "svenska", "se"],
    "PL": ["poland", "polska", "polen", "pl"],
    "CN": ["china", "kina", "gaokao", "cn"]
}

def detect_country(text):
    t = normalize_text(text)
    for code, kws in COUNTRY_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return code
    return "*"

def split_keywords(cell):
    out = []
    for k in (cell or "").split(","):
        k = k.strip().lower()
        if not k:
            continue
        if len(k) < 3 and ("-" not in k and " " not in k and not any(ch.isdigit() for ch in k)):
            continue
        out.append(k)
    return out

def path_specificity(path):
    toks = (path or "").split("/")
    return sum(1 for t in toks if t and t != "*")

def path_matches_pattern(path, pattern):
    p_toks = str(path).split("/")
    pat_toks = str(pattern).split("/")
    if len(p_toks) != len(pat_toks):
        return False
    for a, b in zip(p_toks, pat_toks):
        if b == "*":
            continue
        if a != b:
            return False
    return True

def keyword_matches_text(text, kw):
    if not kw:
        return False
    kw_escaped = re.escape(kw)
    pattern = r'(?i)\b' + kw_escaped + r'\b'
    try:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    except re.error:
        return kw in text

def build_requirement_index(req_df, model):
    rows = []
    texts = []
    for _, r in req_df.iterrows():
        rows.append(r.to_dict())
        text = " ".join([str(r.get("requirements_cache","")), str(r.get("path","")), str(r.get("question_keywords",""))])
        texts.append(text)
    embs = embed_texts(model, texts) if model else None
    return rows, embs

def keyword_boost_for_row(row, incoming_text):
    kws = split_keywords(row.get("question_keywords",""))
    matched = 0
    for kw in kws:
        if keyword_matches_text(incoming_text, kw):
            matched += 1
    return matched * KEYWORD_BOOST, matched

def call_openai_slot_extractor(mail_text, candidate_rows):
    if openai is None or not OPENAI_API_KEY:
        return None
    ctx = []
    for r in candidate_rows:
        ctx.append({
            "id": r.get("id",""),
            "path": r.get("path",""),
            "requirements_cache": r.get("requirements_cache","")[:400],
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
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":system}, {"role":"user","content":user}],
            temperature=0.0,
            max_tokens=400
        )
        out = resp["choices"][0]["message"]["content"].strip()
        try:
            parsed = json.loads(out)
            return parsed
        except Exception:
            m = re.search(r'(\{[\s\S]*\})', out)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    return None
            return None
    except Exception as e:
        st.error(f"OpenAI call failed: {e}")
        return None

# --- Ny funktion: generer draft reply via OpenAI ---
def generate_draft_reply(mail_text, top_candidates, templates_dict, openai_model=OPENAI_MODEL):
    """
    Returnerer dict med keys: subject, body, confidence, notes
    top_candidates: liste af tuples (row, sim, matched_count, spec, combined)
    """
    if openai is None or not OPENAI_API_KEY:
        return None

    ctx_lines = []
    for c in top_candidates[:4]:
        row = c[0] if isinstance(c, (list, tuple)) else c
        ctx_lines.append(f"- id:{row.get('id','')}, path:{row.get('path','')}, topic:{row.get('topic','')}, url:{row.get('requirements_url','')}")
    ctx_text = "\n".join(ctx_lines)

    system = (
        "Du er en hjælpsom assistent der skriver et kort, professionelt svar på en studenterhenvendelse. "
        "Returner KUN et JSON‑objekt med felterne: subject, body, confidence (0-1), notes.\n"
        "Body skal være venlig, kortfattet og indeholde link(e) hvis relevante.\n"
        "Angiv ikke interne systemnavne; referer til links og konkrete næste trin for ansøgeren."
    )

    user = (
        f"Mail content:\n'''{mail_text}'''\n\n"
        f"Top candidate contexts:\n{ctx_text}\n\n"
        "Skriv et forslag til SUBJECT (max 80 chars) og BODY (kort, max ~250-400 words) som en JSON objekt.\n"
        "BODY må gerne indeholde: 1) Tak for henvendelsen, 2) det vigtigste svar eller link, 3) konkret næste skridt for ansøgeren, 4) kontaktinfo hvis nødvendigt.\n"
        "Returner kun et JSON‑objekt (ingen ekstra tekst)."
    )

    try:
        resp = openai.ChatCompletion.create(
            model=openai_model,
            messages=[{"role":"system","content":system}, {"role":"user","content":user}],
            temperature=0.3,
            max_tokens=500
        )
        out = resp["choices"][0]["message"]["content"].strip()
        try:
            parsed = json.loads(out)
            return parsed
        except Exception:
            m = re.search(r'(\{[\s\S]*\})', out)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    return {"subject": "", "body": out, "confidence": 0.0, "notes": "Kunne ikke parse fuld JSON; se body."}
            return {"subject": "", "body": out, "confidence": 0.0, "notes": "Ingen JSON fundet i modeloutput."}
    except Exception as e:
        return {"subject": "", "body": f"Fejl ved opkald til OpenAI: {e}", "confidence": 0.0, "notes": "OpenAI-fejl"}

def choose_best_by_semantic(rows, embs, query_emb, incoming_text):
    if embs is None or query_emb is None:
        return []
    sims = cosine_sim(query_emb, embs)
    candidates = []
    for i, score in enumerate(sims):
        r = rows[i]
        boost, matched_count = keyword_boost_for_row(r, incoming_text)
        spec = path_specificity(r.get("path",""))
        combined = float(score) + boost + spec * SPECIFICITY_BOOST
        candidates.append((r, float(score), matched_count, spec, combined))
    candidates_sorted = sorted(candidates, key=lambda x: x[4], reverse=True)
    return candidates_sorted

# Load data & model
req_path = st.session_state.get("req_path", REQUIREMENTS_CSV)
tpl_path = st.session_state.get("tpl_path", TEMPLATES_CSV)
req_df = load_requirements(req_path)
templates = load_templates(tpl_path)

if req_df.empty:
    st.warning(f"Requirements CSV ikke fundet eller tom: {req_path}")
if not templates:
    st.warning(f"Templates CSV ikke fundet eller tom: {tpl_path}")

model = get_sentence_model(EMBED_MODEL_NAME)
rows_index, index_embs = build_requirement_index(req_df, model) if model is not None and not req_df.empty else ([], None)

st.subheader("Test input (hybrid)")
with st.form("email_form"):
    subject = st.text_input("Subject", value="")
    body = st.text_area("Body", value="", height=200)
    applicant_name = st.text_input("Applicant name (valgfri)", value="")
    program = st.text_input("Program (valgfri)", value="")
    use_openai_checkbox = st.checkbox("Use OpenAI slot extraction for mid-confidence", value=bool(OPENAI_API_KEY))
    submitted = st.form_submit_button("Preview svar")
if not submitted:
    st.info("Indtast subject/body og klik Preview.")
else:
    combined = f"{subject}\n\n{body}"
    country = detect_country(combined)
    st.markdown(f"**Detected country_code (heuristic):** `{country}`")

    if model is None:
        st.warning("Local embedding model ikke tilgængelig. Installer sentence-transformers for semantic search.")
    query_emb = None
    if model is not None:
        query_emb = embed_texts(model, [combined])[0]

    candidates = choose_best_by_semantic(rows_index, index_embs, query_emb, combined) if query_emb is not None and index_embs is not None else []

    if not candidates:
        st.error("Ingen semantiske kandidater fundet. Prøv keywords eller tilføj flere rækker.")
        st.stop()

    topk = candidates[:6]
    out_table = []
    for r, sim, matched_count, spec, combined_score in topk:
        out_table.append({
            "id": r.get("id",""),
            "path": r.get("path",""),
            "topic": r.get("topic",""),
            "sim": round(sim, 3),
            "matched_keywords": matched_count,
            "specificity": spec,
            "score": round(combined_score, 3),
            "exclude_flag": r.get("exclude_flag","")
        })
    st.write("Top candidates (semantic + keyword boost + specificity):")
    st.table(pd.DataFrame(out_table))

    best = topk[0]
    best_row, best_sim, best_kwcount, best_spec, best_combined = best

    st.markdown("### Best candidate (raw)")
    st.json({
        "id": best_row.get("id",""),
        "path": best_row.get("path",""),
        "sim": best_sim,
        "keyword_matches": best_kwcount,
        "specificity": best_spec,
        "combined_score": best_combined
    })

    if best_sim >= HIGH_THRESHOLD and str(best_row.get("exclude_flag","")).lower() not in ("true","1","yes"):
        decision = "auto_accept"
    elif best_sim >= MID_THRESHOLD:
        decision = "confirm_with_llm" if (use_openai_checkbox and OPENAI_API_KEY and openai is not None) else "manual_review"
    else:
        decision = "manual_review"

    st.markdown(f"**Decision:** `{decision}` (raw_sim={best_sim:.3f}, combined={best_combined:.3f})")

    extracted = None
    if decision == "confirm_with_llm":
        st.info("Kald OpenAI for slot extraction / confirmation...")
        candidate_contexts = [c[0] for c in topk[:3]]
        extracted = call_openai_slot_extractor(combined, candidate_contexts)
        st.markdown("LLM extraction result:")
        st.json(extracted or {"error": "no result"})

    take_row = None
    if decision == "auto_accept":
        take_row = best_row
    elif decision == "confirm_with_llm" and extracted:
        intent_ok = True
        country_ok = True
        if extracted.get("intent") and extracted["intent"] not in (best_row.get("topic",""), ""):
            intent_ok = False
        if extracted.get("country_code") and extracted["country_code"] != "*" and extracted["country_code"] != best_row.get("country_code","*"):
            country_ok = False
        if intent_ok and country_ok and str(best_row.get("exclude_flag","")).lower() not in ("true","1","yes"):
            take_row = best_row
        else:
            st.warning("LLM confirmation did not match candidate sufficiently -> manual review recommended.")
            take_row = None

    # Hvis der ikke er en automatisk valgt række: generer udkast til manuel review
    if take_row is None:
        st.warning("Ingen automatisk kandidat blev accepteret. Genererer forslag til manuel review...")
        draft = None
        if OPENAI_API_KEY and openai is not None:
            try:
                draft = generate_draft_reply(combined, topk, templates)
            except Exception as e:
                st.error(f"Generering af udkast fejlede: {e}")
                draft = None

        if draft:
            st.markdown("### Forslag (auto‑genereret udkast — rediger før afsendelse)")
            subj_suggestion = draft.get("subject", "") or f"Vedr. din henvendelse"
            body_suggestion = draft.get("body", "") or ""
            manual_subj = st.text_input("Suggested subject (edit before sending):", value=subj_suggestion, key="manual_subj")
            manual_body = st.text_area("Suggested body (edit before sending):", value=body_suggestion, height=300, key="manual_body")
            st.markdown(f"**Confidence:** {draft.get('confidence', 0):.2f}")
            if draft.get("notes"):
                st.info(f"Notes: {draft.get('notes')}")
            st.success("Rediger forslaget og kopier/brug det i din mailklient. App'en sender ikke automatisk.")
        else:
            # fallback tekst hvis OpenAI ikke er tilgængelig eller fejlede
            st.info("OpenAI ikke tilgængelig eller fejl i opkald. Viser generisk fallback‑forslag.")
            fallback_lines = []
            for r, sim, mc, spec, comb in topk[:3]:
                url = r.get("requirements_url","")
                topic = r.get("topic","ukendt")
                fallback_lines.append(f"- {topic}: {url or 'ingen URL tilgængelig'}")
            fallback_body = (
                "Kære ansøger,\n\n"
                "Tak for din henvendelse. Se venligst følgende oplysninger og links, som kan være relevante:\n\n"
                + "\n".join(fallback_lines) +
                "\n\nHvis du har brug for yderligere hjælp, svar venligst på denne mail.\n\nMed venlig hilsen\nStudiekontoret"
            )
            fallback_subj = "Vedr. din henvendelse"
            st.text_input("Suggested subject (edit):", value=fallback_subj, key="fallback_subj")
            st.text_area("Suggested body (edit):", value=fallback_body, height=300, key="fallback_body")
        st.stop()

    # Fortsæt hvis take_row er valgt (auto_accept eller confirmed)
    tpl_id = take_row.get("response_template_id","")
    tpl_meta = templates.get(tpl_id)
    if not tpl_meta:
        st.error(f"Template '{tpl_id}' ikke fundet.")
        st.stop()

    allowed = tpl_meta.get("allowed_path_patterns", [])
    path_ok = False
    if not allowed:
        path_ok = True
    else:
        for pat in allowed:
            if path_matches_pattern(take_row.get("path",""), pat):
                path_ok = True
                break
    if not path_ok:
        st.warning("Template ikke tilladt for valgt path; sæt til manuel review.")
        st.stop()
    if str(tpl_meta.get("sensitivity","")).lower() in ("sensitive", "true"):
        st.warning("Template er markeret sensitive; manual handling anbefales.")
        st.stop()

    slots = {
        "applicant_name": applicant_name or (extracted.get("applicant_name") if extracted else "Ansøger"),
        "program": program or (extracted.get("program") if extracted else "det valgte program"),
        "country_name": take_row.get("country_name") or take_row.get("country_code"),
        "requirements_url": take_row.get("requirements_url",""),
        "requirements_cache": take_row.get("requirements_cache",""),
        "deadline_date": take_row.get("deadline_date",""),
        "contact_person": take_row.get("contact_person","")
    }

    try:
        subj = Template(tpl_meta.get("subject_template","")).render(**slots)
        body_out = Template(tpl_meta.get("body_template","")).render(**slots)
    except Exception as e:
        st.error(f"Fejl ved rendering: {e}")
        st.stop()

    st.markdown("### Preview af genereret svar")
    st.write("Subject:")
    st.code(subj)
    st.write("Body:")
    st.text_area("Generated body", value=body_out, height=300)

    st.markdown("### Decision metadata")
    st.json({
        "chosen_row_id": take_row.get("id",""),
        "tpl_id": tpl_id,
        "sim": best_sim,
        "combined_score": best_combined,
        "llm_extracted": extracted
    })

st.write("---")
st.caption("Hybrid prototype: sentence-transformers local embeddings + OpenAI LLM for draft generation/slot extraction. Husk at sætte OPENAI_API_KEY i miljøet.")
