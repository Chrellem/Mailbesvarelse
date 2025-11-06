import os
import re
import pandas as pd
import streamlit as st
from jinja2 import Template
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Config / paths (can be overridden by env vars)
REQUIREMENTS_CSV = os.getenv("REQUIREMENTS_CSV", "requirements_suggestions.csv")
TEMPLATES_CSV = os.getenv("TEMPLATES_CSV", "templates_suggestions.csv")
TEST_MODE = os.getenv("TEST_MODE", "true").lower() in ("1", "true", "yes")

st.set_page_config(page_title="Mail‑besvarelse Prototype", layout="wide")

st.title("Prototype: Mail‑besvarelse (path → template)")

st.sidebar.header("Konfiguration")
st.sidebar.text_input("Requirements CSV", value=REQUIREMENTS_CSV, key="req_path")
st.sidebar.text_input("Templates CSV", value=TEMPLATES_CSV, key="tpl_path")
test_mode = st.sidebar.checkbox("Test mode (kun preview, ikke send)", value=TEST_MODE)

@st.cache_data(ttl=60)
def load_requirements(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str).fillna("")
    return df

@st.cache_data(ttl=60)
def load_templates(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str).fillna("")
    # normalize allowed_path_patterns column
    if "allowed_path_patterns" in df.columns:
        df["allowed_path_patterns"] = df["allowed_path_patterns"].apply(lambda s: [p.strip() for p in s.split(",") if p.strip()] if s else [])
    else:
        df["allowed_path_patterns"] = [[] for _ in range(len(df))]
    return df.set_index("template_id").to_dict(orient="index")

def normalize_text(s):
    return (s or "").lower()

# Simple country detector - extend as needed
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
    return "*"  # unknown

def split_keywords(cell):
    return [k.strip().lower() for k in (cell or "").split(",") if k.strip()]

def path_specificity(path):
    # higher = more specific (fewer wildcards)
    toks = path.split("/")
    return sum(1 for t in toks if t != "*" and t != "")

def path_matches_pattern(path, pattern):
    # token-by-token match with '*' wildcard token only
    p_toks = path.split("/")
    pat_toks = pattern.split("/")
    if len(p_toks) != len(pat_toks):
        return False
    for a, b in zip(p_toks, pat_toks):
        if b == "*":
            continue
        if a != b:
            return False
    return True

def find_candidates(req_df, text, country_code):
    t = normalize_text(text)
    # filter rows by country_code or wildcard
    df = req_df[(req_df["country_code"] == country_code) | (req_df["country_code"] == "*")]
    candidates = []
    for _, row in df.iterrows():
        kws = split_keywords(row.get("question_keywords",""))
        for kw in kws:
            if kw and kw in t:
                candidates.append((row, kw))
                break
    return candidates

def choose_best(candidates):
    if not candidates:
        return None, None
    # Exclude rows with exclude_flag true
    filtered = [(r, kw) for r, kw in candidates if str(r.get("exclude_flag","")).lower() not in ("true","1","yes")]
    if not filtered:
        return candidates[0]  # fallback to first even if excluded
    # sort by specificity (desc), then priority (asc int)
    def sort_key(item):
        r, _ = item
        spec = path_specificity(r.get("path",""))
        pr = int(r.get("priority") or 1000)
        return (-spec, pr)
    filtered.sort(key=sort_key)
    return filtered[0]

def render_template(template_meta, row, slots):
    subj_tpl = template_meta.get("subject_template","")
    body_tpl = template_meta.get("body_template","")
    # Render with Jinja2
    try:
        subj = Template(subj_tpl).render(**slots)
        body = Template(body_tpl).render(**slots)
        return subj, body
    except Exception as e:
        return None, f"Error rendering template: {e}"

# Load data
req_path = st.session_state.get("req_path", REQUIREMENTS_CSV)
tpl_path = st.session_state.get("tpl_path", TEMPLATES_CSV)

req_df = load_requirements(req_path)
templates = load_templates(tpl_path)

if req_df.empty:
    st.warning(f"Requirements CSV ikke fundet eller tom: {req_path}")
if not templates:
    st.warning(f"Templates CSV ikke fundet eller tom: {tpl_path}")

st.subheader("Test input")
with st.form("email_form"):
    subject = st.text_input("Subject", value="")
    body = st.text_area("Body", value="", height=200)
    applicant_name = st.text_input("Applicant name (valgfri)", value="")
    program = st.text_input("Program (valgfri)", value="")
    submitted = st.form_submit_button("Preview svar")
if not submitted:
    st.info("Indtast en subject og/eller body, og klik 'Preview svar' for at se hvordan systemet matcher og renderer.")
else:
    combined = f"{subject}\n\n{body}"
    country = detect_country(combined)
    st.markdown(f"**Detected country_code:** `{country}`")
    candidates = find_candidates(req_df, combined, country)
    if not candidates:
        st.error("Ingen keyword‑match fundet. Forsøg med en anden formulering eller brug fallback/manuel review.")
        st.stop()
    # Show candidates
    st.write("Kandidater fundet (rad og trigger keyword):")
    cand_table = []
    for r, kw in candidates:
        cand_table.append({
            "id": r.get("id",""),
            "path": r.get("path",""),
            "country_code": r.get("country_code",""),
            "priority": r.get("priority",""),
            "matched_keyword": kw,
            "exclude_flag": r.get("exclude_flag","")
        })
    st.table(pd.DataFrame(cand_table))

    best_row, used_kw = choose_best(candidates)
    if best_row is None:
        st.error("Kun ekskluderede rækker fundet eller ingen gyldig kandidat.")
        st.stop()

    st.markdown("### Valgt række")
    st.json({
        "id": best_row.get("id",""),
        "path": best_row.get("path",""),
        "country_code": best_row.get("country_code",""),
        "topic": best_row.get("topic",""),
        "response_template_id": best_row.get("response_template_id",""),
        "requirements_url": best_row.get("requirements_url",""),
        "requirements_cache": best_row.get("requirements_cache",""),
        "priority": best_row.get("priority",""),
        "exclude_flag": best_row.get("exclude_flag",""),
        "matched_keyword": used_kw
    })

    tpl_id = best_row.get("response_template_id","")
    tpl_meta = templates.get(tpl_id)
    if not tpl_meta:
        st.error(f"Template '{tpl_id}' ikke fundet i templates CSV.")
        st.stop()

    # Check allowed patterns
    allowed = tpl_meta.get("allowed_path_patterns", [])
    path_ok = False
    if not allowed:
        path_ok = True
    else:
        for pat in allowed:
            if path_matches_pattern(best_row.get("path",""), pat):
                path_ok = True
                break
    if not path_ok:
        st.warning(f"Template '{tpl_id}' er ikke tilladt for path '{best_row.get('path','')}'. Manual review anbefales.")
    if str(tpl_meta.get("sensitivity","")).lower() in ("sensitive", "true"):
        st.warning(f"Template '{tpl_id}' er markeret som sensitive; overvej manuel behandling.")

    # Prepare slots
    slots = {
        "applicant_name": applicant_name or "Ansøger",
        "program": program or "det valgte program",
        "country_name": best_row.get("country_name") or best_row.get("country_code") or "",
        "requirements_url": best_row.get("requirements_url",""),
        "requirements_cache": best_row.get("requirements_cache",""),
        "deadline_date": best_row.get("deadline_date",""),
        "contact_person": best_row.get("contact_person","")
    }
    subj, body_out = render_template(tpl_meta, best_row, slots)
    if subj is None:
        st.error(body_out)
    else:
        st.markdown("### Preview af genereret svar")
        st.write("Subject:")
        st.code(subj)
        st.write("Body:")
        st.text_area("Generated body", value=body_out, height=300)

    st.markdown("### Debug / log")
    st.write(f"Template id: {tpl_id}  — sensitivity: {tpl_meta.get('sensitivity','')}")
    st.write(f"Test mode: {test_mode}")
    st.write("Slots brugt:")
    st.json(slots)

st.write("---")
st.caption("Denne prototype bruger keywords + path-match. For produktion: synkroniser Sheets og tilføj ML‑fallback, logging og en manuel review‑queue.")
