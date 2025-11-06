# Opdateret streamlit prototype med deadline-detektion
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
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
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
    """
    Return cleaned keywords list.
    - Trim whitespace
    - Lowercase
    - Keep numeric tokens and short tokens if they contain digits or punctuation (so '5 juli' can be a keyword)
    - Ignore overly generic tokens like 'ap' unless explicitly present as phrase 'ap-test'
    """
    out = []
    for k in (cell or "").split(","):
        k = k.strip().lower()
        if not k:
            continue
        # keep tokens with digits (dates like '5 juli') and tokens >= 3 chars,
        # or tokens containing '-' or space (phrases like 'ap-test' or '5 juli')
        if len(k) < 3 and ("-" not in k and " " not in k and not any(ch.isdigit() for ch in k)):
            # ignore too-short non-numeric tokens like 'ap'
            continue
        out.append(k)
    return out

def path_specificity(path):
    toks = path.split("/")
    return sum(1 for t in toks if t != "*" and t != "")

def path_matches_pattern(path, pattern):
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

def keyword_matches_text(text, kw):
    """
    Use regex word boundaries for keyword matching.
    For keywords with digits or containing spaces/dashes, build a phrase match.
    """
    if not kw:
        return False
    kw_escaped = re.escape(kw)
    # If kw contains digits or non-letter (space or dash), match the phrase exactly (still case-insensitive)
    if re.search(r'[\d\W]', kw):
        pattern = r'(?i)\b' + kw_escaped + r'\b'
    else:
        pattern = r'(?i)\b' + kw_escaped + r'\b'
    try:
        return re.search(pattern, text, flags=0) is not None
    except re.error:
        return kw in text

# New: detect explicit deadline references (e.g. "5 juli", "5 July", "5. juli", "5/7", "5.7")
DEADLINE_PATTERNS = [
    r'\b5[\s\./-]*(?:juli|jul|july)\b',
    r'\b5[\s\./-]*(?:7|07)\b',           # 5/7 or 5.7 forms
    r'\b5(?:th|st|nd|rd)?\s*(?:july|juli)\b',
    r'\bJuly\s*5\b',
    r'\b5\s*July\b'
]

def detect_deadline_reference(text):
    t = normalize_text(text)
    for pat in DEADLINE_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            # return the matched string for debug
            m = re.search(pat, t, flags=re.IGNORECASE)
            return True, (m.group(0) if m else "5 juli")
    return False, None

def find_candidates(req_df, text, country_code):
    """
    Return list of candidate tuples: (row_series, matched_keywords_list, match_count)
    This function now also tries a special deadline-detection before normal keyword matching.
    """
    t = normalize_text(text)
    candidates = []

    # 1) deadline detection: if found, add all deadline rows as candidates (country-specific first)
    found_deadline, matched_deadline = detect_deadline_reference(text)
    if found_deadline:
        # prefer rows where topic == 'deadlines' or path contains '/deadlines/'
        df_deadline = req_df[(req_df["topic"].str.lower() == "deadlines") | (req_df["path"].str.contains("/deadlines/"))]
        # also allow country-specific deadlines first
        df_deadline_specific = df_deadline[(df_deadline["country_code"] == country_code) | (df_deadline["country_code"] == "*")]
        for _, row in df_deadline_specific.iterrows():
            candidates.append((row, [matched_deadline], 1))
        if candidates:
            return candidates

    # 2) normal keyword matching against rows for country_code or wildcard
    df = req_df[(req_df["country_code"] == country_code) | (req_df["country_code"] == "*")]
    for _, row in df.iterrows():
        kws = split_keywords(row.get("question_keywords",""))
        matched = []
        for kw in kws:
            if keyword_matches_text(t, kw):
                matched.append(kw)
        if matched:
            matched_unique = sorted(set(matched), key=lambda x: matched.index(x))
            candidates.append((row, matched_unique, len(matched_unique)))

    return candidates

def choose_best(candidates):
    if not candidates:
        return None, None, 0
    non_excluded = [c for c in candidates if str(c[0].get("exclude_flag","")).lower() not in ("true","1","yes")]
    pool = non_excluded if non_excluded else candidates
    def sort_key(item):
        row, matched, count = item
        spec = path_specificity(row.get("path",""))
        pr = int(row.get("priority") or 1000)
        return (-count, -spec, pr)
    pool_sorted = sorted(pool, key=sort_key)
    return pool_sorted[0]  # (row, matched_keywords, match_count)

def render_template(template_meta, row, slots):
    subj_tpl = template_meta.get("subject_template","")
    body_tpl = template_meta.get("body_template","")
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

    # Find candidates (deadline detection included)
    candidates = find_candidates(req_df, combined, country)
    if not candidates:
        st.error("Ingen keyword‑match fundet. Forsøg med en anden formulering eller brug fallback/manuel review.")
        st.stop()

    # Show candidates
    st.write("Kandidater fundet (rad, matched keywords og match_count):")
    cand_table = []
    for r, matched, count in candidates:
        cand_table.append({
            "id": r.get("id",""),
            "path": r.get("path",""),
            "country_code": r.get("country_code",""),
            "priority": r.get("priority",""),
            "matched_keywords": ", ".join(matched),
            "match_count": count,
            "exclude_flag": r.get("exclude_flag","")
        })
    st.table(pd.DataFrame(cand_table))

    best_row, best_matched, best_count = choose_best(candidates)
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
        "matched_keywords": best_matched,
        "match_count": best_count
    })

    tpl_id = best_row.get("response_template_id","")
    tpl_meta = templates.get(tpl_id)
    if not tpl_meta:
        st.error(f"Template '{tpl_id}' ikke fundet i templates CSV.")
        st.stop()

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
