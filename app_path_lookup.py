import os
import csv
import json
from flask import Flask, request, jsonify
from jinja2 import Template

SHEET_PATH = os.environ.get("SHEET_PATH", "sheet_example_paths.csv")
TEMPLATES_PATH = os.environ.get("TEMPLATES_PATH", "templates_sheet_example.csv")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.json")

app = Flask(__name__)

def load_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

requirements_rows = load_csv(SHEET_PATH)
template_rows = {r['template_id']: r for r in load_csv(TEMPLATES_PATH)}

def normalize_text(s):
    return (s or "").lower()

def detect_country_from_text(text):
    # Simpelt rule: hvis landskode findes i tekst (kan udvides)
    country_keywords = {
        "dk": ["danmark", "danish", "denmark"],
        "se": ["sverige", "sweden", "svenska"],
        "cn": ["china", "kina", "gaokao"]
    }
    t = normalize_text(text)
    for code, kws in country_keywords.items():
        for kw in kws:
            if kw in t:
                return code.upper()
    return "*"  # unknown -> wildcard

def build_path_candidates(country_code, text):
    # Sammensæt kandidater ved at matche keywords i requirements_rows
    t = normalize_text(text)
    matches = []
    for r in requirements_rows:
        # kun rækker for samme country eller wildcard
        if r['country_code'] not in (country_code, '*'):
            continue
        kws = [k.strip().lower() for k in (r.get('question_keywords') or "").split(',') if k.strip()]
        for kw in kws:
            if kw and kw in t:
                matches.append(r)
                break
    return matches

def specificity_score(path):
    # højere = mere specifik (tæl non-wildcard tokens)
    tokens = path.split('/')
    return sum(1 for tok in tokens if tok != '*' and tok != '')

def path_matches_pattern(path, pattern):
    # pattern og path dele by slash, pattern kan indeholde '*' wildcard token
    ptoks = path.split('/')
    pattern_toks = pattern.split('/')
    if len(ptoks) != len(pattern_toks):
        return False
    for p, q in zip(ptoks, pattern_toks):
        if q == '*':
            continue
        if p != q:
            return False
    return True

def find_best_requirement_match(country_code, text):
    # 1) build rule matches
    matches = build_path_candidates(country_code, text)
    if not matches:
        # fallback: try all rows for country or wildcard and choose based on keywords absence
        return None
    # 2) filter excludes
    for m in matches:
        if m.get('exclude_flag','').lower() in ('true','1','yes'):
            return {'action':'excluded','row':m}
    # 3) choose most specific match -> then lowest priority
    matches_sorted = sorted(matches, key=lambda r: (-specificity_score(r['path']), int(r.get('priority') or 1000)))
    return {'action':'matched','row':matches_sorted[0]}

def render_template(row, applicant_name, program):
    tpl_id = row.get('response_template_id')
    tpl = template_rows.get(tpl_id)
    if not tpl:
        return None, None
    # check allowed path patterns
    allowed = [p.strip() for p in (tpl.get('allowed_path_patterns') or "").split(',') if p.strip()]
    # ensure at least one pattern matches the row.path (or allow *)
    if allowed:
        ok = any(path_matches_pattern(row['path'], pat) for pat in allowed)
        if not ok:
            return None, None
    slots = {
        "applicant_name": applicant_name,
        "program": program,
        "country_name": row.get('country_name',''),
        "requirements_url": row.get('requirements_url',''),
        "requirements_cache": row.get('requirements_cache','')
    }
    # render
    subj = Template(tpl['subject_template']).render(**slots)
    body = Template(tpl['body_template']).render(**slots)
    return subj, body

@app.route("/webhook/email", methods=["POST"])
def receive_email():
    data = request.json or {}
    subject = data.get("subject","")
    body = data.get("body","")
    from_email = data.get("from","")
    applicant_name = data.get("applicant_name","Ansøger")
    program = data.get("program","det valgte program")

    text = subject + " " + body
    country = detect_country_from_text(text)  # simple detector
    match = find_best_requirement_match(country, text)

    if not match:
        return jsonify({"status":"no_match","action":"manual_review"}), 200
    if match['action'] == 'excluded':
        subj, body_out = render_template(match['row'], applicant_name, program)
        return jsonify({"status":"excluded","to":from_email,"subject":subj,"body":body_out}), 200
    row = match['row']
    subj, body_out = render_template(row, applicant_name, program)
    if not subj:
        return jsonify({"status":"template_missing_or_not_allowed","action":"manual_review"}), 200
    return jsonify({"status":"ok","mode":"test","to":from_email,"subject":subj,"body":body_out,"matched_path":row['path']}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)