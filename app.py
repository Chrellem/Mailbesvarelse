import os
import csv
from flask import Flask, request, jsonify
from jinja2 import Template
from datetime import datetime

# Simpelt reference-flow: in-memory CSV-læsning + rule-based keyword match

SHEET_PATH = os.environ.get("SHEET_PATH", "sheet_example.csv")

app = Flask(__name__)

def load_sheet(path=SHEET_PATH):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def classify_mail_simple(subject, body):
    text = (subject + " " + body).lower()
    # Eksempelregelsæt: check keywords i sheet
    sheet = load_sheet()
    matches = []
    for row in sheet:
        keywords = row.get("question_keywords","")
        for kw in [k.strip().lower() for k in keywords.split(',') if k.strip()]:
            if kw and kw in text:
                matches.append(row)
                break
    # Return first match or None
    return matches[0] if matches else None

def render_template_from_row(row, slots):
    tpl_id = row.get("response_template_id")
    if not tpl_id:
        return None, None
    tpl_path = f"templates/{tpl_id}.txt"
    if not os.path.exists(tpl_path):
        return None, None
    with open(tpl_path, encoding='utf-8') as f:
        tpl_text = f.read()
    t = Template(tpl_text)
    body = t.render(**slots)
    subject = ""
    # find subject line in template
    if tpl_text.splitlines():
        first_line = tpl_text.splitlines()[0]
        if first_line.lower().startswith("subject:"):
            # render subject separately
            sub_tpl = Template(first_line[len("subject:"):].strip())
            subject = sub_tpl.render(**slots)
            # remove subject line from body if needed
            body = "\n".join(tpl_text.splitlines()[1:])
            body = Template(body).render(**slots)
    return subject, body

@app.route("/webhook/email", methods=["POST"])
def receive_email():
    data = request.json
    subject = data.get("subject","")
    body = data.get("body","")
    from_email = data.get("from","")
    # Klassificer
    matched_row = classify_mail_simple(subject, body)
    if not matched_row:
        return jsonify({"status":"no_match","action":"manual_review"}), 200
    # Prepare slots
    slots = {
        "applicant_name": data.get("applicant_name","Ansøger"),
        "program": data.get("program","det valgte program"),
        "country_name": matched_row.get("country_name",""),
        "requirements_url": matched_row.get("requirements_url",""),
        "requirements_cache": matched_row.get("requirements_cache","")
    }
    subject_out, body_out = render_template_from_row(matched_row, slots)
    if not subject_out:
        return jsonify({"status":"template_missing","action":"manual_review"}), 200
    # TODO: send email via SMTP or Gmail API. Her returneres blot preview.
    return jsonify({
        "status":"ok",
        "to": from_email,
        "subject": subject_out,
        "body": body_out
    }), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)