```markdown
# Mailbesvarelse-prototype (fokus: adgangsgivende uddannelser pr. land)

Dette repo indeholder en simpel prototype der:
- Klassificerer indkommende mails med fokus på "adgangsgivende uddannelser" per land.
- Henter svarskabeloner fra templates/ og indhold fra et simpelt CSV-sheet.
- Understøtter en liste over emner/keywords der skal være udelukket fra automatiske svar (blacklist).
- Returnerer forhåndsvisning af svar i stedet for at sende direkte (test-mode).

Filer i eksemplet:
- sheet_example.csv — data for landespecifikke krav, templates og keywords.
- templates/*.txt — mail-skabeloner (subject: ... i første linje).
- app.py — minimal Flask-app med rule-based klassifikation og exclusion-check.
- config.json — konfigurationsfil med global blacklist.
- requirements.txt — Python-afhængigheder.

Kør lokalt:
1. Opret et virtuelt miljø og installer requirements.txt
2. Sæt environment variabel SHEET_PATH hvis du vil bruge en anden CSV
3. Kør `python app.py`
4. POST til /webhook/email med JSON: {"subject":"...", "body":"...", "from":"ansøger@example.com", "applicant_name":"Navn", "program":"Biologi"}

Notes:
- For produktion: integrer Google Sheets (Service Account) i stedet for CSV, tilføj send via SMTP/Gmail API, log til DB, og evt. ML/LLM-intent-classifier.
- Sørg for at vedligeholde blacklist i config.json eller en admin-UI for at styre hvilke emner der aldrig autosvareres.
```