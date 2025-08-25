# PDF Compliance Form App

End-to-end app to analyze a PDF with an AI Vision/LLM (Gemini), then auto-fill a checklist as **Satisfactory / Unsatisfactory / Information not present**, with **Page** and **Section**. You can review and export the results as JSON.

## Features
- Upload a PDF
- Optional OCR for scanned PDFs (Tesseract + poppler required)
- AI analysis via Gemini (set `GEMINI_API_KEY`)
- Dynamic HTML form with dropdowns
- Page/Section fields auto-enabled when status is Sat/Unsat
- Export results as JSON

## Quickstart
```bash
# 1) Create venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Configure environment
cp .env.example .env
# Edit .env to add GEMINI_API_KEY (optional, needed for AI)

# 4) Run
uvicorn app.main:app --reload
# Open http://127.0.0.1:8000/
```

## Configure Checklist Fields
Edit `app/config/fields.json` to set your own requirement list:
```json
[
  {"id": "req_1", "name": "Applicant Identity Proof"},
  {"id": "req_2", "name": "Equipment Specifications"}
]
```

## Notes
- **OCR**: If your PDFs are images, enable **Use OCR** when uploading. Install Tesseract and poppler (pdf2image requirement).
- **Gemini**: The app uses the `gemini-1.5-pro` model via `google-generativeai`. If no API key is set, the app will run in manual mode and returns a note instead of AI results (you can still fill the form and export).
- **Exports**: After export, download at `/download/results.json`.

## Customize
- Improve prompts in `build_prompt()` (app/main.py)
- Implement page-aware chunking (split pages and query LLM per field for better page localization).
- Add CSV export, database storage, or authentication as needed.
