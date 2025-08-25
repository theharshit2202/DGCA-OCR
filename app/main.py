import os
import io
import json
import re
import fitz  # PyMuPDF
import time
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Optional LLM (Gemini)
try:
    import google.generativeai as genai
    print("[INFO] Google Generative AI imported successfully")
except Exception as e:
    print(f"[WARN] Google Generative AI import failed: {e}")
    genai = None

# Optional OCR support
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    print("[INFO] OCR libraries imported successfully")
except Exception as e:
    print(f"[WARN] OCR libraries import failed: {e}")
    pytesseract = None

load_dotenv()
print("[INFO] Environment variables loaded")

# --- FastAPI & static/templates setup (REPLACE THIS WHOLE BLOCK) ---
print("[INFO] Initializing FastAPI app and static/template paths...")

app = FastAPI(title="PDF Compliance Form App")

# Resolve safe defaults if your files are not under app/static or app/templates.
static_candidates = [
    "app/static",      # original
    ".",               # project root (where styles.css/app.js were uploaded)
    "static",          # common alt
]
templates_candidates = [
    "app/templates",   # original
    "templates",       # common alt
    ".",
]

def _first_existing_path(paths):
    for p in paths:
        if os.path.isdir(p):
            print(f"[DEBUG] Using path: {os.path.abspath(p)}")
            return p
        else:
            print(f"[WARN] Path not found (skipping): {os.path.abspath(p)}")
    # fall back to current dir; still mount to avoid crashes
    print("[WARN] No template/static path found from candidates; falling back to current directory")
    return "."

_static_dir = _first_existing_path(static_candidates)
_templates_dir = _first_existing_path(templates_candidates)

print(f"[INFO] Mounting static from: {_static_dir}")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

print(f"[INFO] Using Jinja2 templates dir: {_templates_dir}")
templates = Jinja2Templates(directory=_templates_dir)
# --- end replacement block ---

# ------------- Config -------------
# Define the checklist fields (edit app/config/fields.json to customize)
DEFAULT_FIELDS_PATH = "app/config/fields.json"

# ------------- Models -------------
class FieldResult(BaseModel):
    field_id: str
    field_name: str
    status: str  # Satisfactory | Unsatisfactory | Information not present
    reason: str | None = None
    page: int | None = None
    section: str | None = None
    information_found: str | None = None  # New field for extracted information

class AnalyzeResponse(BaseModel):
    results: List[FieldResult]

# ------------- Helpers -------------
def load_fields() -> List[Dict[str, Any]]:
    print(f"[INFO] Loading fields from {DEFAULT_FIELDS_PATH}")
    try:
        with open(DEFAULT_FIELDS_PATH, "r", encoding="utf-8") as f:
            fields = json.load(f)
            print(f"[SUCCESS] Loaded {len(fields)} fields: {[f['name'] for f in fields]}")
            return fields
    except Exception as e:
        print(f"[ERROR] Failed to load fields: {e}")
        return []

def extract_text_pages_pdf(file_bytes: bytes, use_ocr: bool = False) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts: [{page: 1, text: "..."}]
    For scanned PDFs, set use_ocr=True (requires pytesseract and poppler).
    """
    print(f"[INFO] Starting PDF text extraction. File size: {len(file_bytes)} bytes, OCR: {use_ocr}")
    pages: List[Dict[str, Any]] = []
    
    if use_ocr and pytesseract is not None:
        print("[INFO] Using OCR path for text extraction...")
        # OCR path
        try:
            images = convert_from_bytes(file_bytes)
            print(f"[INFO] Converted PDF to {len(images)} images")
            for idx, img in enumerate(images, start=1):
                print(f"[INFO] Processing image {idx}/{len(images)} with OCR...")
                text = pytesseract.image_to_string(img)
                text_length = len(text.strip())
                print(f"[INFO] Page {idx}: Extracted {text_length} characters via OCR")
                pages.append({"page": idx, "text": text})
        except Exception as e:
            print(f"[ERROR] OCR processing failed: {e}")
            return []
    else:
        print("[INFO] Using native PyMuPDF text extraction...")
        # Native text extraction via PyMuPDF
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)
            print(f"[INFO] PDF opened successfully. Total pages: {total_pages}")
            
            for i, page in enumerate(doc, start=1):
                print(f"[INFO] Processing page {i}/{total_pages}...")
                text = page.get_text("text") or ""
                text_length = len(text.strip())
                print(f"[INFO] Page {i}: Extracted {text_length} characters")
                
                if not text.strip():
                    print(f"[WARN] No text detected on page {i}. This might be a scanned/image page.")
                
                pages.append({"page": i, "text": text})
            
            doc.close()
            print(f"[SUCCESS] Text extraction complete. Processed {len(pages)} pages")
            
        except Exception as e:
            print(f"[ERROR] PyMuPDF processing failed: {e}")
            return []
    
    return pages

def extract_pdf_pages(file_bytes: bytes, use_ocr: bool = False) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts: [{page: 1, text: "...", image: PIL.Image}]
    For scanned PDFs, set use_ocr=True to also extract images for Gemini analysis.
    """
    print(f"[INFO] Starting PDF processing. File size: {len(file_bytes)} bytes, OCR mode: {use_ocr}")
    pages: List[Dict[str, Any]] = []
    
    try:
        # Always convert to images for Gemini analysis
        print("[INFO] Converting PDF to images for Gemini analysis...")
        images = convert_from_bytes(file_bytes)
        print(f"[INFO] Converted PDF to {len(images)} images")
        
        for idx, img in enumerate(images, start=1):
            print(f"[INFO] Processing page {idx}/{len(images)}...")
            
            # Extract text using OCR for reference
            if pytesseract is not None:
                text = pytesseract.image_to_string(img)
                text_length = len(text.strip())
                print(f"[INFO] Page {idx}: Extracted {text_length} characters via OCR")
            else:
                text = ""
                print(f"[WARN] OCR not available, no text extracted for page {idx}")
            
            # Store both image and text
            pages.append({
                "page": idx, 
                "text": text,
                "image": img  # Store PIL Image for Gemini
            })
            
        print(f"[SUCCESS] PDF processing complete. Pages: {len(pages)}")
        return pages
        
    except Exception as e:
        print(f"[ERROR] PDF processing failed: {e}")
        return []

def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[WARN] GEMINI_API_KEY not found in environment variables")
        return '{"note": "Gemini not configured. Provide GEMINI_API_KEY to enable AI analysis."}'
    
    if genai is None:
        print("[WARN] Google Generative AI library not available")
        return '{"note": "Gemini library not available. Install google-generativeai package."}'
    
    print("[INFO] Configuring Gemini API...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        print("[INFO] Sending prompt to Gemini API...")
        print(f"[DEBUG] Prompt length: {len(prompt)} characters")
        
        start_time = time.time()
        resp = model.generate_content(prompt)
        end_time = time.time()
        
        print(f"[SUCCESS] Gemini API response received in {end_time - start_time:.2f} seconds")
        print(f"[DEBUG] Response length: {len(resp.text)} characters")
        
        return resp.text.strip()
        
    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {e}")
        return f'{{"error": "Gemini API call failed: {str(e)}"}}'

def clean_gemini_response(response: str) -> str:
    """Clean Gemini response to fix JSON parsing issues"""
    print(f"[INFO] Cleaning Gemini response for JSON parsing...")
    
    # Remove any markdown formatting
    cleaned = response.strip()
    
    # Find JSON content between ```json and ``` or just the JSON part
    if "```json" in cleaned:
        start = cleaned.find("```json") + 7
        end = cleaned.find("```", start)
        if end != -1:
            cleaned = cleaned[start:end].strip()
    elif "```" in cleaned:
        start = cleaned.find("```") + 3
        end = cleaned.find("```", start)
        if end != -1:
            cleaned = cleaned[start:end].strip()
    
    # Remove any leading/trailing text that's not JSON
    if cleaned.startswith("{"):
        # Find the last closing brace
        brace_count = 0
        for i, char in enumerate(cleaned):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    cleaned = cleaned[:i+1]
                    break
    
    # Clean up any invalid control characters
    cleaned = "".join(char for char in cleaned if ord(char) >= 32 or char in "\n\r\t")
    
    print(f"[DEBUG] Cleaned response length: {len(cleaned)} characters")
    print(f"[DEBUG] Cleaned response preview: {cleaned[:200]}...")
    
    return cleaned

def call_gemini_with_images(prompt: str, images: List) -> str:
    """Send prompt with images directly to Gemini for visual analysis"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[WARN] GEMINI_API_KEY not found in environment variables")
        return '{"note": "Gemini not configured. Provide GEMINI_API_KEY to enable AI analysis."}'
    
    if genai is None:
        print("[WARN] Google Generative AI library not available")
        return '{"note": "Gemini library not available. Install google-generativeai package."}'
    
    print(f"[INFO] Sending prompt with {len(images)} images to Gemini API...")
    print(f"[DEBUG] Prompt length: {len(prompt)} characters")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Prepare images for Gemini - send PIL Image objects directly
        gemini_images = []
        for img in images:
            # Gemini expects PIL Image objects directly, not bytes
            gemini_images.append(img)
        
        start_time = time.time()
        
        # Send prompt with images
        response = model.generate_content([prompt] + gemini_images)
        
        end_time = time.time()
        
        print(f"[SUCCESS] Gemini API response received in {end_time - start_time:.2f} seconds")
        print(f"[DEBUG] Response length: {len(response.text)} characters")
        
        return response.text.strip()
        
    except Exception as e:
        print(f"[ERROR] Gemini API call with images failed: {e}")
        return f'{{"error": "Gemini API call with images failed: {str(e)}"}}'

def build_prompt(all_text: str, field_name: str) -> str:
    print(f"[INFO] Building prompt for field: {field_name}")
    print(f"[DEBUG] Total text length: {len(all_text)} characters")
    
    prompt = f"""
You are an AI compliance reviewer. Determine whether the required information for the field '{field_name}' is present in the document text below.
If present, evaluate if it meets expectations (Satisfactory) or has issues (Unsatisfactory). If absent, mark 'Information not present'.

Document Text (may be long, consider headings/sections if visible):
---
{all_text[:200000]}
---

Return ONLY valid JSON, no markdown, in the following schema:
{{
  "status": "Satisfactory" | "Unsatisfactory" | "Information not present",
  "reason": "short concise justification",
  "page": 12,
  "section": "heading/subheading or brief locator if any",
  "information_found": "actual text/information found for this field, or null if not found"
}}
"""
    print(f"[DEBUG] Generated prompt length: {len(prompt)} characters")
    return prompt

def analyze_fields_with_images_and_llm(pages: List[Dict[str, Any]], fields: List[Dict[str, Any]], use_ocr: bool = False) -> List[FieldResult]:
    print(f"[INFO] Starting OCR-first analysis for {len(fields)} fields across {len(pages)} pages")
    print(f"[INFO] OCR mode: {use_ocr}")
    
    try:
        # Step 1: ALWAYS try OCR first for all fields
        print(f"[INFO] Step 1: Attempting to fill ALL fields using OCR text extraction...")
        ocr_results = fill_fields_with_ocr_text(fields, pages, "".join([p["text"] for p in pages]))
        
        # Step 2: Identify fields that couldn't be filled by OCR
        unfilled_fields = [f for f in ocr_results if f.status == "Information not present"]
        filled_fields = [f for f in ocr_results if f.status != "Information not present"]
        
        print(f"[INFO] OCR filled {len(filled_fields)} fields, {len(unfilled_fields)} fields need LLM analysis")
        
        # Step 3: Only use LLM for fields that OCR couldn't fill
        if unfilled_fields:
            print(f"[INFO] Step 2: Using LLM to analyze {len(unfilled_fields)} unfilled fields...")
            
            # Try text-based LLM first (faster and cheaper)
            try:
                llm_results = analyze_unfilled_fields_with_llm_text(unfilled_fields, "".join([p["text"] for p in pages]))
                print(f"[INFO] Text-based LLM analysis successful")
            except Exception as e:
                print(f"[WARN] Text-based LLM failed: {e}, trying image-based approach...")
                # Fallback to image-based analysis if text LLM fails
                images = [p["image"] for p in pages if "image" in p]
                llm_results = analyze_unfilled_fields_with_gemini_images(unfilled_fields, images)
            
            # Combine OCR results with LLM results - ensure ALL fields are included
            final_results = []
            for field in fields:
                # Find OCR result
                ocr_result = next((r for r in ocr_results if r.field_id == field["id"]), None)
                # Find LLM result (if field was unfilled)
                llm_result = next((r for r in llm_results if r.field_id == field["id"]), None)
                
                if ocr_result and ocr_result.status != "Information not present":
                    # Use OCR result (priority)
                    final_results.append(ocr_result)
                    print(f"[INFO] Using OCR result for '{field['name']}': {ocr_result.status}")
                elif llm_result and llm_result.status != "Information not present":
                    # Use LLM result for unfilled fields - update status to Satisfactory if found
                    if llm_result.information_found:
                        llm_result.status = "Satisfactory"
                        print(f"[INFO] LLM filled field '{field['name']}' - Status updated to Satisfactory")
                    final_results.append(llm_result)
                else:
                    # Use OCR result if available, otherwise create fallback
                    if ocr_result:
                        final_results.append(ocr_result)
                    else:
                        final_results.append(FieldResult(
                            field_id=field["id"],
                            field_name=field["name"],
                            status="Information not present",
                            reason="Analysis failed",
                            page=None,
                            section=None,
                            information_found=None
                        ))
            
            print(f"[SUCCESS] Combined OCR + LLM analysis complete. Total results: {len(final_results)}")
            return final_results
        else:
            print(f"[SUCCESS] OCR filled all fields! No LLM needed.")
            return ocr_results
            
    except Exception as e:
        print(f"[ERROR] OCR-first analysis failed: {e}")
        print(f"[INFO] Falling back to comprehensive LLM analysis...")
        return analyze_fields_with_llm_fallback(pages, fields, "".join([p["text"] for p in pages]))

def fill_fields_with_ocr_text(fields: List[Dict[str, Any]], pages: List[Dict[str, Any]], all_text: str) -> List[FieldResult]:
    """Attempt to fill fields using OCR/extracted text without LLM"""
    print(f"[INFO] Attempting to fill {len(fields)} fields using extracted text...")
    
    results = []
    for field in fields:
        field_name = field["name"]
        field_id = field["id"]
        
        # Try to find information for this field in the extracted text
        found_info = find_field_information_in_text(field_name, all_text, pages)
        
        if found_info:
            # Field found via text analysis
            status = "Satisfactory" if found_info["confidence"] > 0.7 else "Unsatisfactory"
            results.append(FieldResult(
                field_id=field_id,
                field_name=field_name,
                status=status,
                reason=f"Found via text analysis (confidence: {found_info['confidence']:.2f})",
                page=found_info["page"],
                section=found_info["section"],
                information_found=found_info["text"]
            ))
            print(f"[SUCCESS] OCR filled field '{field_name}' - Status: {status}, Page: {found_info['page']}")
        else:
            # Field not found
            results.append(FieldResult(
                field_id=field_id,
                field_name=field_name,
                status="Information not present",
                reason="Not found in extracted text",
                page=None,
                section=None,
                information_found=None
            ))
            print(f"[INFO] OCR could not fill field '{field_name}'")
    
    return results

def find_field_information_in_text(field_name: str, all_text: str, pages: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Intelligently search for field information in extracted text across ALL pages"""
    
    # Define keywords and patterns for different field types
    field_patterns = {
        "Applicant Identity Proof": ["identity", "proof", "id", "passport", "driving license", "aadhar", "pan", "voter id"],
        "Equipment Specifications": ["equipment", "specifications", "model", "serial", "technical", "specs", "type", "brand"],
        "Training/Certification Details": ["training", "certification", "certificate", "course", "qualification", "license", "diploma"],
        "Compliance with DGCA Guidelines": ["dgca", "compliance", "guidelines", "regulations", "standards", "approval", "certified"],
        "Maintenance Logs / Service History": ["maintenance", "logs", "service", "history", "repair", "inspection", "overhaul"],
        "LIC Confirmation - Payee Name": ["payee", "name", "lic", "life insurance", "policy holder", "insured", "beneficiary"],
        "LIC Confirmation - To Whom Payment Made": ["payment", "made", "to", "whom", "beneficiary", "recipient", "payee"],
        "LIC Confirmation - Payment Date": ["payment date", "date", "when", "paid", "transaction date", "due date", "receipt date"],
        "LIC Confirmation - Payment Mode": ["payment mode", "mode", "method", "cash", "cheque", "online", "transfer", "card"],
        "LIC Confirmation - Policy Number": ["policy number", "policy", "number", "lic policy", "policy id", "policy no"],
        "LIC Confirmation - Payment Status": ["payment status", "status", "successful", "unsuccessful", "pending", "failed", "cleared"]
    }
    
    # Get relevant keywords for this field
    keywords = field_patterns.get(field_name, [field_name.lower().split()])
    
    # Search for exact field name matches first (highest priority)
    field_name_lower = field_name.lower()
    
    # Look for exact field name matches across ALL pages
    for page in pages:
        page_text_lower = page["text"].lower()
        if field_name_lower in page_text_lower:
            # Extract context around the field name
            start_idx = page_text_lower.find(field_name_lower)
            context_start = max(0, start_idx - 200)
            context_end = min(len(page["text"]), start_idx + len(field_name_lower) + 200)
            context = page["text"][context_start:context_end]
            
            # Try to find a heading or section name
            section = find_nearest_heading(page["text"], start_idx)
            
            print(f"[DEBUG] Found exact field name '{field_name}' on page {page['page']} in section: {section}")
            return {
                "text": context.strip(),
                "page": page["page"],  # Use actual page number from page object
                "section": section,
                "confidence": 0.9
            }
    
    # Look for keyword matches across ALL pages (second priority)
    for keyword in keywords:
        if isinstance(keyword, str):  # Ensure keyword is a string
            keyword_lower = keyword.lower()
            
            # Search across all pages for this keyword
            for page in pages:
                page_text_lower = page["text"].lower()
                if keyword_lower in page_text_lower:
                    # Extract context around the keyword
                    start_idx = page_text_lower.find(keyword_lower)
                    context_start = max(0, start_idx - 150)
                    context_end = min(len(page["text"]), start_idx + len(keyword) + 150)
                    context = page["text"][context_start:context_end]
                    
                    # Try to find a heading or section name
                    section = find_nearest_heading(page["text"], start_idx)
                    
                    print(f"[DEBUG] Found keyword '{keyword}' for field '{field_name}' on page {page['page']} in section: {section}")
                    return {
                        "text": context.strip(),
                        "page": page["page"],  # Use actual page number from page object
                        "section": section,
                        "confidence": 0.7
                    }
    
    # No information found
    print(f"[DEBUG] No information found for field '{field_name}' across all {len(pages)} pages")
    return None

def find_nearest_heading(text: str, position: int) -> str:
    """Find the nearest heading or section name before the given position"""
    # Look for common heading patterns
    heading_patterns = [
        r'^[A-Z][A-Z\s]+$',  # ALL CAPS headings
        r'^\d+\.\s+[A-Z][a-z\s]+',  # Numbered sections like "1. Introduction"
        r'^[A-Z][a-z\s]+:$',  # Title case with colon
        r'^[A-Z][a-z\s]+$',   # Title case headings
    ]
    
    # Search backwards from the position for headings
    lines = text.split('\n')
    current_line = 0
    char_count = 0
    
    for i, line in enumerate(lines):
        line_length = len(line) + 1  # +1 for newline
        if char_count <= position < char_count + line_length:
            current_line = i
            break
        char_count += line_length
    
    # Look backwards for the most recent heading
    for i in range(current_line, max(0, current_line - 10), -1):
        line = lines[i].strip()
        if line and any(re.match(pattern, line) for pattern in heading_patterns):
            return line[:100]  # Limit length
    
    # If no heading found, look for context words
    context_start = max(0, position - 100)
    context_end = min(len(text), position + 100)
    context = text[context_start:context_end]
    
    # Look for any capitalized words that might be section names
    words = context.split()
    for word in words:
        if word and word[0].isupper() and len(word) > 3:
            return f"Context: {word}"
    
    return "General content"

def analyze_unfilled_fields_with_llm_text(unfilled_fields: List[FieldResult], all_text: str) -> List[FieldResult]:
    """Use text-based LLM to analyze only the fields that OCR couldn't fill"""
    print(f"[INFO] Using text-based LLM to analyze {len(unfilled_fields)} unfilled fields")
    
    # Build comprehensive prompt for all unfilled fields
    field_names = [f.field_name for f in unfilled_fields]
    
    # Determine text length based on document size
    text_length = len(all_text)
    if text_length > 1000000:  # More than 1MB of text (very large PDF)
        max_text = 500000  # Use 500K characters for very large PDFs
        print(f"[INFO] Large PDF detected ({text_length} chars), using {max_text} chars for LLM analysis")
    elif text_length > 500000:  # More than 500K characters (large PDF)
        max_text = 300000  # Use 300K characters for large PDFs
        print(f"[INFO] Large PDF detected ({text_length} chars), using {max_text} chars for LLM analysis")
    else:
        max_text = 200000  # Default for normal PDFs
        print(f"[INFO] Normal PDF size ({text_length} chars), using {max_text} chars for LLM analysis")
    
    comprehensive_prompt = f"""
You are an AI compliance reviewer. The following fields could not be filled using OCR text extraction, so please analyze them using AI.

Unfilled Fields to Analyze:
{chr(10).join([f"{i+1}. {name}" for i, name in enumerate(field_names)])}

Document Text (analyze thoroughly across all content):
---
{all_text[:max_text]}
---

IMPORTANT INSTRUCTIONS:
1. Search through ALL the provided text content thoroughly
2. Only fill page numbers when you actually find information for a field
3. If no information is found, set page to null and status to "Information not present"
4. For large documents, information might be spread across different sections
5. Be thorough in your search - don't miss information
6. Look for headings, sections, and contextual information to populate the section field

Return ONLY valid JSON, no markdown, in the following schema:
{{
  "fields": [
    {{
      "field_name": "exact field name from the list above",
      "status": "Satisfactory" | "Unsatisfactory" | "Information not present",
      "reason": "short concise justification",
      "page": 12,
      "section": "heading/subheading or brief locator if any",
      "information_found": "actual text/information found for this field, or null if not found"
    }}
  ]
}}

Analyze each field thoroughly and provide accurate page numbers and sections where information is found.
"""
    
    print(f"[INFO] Sending comprehensive text-based LLM prompt for {len(unfilled_fields)} unfilled fields")
    
    try:
        raw_response = call_gemini(comprehensive_prompt)
        
        # Clean the response to fix JSON parsing issues
        cleaned_response = clean_gemini_response(raw_response)
        
        data = json.loads(cleaned_response)
        extracted_fields = data.get("fields", [])
        
        print(f"[SUCCESS] Parsed text-based LLM response with {len(extracted_fields)} field results")
        
        # Map the extracted results to our field structure
        results: List[FieldResult] = []
        for unfilled_field in unfilled_fields:
            field_name = unfilled_field.field_name
            
            # Find matching result from LLM response
            matching_result = None
            for extracted in extracted_fields:
                if extracted.get("field_name") == field_name:
                    matching_result = extracted
                    break
            
            if matching_result:
                status = matching_result.get("status", "Information not present")
                reason = matching_result.get("reason")
                page = matching_result.get("page")
                section = matching_result.get("section")
                information_found = matching_result.get("information_found")
                
                # Update status to Satisfactory if information was found
                if information_found and status == "Information not present":
                    status = "Satisfactory"
                    print(f"[INFO] Updating status to Satisfactory for '{field_name}' as information was found")
                
                print(f"[SUCCESS] Text-based LLM filled field '{field_name}' - Status: {status}, Page: {page}, Section: {section}")
                
            else:
                # Keep original unfilled status
                status = "Information not present"
                reason = "Text-based LLM could not find information"
                page = None
                section = None
                information_found = None
                print(f"[WARN] Text-based LLM could not fill field '{field_name}'")
            
            results.append(FieldResult(
                field_id=unfilled_field.field_id,
                field_name=field_name,
                status=status,
                reason=reason,
                page=page if isinstance(page, int) else None,
                section=section if isinstance(section, str) else None,
                information_found=information_found
            ))
        
        print(f"\n[SUCCESS] Text-based LLM analysis complete for {len(results)} unfilled fields")
        return results
        
    except Exception as e:
        print(f"[ERROR] Text-based LLM analysis failed: {e}")
        print(f"[ERROR] Raw response: {raw_response[:500] if 'raw_response' in locals() else 'No response'}")
        raise e  # Re-raise to trigger fallback to image-based approach

def analyze_unfilled_fields_with_gemini_images(unfilled_fields: List[FieldResult], images: List) -> List[FieldResult]:
    """Use Gemini with images to analyze only the fields that OCR couldn't fill"""
    print(f"[INFO] Using Gemini with images to analyze {len(unfilled_fields)} unfilled fields")
    
    # Build comprehensive prompt for all unfilled fields
    field_names = [f.field_name for f in unfilled_fields]
    
    comprehensive_prompt = f"""
You are an AI compliance reviewer analyzing scanned document images. The following fields could not be filled using text extraction, so please analyze the images visually.

Unfilled Fields to Analyze:
{chr(10).join([f"{i+1}. {name}" for i, name in enumerate(field_names)])}

IMPORTANT INSTRUCTIONS:
1. Analyze ALL the provided images thoroughly and visually
2. Look for visual information related to each field (text, tables, forms, signatures, etc.)
3. Only fill page numbers when you actually find information for a field
4. If no information is found, set page to null and status to "Information not present"
5. For multi-page documents, information might be spread across different pages
6. Be thorough in your visual analysis - don't miss information
7. Consider both text content and visual layout/structure

Return ONLY valid JSON, no markdown, in the following schema:
{{
  "fields": [
    {{
      "field_name": "exact field name from the list above",
      "status": "Satisfactory" | "Unsatisfactory" | "Information not present",
      "reason": "short concise justification",
      "page": 12,
      "section": "heading/subheading or brief locator if any",
      "information_found": "actual text/information found for this field, or null if not found"
    }}
  ]
}}

Analyze each field thoroughly by examining the images and provide accurate page numbers and sections where information is found.
"""
    
    print(f"[INFO] Sending comprehensive Gemini prompt with {len(images)} images for {len(unfilled_fields)} unfilled fields")
    
    try:
        raw_response = call_gemini_with_images(comprehensive_prompt, images)
        
        # Clean the response to fix JSON parsing issues
        cleaned_response = clean_gemini_response(raw_response)
        
        data = json.loads(cleaned_response)
        extracted_fields = data.get("fields", [])
        
        print(f"[SUCCESS] Parsed Gemini response with {len(extracted_fields)} field results")
        
        # Map the extracted results to our field structure
        results: List[FieldResult] = []
        for unfilled_field in unfilled_fields:
            field_name = unfilled_field.field_name
            
            # Find matching result from Gemini response
            matching_result = None
            for extracted in extracted_fields:
                if extracted.get("field_name") == field_name:
                    matching_result = extracted
                    break
            
            if matching_result:
                status = matching_result.get("status", "Information not present")
                reason = matching_result.get("reason")
                page = matching_result.get("page")
                section = matching_result.get("section")
                information_found = matching_result.get("information_found")
                
                # Update status to Satisfactory if information was found
                if information_found and status == "Information not present":
                    status = "Satisfactory"
                    print(f"[INFO] Updating status to Satisfactory for '{field_name}' as information was found")
                
                print(f"[SUCCESS] Gemini filled field '{field_name}' - Status: {status}, Page: {page}")
                
            else:
                # Keep original unfilled status
                status = "Information not present"
                reason = "Gemini could not find information"
                page = None
                section = None
                information_found = None
                print(f"[WARN] Gemini could not fill field '{field_name}'")
            
            results.append(FieldResult(
                field_id=unfilled_field.field_id,
                field_name=field_name,
                status=status,
                reason=reason,
                page=page if isinstance(page, int) else None,
                section=section if isinstance(section, str) else None,
                information_found=information_found
            ))
        
        print(f"\n[SUCCESS] Gemini image analysis complete for {len(results)} unfilled fields")
        return results
        
    except Exception as e:
        print(f"[ERROR] Gemini image analysis failed: {e}")
        print(f"[ERROR] Raw response: {raw_response[:500] if 'raw_response' in locals() else 'No response'}")
        print(f"[INFO] Returning original unfilled results")
        return unfilled_fields
    
    print(f"[INFO] Sending comprehensive prompt to Gemini API (single call for all fields)")
    print(f"[DEBUG] Comprehensive prompt length: {len(comprehensive_prompt)} characters")
    
    start_time = time.time()
    raw_response = call_gemini(comprehensive_prompt)
    end_time = time.time()
    
    print(f"[INFO] Gemini API response received in {end_time - start_time:.2f} seconds")
    print(f"[DEBUG] Raw response: {raw_response[:200]}...")
    
    # Parse the comprehensive response
    try:
        data = json.loads(raw_response)
        extracted_fields = data.get("fields", [])
        
        print(f"[SUCCESS] Parsed response with {len(extracted_fields)} field results")
        
        # Map the extracted results to our field structure
        results: List[FieldResult] = []
        for field in fields:
            field_name = field["name"]
            
            # Find matching result from LLM response
            matching_result = None
            for extracted in extracted_fields:
                if extracted.get("field_name") == field_name:
                    matching_result = extracted
                    break
            
            if matching_result:
                status = matching_result.get("status", "Information not present")
                reason = matching_result.get("reason")
                page = matching_result.get("page")
                section = matching_result.get("section")
                information_found = matching_result.get("information_found")
                
                # Update status to Satisfactory if information was found
                if information_found and status == "Information not present":
                    status = "Satisfactory"
                    print(f"[INFO] Updating status to Satisfactory for '{field_name}' as information was found")
                
                print(f"[SUCCESS] Field '{field_name}' - Status: {status}, Page: {page}, Section: {section}")
                print(f"[DEBUG] Reason: {reason}")
                print(f"[DEBUG] Information Found: {information_found}")
                
            else:
                # Fallback if field not found in response
                status = "Information not present"
                reason = "Field not analyzed in LLM response"
                page = None
                section = None
                information_found = None
                print(f"[WARN] Field '{field_name}' not found in LLM response, using fallback")
            
            results.append(FieldResult(
                field_id=field["id"],
                field_name=field_name,
                status=status,
                reason=reason,
                page=page if isinstance(page, int) else None,
                section=section if isinstance(section, str) else None,
                information_found=information_found
            ))
        
        print(f"\n[SUCCESS] Optimized LLM analysis complete. Processed {len(results)} fields in single API call")
        return results
        
    except Exception as e:
        print(f"[ERROR] JSON parsing failed for comprehensive response: {e}")
        print(f"[ERROR] Raw response: {raw_response[:1000]}")
        
        # Fallback to individual field processing if comprehensive approach fails
        print(f"[INFO] Falling back to individual field processing...")
        return analyze_fields_with_llm_fallback(pages, fields, all_text)

def analyze_fields_with_llm_fallback(pages: List[Dict[str, Any]], fields: List[Dict[str, Any]], all_text: str) -> List[FieldResult]:
    """Fallback method if OCR-first analysis fails - uses single comprehensive LLM call"""
    print(f"[INFO] Using fallback method with single comprehensive LLM call for {len(fields)} fields")
    
    # Build a comprehensive prompt for all fields
    field_names = [f["name"] for f in fields]
    
    # Determine text length based on document size
    text_length = len(all_text)
    if text_length > 1000000:  # More than 1MB of text (very large PDF)
        max_text = 500000  # Use 500K characters for very large PDFs
        print(f"[INFO] Large PDF detected ({text_length} chars), using {max_text} chars for fallback analysis")
    elif text_length > 500000:  # More than 500K characters (large PDF)
        max_text = 300000  # Use 300K characters for large PDFs
        print(f"[INFO] Large PDF detected ({text_length} chars), using {max_text} chars for fallback analysis")
    else:
        max_text = 200000  # Default for normal PDFs
        print(f"[INFO] Normal PDF size ({text_length} chars), using {max_text} chars for fallback analysis")
    
    fallback_prompt = f"""
You are an AI compliance reviewer. The OCR-first analysis failed, so please analyze the document text below for ALL the following compliance fields in a single response.

Required Fields to Analyze:
{chr(10).join([f"{i+1}. {name}" for i, name in enumerate(field_names)])}

Document Text (analyze thoroughly across all content):
---
{all_text[:max_text]}
---

IMPORTANT INSTRUCTIONS:
1. Search through ALL the provided text content thoroughly
2. Only fill page numbers when you actually find information for a field
3. If no information is found, set page to null and status to "Information not present"
4. For large documents, information might be spread across different sections
5. Be thorough in your search - don't miss information

Return ONLY valid JSON, no markdown, in the following schema:
{{
  "fields": [
    {{
      "field_name": "exact field name from the list above",
      "status": "Satisfactory" | "Unsatisfactory" | "Information not present",
      "reason": "short concise justification",
      "page": 12,
      "section": "heading/subheading or brief locator if any",
      "information_found": "actual text/information found for this field, or null if not found"
    }}
  ]
}}

Analyze each field thoroughly and provide accurate page numbers, sections, and extracted information.
"""
    
    print(f"[INFO] Sending fallback comprehensive prompt to Gemini API (single call for all fields)")
    
    try:
        raw_response = call_gemini(fallback_prompt)
        data = json.loads(raw_response)
        extracted_fields = data.get("fields", [])
        
        print(f"[SUCCESS] Parsed fallback response with {len(extracted_fields)} field results")
        
        # Map the extracted results to our field structure
        results: List[FieldResult] = []
        for field in fields:
            field_name = field["name"]
            
            # Find matching result from LLM response
            matching_result = None
            for extracted in extracted_fields:
                if extracted.get("field_name") == field_name:
                    matching_result = extracted
                    break
            
            if matching_result:
                status = matching_result.get("status", "Information not present")
                reason = matching_result.get("reason")
                page = matching_result.get("page")
                section = matching_result.get("section")
                information_found = matching_result.get("information_found")
                
                print(f"[SUCCESS] Fallback field '{field_name}' - Status: {status}, Page: {page}")
                
            else:
                # Final fallback if field still not found
                status = "Information not present"
                reason = "Field not analyzed in fallback response"
                page = None
                section = None
                information_found = None
                print(f"[WARN] Field '{field_name}' not found in fallback response")
            
            results.append(FieldResult(
                field_id=field["id"],
                field_name=field_name,
                status=status,
                reason=reason,
                page=page if isinstance(page, int) else None,
                section=section if isinstance(section, str) else None,
                information_found=information_found
            ))
        
        print(f"\n[SUCCESS] Fallback analysis complete. Processed {len(results)} fields in single API call")
        return results
        
    except Exception as e:
        print(f"[ERROR] Fallback JSON parsing failed: {e}")
        
        # Ultimate fallback - return basic results
        print(f"[INFO] Using ultimate fallback - basic results")
        results: List[FieldResult] = []
        for field in fields:
            results.append(FieldResult(
                field_id=field["id"],
                field_name=field["name"],
                status="Information not present",
                reason="All analysis methods failed",
                page=None,
                section=None,
                information_found=None
            ))
        
        return results

def enhance_missing_information_with_gemini(results: List[FieldResult], all_text: str, pages: List[Dict[str, Any]] = None) -> List[FieldResult]:
    """Enhance results with missing information using Gemini AI - prioritizes images when available"""
    print(f"[INFO] Enhancing {len(results)} results with missing information using Gemini AI")
    
    # Find fields that need enhancement
    fields_to_enhance = [r for r in results if r.status == "Information not present" and not r.information_found]
    
    if not fields_to_enhance:
        print(f"[INFO] No fields need enhancement")
        return results
    
    print(f"[INFO] Found {len(fields_to_enhance)} fields that need enhancement")
    
    # Check if we have images available for better analysis
    has_images = pages and any("image" in p for p in pages)
    
    if has_images:
        print(f"[INFO] Images available - using image-based enhancement for better accuracy")
        return enhance_with_gemini_images(fields_to_enhance, pages)
    else:
        print(f"[INFO] No images available - using text-based enhancement")
        return enhance_with_gemini_text(fields_to_enhance, all_text)

def enhance_with_gemini_images(fields_to_enhance: List[FieldResult], pages: List[Dict[str, Any]]) -> List[FieldResult]:
    """Enhance fields using Gemini with images for better visual analysis"""
    print(f"[INFO] Using Gemini with images to enhance {len(fields_to_enhance)} fields")
    
    # Extract images from pages
    images = [p["image"] for p in pages if "image" in p]
    print(f"[INFO] Extracted {len(images)} images for Gemini analysis")
    
    # Build comprehensive enhancement prompt for all missing fields
    field_names = [r.field_name for r in fields_to_enhance]
    
    enhancement_prompt = f"""
You are an AI compliance reviewer analyzing scanned document images. The following fields were marked as "Information not present" but we need to double-check.
Please analyze the images visually to look for ANY information related to these fields, even if it's incomplete or unclear.

Fields to Re-analyze:
{chr(10).join([f"{i+1}. {name}" for i, name in enumerate(field_names)])}

IMPORTANT INSTRUCTIONS:
1. Analyze ALL the provided images thoroughly and visually
2. Look for visual information related to each field (text, tables, forms, signatures, etc.)
3. Only fill page numbers when you actually find information for a field
4. If no information is found, set page to null
5. For multi-page documents, information might be spread across different pages
6. Be thorough in your visual analysis - don't miss information
7. Consider both text content and visual layout/structure

Return ONLY valid JSON, no markdown, in the following schema:
{{
  "fields": [
    {{
      "field_name": "exact field name from the list above",
      "information_found": "any text/information found related to this field, or null if absolutely nothing found",
      "page": 12,
      "section": "heading/subheading or brief locator if any"
    }}
  ]
}}

Be thorough and extract any relevant information, even if it's partial or unclear. If you find information, update the page and section accordingly.
"""
    
    print(f"[INFO] Sending comprehensive image-based enhancement prompt to Gemini API")
    
    try:
        raw_response = call_gemini_with_images(enhancement_prompt, images)
        
        # Clean the response to fix JSON parsing issues
        cleaned_response = clean_gemini_response(raw_response)
        
        if not cleaned_response.strip():
            print(f"[WARN] Empty response from Gemini, skipping enhancement")
            return []
        
        data = json.loads(cleaned_response)
        enhanced_fields = data.get("fields", [])
        
        print(f"[SUCCESS] Parsed image-based enhancement response with {len(enhanced_fields)} field results")
        
        # Update results with enhanced information
        enhanced_results = []
        for field in fields_to_enhance:
            # Find matching enhancement
            matching_enhancement = None
            for enhanced in enhanced_fields:
                if enhanced.get("field_name") == field.field_name:
                    matching_enhancement = enhanced
                    break
            
            if matching_enhancement and matching_enhancement.get("information_found"):
                field.information_found = matching_enhancement.get("information_found")
                field.page = matching_enhancement.get("page") if isinstance(matching_enhancement.get("page"), int) else field.page
                field.section = matching_enhancement.get("section") if isinstance(matching_enhancement.get("section"), str) else field.section
                print(f"[SUCCESS] Enhanced field '{field.field_name}' with information: {field.information_found[:100]}...")
            else:
                print(f"[INFO] No additional information found for field '{field.field_name}'")
            
            enhanced_results.append(field)
        
        print(f"[SUCCESS] Enhanced {len(enhanced_results)} results with image-based Gemini AI call")
        return enhanced_results
        
    except Exception as e:
        print(f"[ERROR] Image-based enhancement failed: {e}")
        print(f"[ERROR] Raw response: {raw_response[:500] if 'raw_response' in locals() else 'No response'}")
        print(f"[INFO] Returning original results without enhancement")
        return fields_to_enhance

def enhance_with_gemini_text(fields_to_enhance: List[FieldResult], all_text: str) -> List[FieldResult]:
    """Enhance fields using Gemini with text for fallback analysis"""
    print(f"[INFO] Using Gemini with text to enhance {len(fields_to_enhance)} fields")
    
    # Build comprehensive enhancement prompt for all missing fields
    field_names = [r.field_name for r in fields_to_enhance]
    
    # Determine text length based on document size
    text_length = len(all_text)
    if text_length > 1000000:  # More than 1MB of text (very large PDF)
        max_text = 500000  # Use 500K characters for very large PDFs
        print(f"[INFO] Large PDF detected ({text_length} chars), using {max_text} chars for enhancement")
    elif text_length > 500000:  # More than 500K characters (large PDF)
        max_text = 300000  # Use 300K characters for large PDFs
        print(f"[INFO] Large PDF detected ({text_length} chars), using {max_text} chars for enhancement")
    else:
        max_text = 150000  # Default for normal PDFs
        print(f"[INFO] Normal PDF size ({text_length} chars), using {max_text} chars for enhancement")
    
    enhancement_prompt = f"""
You are an AI compliance reviewer. The following fields were marked as "Information not present" but we need to double-check.
Please look for ANY information related to these fields in the document text below, even if it's incomplete or unclear.

Fields to Re-analyze:
{chr(10).join([f"{i+1}. {name}" for i, name in enumerate(field_names)])}

Document Text (analyze thoroughly across all content):
---
{all_text[:max_text]}
---

IMPORTANT INSTRUCTIONS:
1. Search through ALL the provided text content thoroughly
2. Only fill page numbers when you actually find information for a field
3. If no information is found, set page to null
4. For large documents, information might be spread across different sections
5. Be thorough in your search - don't miss information

Return ONLY valid JSON, no markdown, in the following schema:
{{
  "fields": [
    {{
      "field_name": "exact field name from the list above",
      "information_found": "any text/information found related to this field, or null if absolutely nothing found",
      "page": 12,
      "section": "heading/subheading or brief locator if any"
    }}
  ]
}}

Be thorough and extract any relevant information, even if it's partial or unclear. If you find information, update the page and section accordingly.
"""
    
    print(f"[INFO] Sending comprehensive text-based enhancement prompt to Gemini API")
    
    try:
        raw_response = call_gemini(enhancement_prompt)
        
        # Clean the response to fix JSON parsing issues
        cleaned_response = clean_gemini_response(raw_response)
        
        if not cleaned_response.strip():
            print(f"[WARN] Empty response from Gemini, skipping enhancement")
            return []
        
        data = json.loads(cleaned_response)
        enhanced_fields = data.get("fields", [])
        
        print(f"[SUCCESS] Parsed text-based enhancement response with {len(enhanced_fields)} field results")
        
        # Update results with enhanced information
        enhanced_results = []
        for field in fields_to_enhance:
            # Find matching enhancement
            matching_enhancement = None
            for enhanced in enhanced_fields:
                if enhanced.get("field_name") == field.field_name:
                    matching_enhancement = enhanced
                    break
            
            if matching_enhancement and matching_enhancement.get("information_found"):
                field.information_found = matching_enhancement.get("information_found")
                field.page = matching_enhancement.get("page") if isinstance(matching_enhancement.get("page"), int) else field.page
                field.section = matching_enhancement.get("section") if isinstance(matching_enhancement.get("section"), str) else field.section
                print(f"[SUCCESS] Enhanced field '{field.field_name}' with information: {field.information_found[:100]}...")
            else:
                print(f"[INFO] No additional information found for field '{field.field_name}'")
            
            enhanced_results.append(field)
        
        print(f"[SUCCESS] Enhanced {len(enhanced_results)} results with text-based Gemini AI call")
        return enhanced_results
        
    except Exception as e:
        print(f"[ERROR] Text-based enhancement failed: {e}")
        print(f"[ERROR] Raw response: {raw_response[:500] if 'raw_response' in locals() else 'No response'}")
        print(f"[INFO] Returning original results without enhancement")
        return fields_to_enhance

# ------------- Routes -------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    print("[INFO] Root endpoint accessed")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/config")
async def get_config():
    print("[INFO] Config endpoint accessed")
    fields = load_fields()
    return {"fields": fields}

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(pdf: UploadFile = File(...), use_ocr: bool = Form(False)):
    print(f"\n[INFO] ===== Starting PDF Analysis =====")
    print(f"[INFO] File received: {pdf.filename}")
    print(f"[INFO] File size: {pdf.size} bytes")
    print(f"[INFO] Content type: {pdf.content_type}")
    print(f"[INFO] OCR enabled: {use_ocr}")
    
    start_time = time.time()
    
    try:
        # Read file
        print("[INFO] Reading uploaded file...")
        file_bytes = await pdf.read()
        print(f"[SUCCESS] File read successfully. Bytes: {len(file_bytes)}")
        
        # Extract PDF pages (text + images)
        print("[INFO] Extracting PDF pages with images...")
        pages = extract_pdf_pages(file_bytes, use_ocr=use_ocr)
        print(f"[SUCCESS] PDF extraction complete. Pages: {len(pages)}")
        
        # Load fields
        print("[INFO] Loading compliance fields...")
        fields = load_fields()
        print(f"[SUCCESS] Fields loaded: {len(fields)}")
        
        # Analyze with image-based approach
        print("[INFO] Starting image-based analysis...")
        results = analyze_fields_with_images_and_llm(pages, fields, use_ocr=use_ocr)
        print(f"[SUCCESS] Image-based analysis complete. Results: {len(results)}")
        
        # Only enhance if there are still missing fields
        missing_fields = [r for r in results if r.status == "Information not present" and not r.information_found]
        if missing_fields:
            print(f"[INFO] Enhancing {len(missing_fields)} missing fields with Gemini AI fallback...")
            enhanced_results = enhance_missing_information_with_gemini(results, "".join([p["text"] for p in pages]), pages)
            print(f"[SUCCESS] Enhancement complete. Enhanced results: {len(enhanced_results)}")
        else:
            print(f"[INFO] No missing fields to enhance")
            enhanced_results = results
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"[SUCCESS] ===== Analysis Complete =====")
        print(f"[INFO] Total processing time: {total_time:.2f} seconds")
        print(f"[INFO] Average time per field: {total_time/len(fields):.2f} seconds")
        
        return AnalyzeResponse(results=enhanced_results)
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        raise

@app.post("/api/export")
async def export_results(payload: Dict[str, Any]):
    print("[INFO] Export endpoint accessed")
    os.makedirs("exports", exist_ok=True)
    path = os.path.join("exports", "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Results exported to {path}")
    return {"ok": True, "download": "/download/results.json"}

@app.get("/download/results.json")
async def download_results():
    print("[INFO] Download endpoint accessed")
    path = os.path.join("exports", "results.json")
    if not os.path.exists(path):
        print("[ERROR] Export file not found")
        return JSONResponse({"error": "No export found yet"}, status_code=404)
    print(f"[SUCCESS] Serving file: {path}")
    return FileResponse(path, filename="results.json")

if __name__ == "__main__":
    import uvicorn
    print("[INFO] PDF Compliance Form App starting...")
    print("[INFO] Available endpoints:")
    print("  - GET  / - Main form interface")
    print("  - GET  /api/config - Get compliance fields")
    print("  - POST /api/analyze - Analyze PDF for compliance")
    print("  - POST /api/export - Export results")
    print("  - GET  /download/results.json - Download results")
    
    # Run with uvicorn to avoid logging conflicts
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="info"
    )
