# app.py

"""
AI DPR Generator — Government-focused (English + Telugu)
Features:
- Streamlit front-end for DPR inputs
- LLM-based DPR generation (OpenAI API)
- Clean markdown -> HTML conversion for PDF
- PDF export using pdfkit/wkhtmltopdf with professional styling
- Simple ML financial forecast (Linear Regression fallback)
- Optional translation using Hugging Face transformers (if installed)
- File upload handling (PDF/Excel parsing placeholders)
- User consent & data privacy notice
- Local sqlite logging for submissions (SQLAlchemy)
- Placeholders for AP MSME ONE Portal integration / RAG / feedback
"""

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from datetime import date
import requests
import tempfile
from jinja2 import Template
import pdfkit
import re
import json
import pandas as pd
import pdfplumber
from sklearn.linear_model import LinearRegression
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Table, MetaData
from sqlalchemy.sql import func
import logging

# Optional translation (if transformers installed)
try:
    from transformers import pipeline
    TRANSLATION_AVAILABLE = True
except Exception:
    TRANSLATION_AVAILABLE = False

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AP_MSME_API_URL = os.getenv("AP_MSME_API_URL", "")
WKHTMLTOPDF_PATH = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"  # update if needed

# ================================================================
#  NAVIGATION / INTRODUCTORY SECTION
# ================================================================
st.sidebar.title(" Navigation")
page = st.sidebar.radio("Go to", [" About", " DPR Generator"])

# ================================================================
#  ABOUT PAGE (Enhanced with Multi-Model Pipeline)
# ================================================================
if page == " About":
    st.title("AI-Based Tool for Detailed Project Report (DPR) Preparation")
    st.markdown("""
    ### 🌍 Problem Statement
    Many **MSME entrepreneurs**, especially first-time founders and rural innovators, struggle to prepare  
    **Detailed Project Reports (DPRs)** required for bank loans, subsidies, and investments.

    **Key Challenges:**
    -  Time-consuming manual preparation  
    -  Costly professional consultants  
    -  Poor quality and non-standard DPRs  
    -  Lack of bilingual (English–Telugu) support  

    ###  Proposed AI Solution
    This application uses a **multi-model AI pipeline** to automate, simplify, and standardize the DPR creation process.

    ---
    ###  Core Features
    -  **LLM-Based Content Generation:** Creates complete, structured DPRs in English or Telugu  
    -  **Financial Projection Engine (Planned):** Uses regression/time-series ML models to predict future cash flow  
    -  **Machine Translation:** Converts DPRs between English ↔ Telugu for inclusivity  
    -  **Document Analysis:** (Future) Extracts data from uploaded PDFs/Excel for auto-filling cost & revenue tables  
    -  **Feedback Learning:** Continuously improves suggestions based on user ratings & success metrics  

    ---
    ###  Expected Outcomes
    -  90% reduction in DPR preparation time  
    -  20–30% increase in MSME loan approval rate  
    -  Dual-language accessibility (English & Telugu)  
    -  AI-driven ecosystem for credit access & policy analytics  

    ---
    """)

    # --- MULTI-MODEL PIPELINE DIAGRAM ---
    st.subheader(" Multi-Model AI Pipeline Architecture")
    st.markdown("""
    ```mermaid
    graph TD
        A[User Inputs / Uploads] --> B[Streamlit Frontend]
        B --> C[LLM: GPT-4o-mini - DPR Content Generator]
        C --> D[Financial Forecasting Engine - Prophet/XGBoost]
        C --> E[Translation Engine - M2M100 / OpusMT (En↔Te)]
        C --> F[Summarization & NLP Refinement - T5/Flan-T5]
        F --> G[Formatted DPR Report Builder (PDF Generator)]
        D --> G
        E --> G
        G --> H[Downloadable DPR PDF]
        H --> I[Feedback & Learning (BERT/Clustering)]
    ```
    """)

    st.info("""
     **Planned Model Integrations:**
    | Module | Example Models | Function |
    |---------|----------------|-----------|
    | Financial Forecasting | Prophet, Linear Regression, XGBoost | Predict 3-year revenue & repayment capacity |
    | Text Summarization | T5, Flan-T5 | Refine user text into formal bank language |
    | Translation | facebook/m2m100_418M, Helsinki-NLP/opus-mt-en-te | Bilingual support (English ↔ Telugu) |
    | Feedback Learning | Sentence-BERT, KMeans | Learn from user feedback & improve suggestions |
    | Document AI | LayoutLMv3, Donut | Extract data from uploaded financial PDFs/Excel |
    """)

    st.markdown("""
    ---
    ###  Scalability & Integration Plan
    - **Phase 1 (MVP):** LLM + PDF generation (completed )  
    - **Phase 2:** Integrate forecasting & translation APIs  
    - **Phase 3:** Add document intelligence & feedback loops  
    - **Phase 4:** Full integration with **AP MSME ONE Portal**  

    ---
    **Developed For:** Andhra Pradesh MSME Hackathon  
    **Developed By:** [Team Name:Dr.K.Murali, G.Vijayalakshmi, N.Ganesh Reddy]  
    **Role:** AI & Data Science Developer  
    ---
    """)

    st.stop()  # Stop execution on this page to avoid showing the form


# configure pdfkit
config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-dpr")

# Simple local DB (sqlite) to log submissions and consent
DB_PATH = "ai_dpr_submissions.db"
engine = create_engine(f"sqlite:///{DB_PATH}")
meta = MetaData()
submissions_tbl = Table(
    "submissions", meta,
    Column("id", Integer, primary_key=True),
    Column("project_name", String),
    Column("entrepreneur", String),
    Column("sector", String),
    Column("location", String),
    Column("created_at", DateTime, server_default=func.now()),
    Column("status", String, default="draft"),
    Column("dpr_text", Text),
)
meta.create_all(engine)

# ---------- Utilities ----------
def save_submission_log(record):
    with engine.connect() as conn:
        conn.execute(submissions_tbl.insert().values(**record))

def call_openai_chat(prompt: str, model="gpt-4o-mini", max_tokens=2500):
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not set.")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        logger.error("OpenAI error: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"LLM API Error: {resp.status_code} {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]

def clean_markdown_to_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"^### (.*)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.*)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.*)$", r"<h1>\1</h1>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    text = re.sub(r"---+", r"<hr/>", text)
    text = re.sub(r"(?m)^\s*-\s+(.*)$", r"<li>\1</li>", text)
    table_blocks = re.findall(r"((?:\|.*?\|\s*\n)+)", text)
    for tbl in table_blocks:
        rows = [r.strip() for r in tbl.strip().split("\n") if "|" in r]
        html_table = "<table><tbody>"
        for i, r in enumerate(rows):
            cols = [c.strip() for c in r.split("|")[1:-1]]
            tag = "th" if i == 0 else "td"
            html_table += "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cols) + "</tr>"
        html_table += "</tbody></table>"
        text = text.replace(tbl, html_table)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    html = "<br><br>".join(paragraphs)
    return html

def generate_pdf_from_html(project_name: str, html: str, output_filename: str = None):
    if not output_filename:
        safe_name = re.sub(r"[^\w\s-]", "", project_name).strip().replace(" ", "_")
        output_filename = f"{safe_name}_DPR.pdf"
    options = {"quiet": "", "margin-top": "15mm", "margin-bottom": "15mm", "margin-left": "18mm", "margin-right": "18mm"}
    pdfkit.from_string(html, output_filename, configuration=config, options=options)
    return output_filename

def forecast_revenue(first_year_value: float):
    if first_year_value <= 0:
        base = 100000
    else:
        base = float(first_year_value)
    X = np.array([[0],[1]])
    y = np.array([base*0.85, base])
    model = LinearRegression().fit(X, y)
    years = np.array([[2],[3],[4]])
    preds = model.predict(years) * 1.12
    start_year = date.today().year + 1
    forecast = {}
    for i, p in enumerate(preds):
        y_label = start_year + i
        forecast[str(y_label)] = float(round(p, 2))
    return forecast

def extract_text_from_pdf_bytes(file_bytes):
    try:
        with pdfplumber.open(file_bytes) as pdf:
            all_text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        return all_text
    except Exception as e:
        logger.warning("pdfplumber extraction failed: %s", e)
        return ""

translator = None
if TRANSLATION_AVAILABLE:
    try:
        translator = pipeline("translation_en_to_xx", model="Helsinki-NLP/opus-mt-en-te")
    except Exception as e:
        translator = None
        logger.info("Translation pipeline not available locally: %s", e)

def translate_to_telugu(text: str):
    if translator:
        try:
            out = translator(text, max_length=2000)
            return out[0]["translation_text"]
        except Exception:
            return text
    return text

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI DPR Generator", page_icon="📄", layout="wide")
st.title("📄 AI DPR Generator — AP MSME Focused")

st.sidebar.markdown("## Project Info\nGovernment project: DPR Automation for AP MSME")
st.markdown("### Instructions")
st.markdown(
    "- Fill project details and a clear brief. Upload supporting documents (balance sheet, estimates).\n"
    "- Check the consent box (we log consent and keep data local)."
)

# Consent checkbox
consent = st.checkbox(
    "I consent to upload these details to be processed for DPR generation. "
    "I confirm data is accurate and understand this information will be processed locally."
)
st.session_state["consent_given"] = consent

if not consent:
    st.info("⚠️ Please check the consent box above to enable DPR generation.")

# DPR Input Form — only one form
with st.form("dpr_input_form_main"):
    project_name = st.text_input("Project Name", disabled=not consent)
    entrepreneur_name = st.text_input("Entrepreneur Name", disabled=not consent)
    sector = st.selectbox("Sector", ["Manufacturing", "Services", "Agriculture", "Other"], disabled=not consent)
    location = st.text_input("Location", disabled=not consent)
    project_cost = st.number_input("Total Project Cost (INR)", min_value=0.0, step=1000.0, disabled=not consent)
    equity = st.number_input("Equity Contribution (INR)", min_value=0.0, step=1000.0, disabled=not consent)
    loan_required = st.number_input("Loan Required (INR)", min_value=0.0, step=1000.0, disabled=not consent)
    expected_annual_revenue = st.number_input("Expected Annual Revenue (INR)", min_value=0.0, step=1000.0, disabled=not consent)
    duration_months = st.number_input("Project Duration (Months)", min_value=1, step=1, disabled=not consent)
    start_date = st.date_input("Project Start Date", date.today(), disabled=not consent)
    brief_description = st.text_area("Brief Project Description", disabled=not consent)
    uploaded_files = st.file_uploader("Upload Supporting Documents (PDF/Excel)", type=["pdf", "xlsx", "xls"], accept_multiple_files=True, disabled=not consent)
    language = st.radio("Generate DPR in:", ["English", "Telugu"], index=0, disabled=not consent)

    submit_btn = st.form_submit_button("Generate DPR", disabled=not consent)

# ---------- DPR Processing ----------
if submit_btn:
    inputs = {
        "project_name": project_name.strip() or "Untitled Project",
        "entrepreneur_name": entrepreneur_name.strip() or "N/A",
        "sector": sector,
        "location": location.strip() or "N/A",
        "project_cost": float(project_cost or 0.0),
        "equity": float(equity or 0.0),
        "loan_required": float(loan_required or 0.0),
        "start_date": str(start_date),
        "duration_months": int(duration_months),
        "expected_annual_revenue": float(expected_annual_revenue or 0.0),
        "brief_description": brief_description.strip() or "No description provided.",
    }

    uploaded_texts = []
    if uploaded_files:
        for up in uploaded_files:
            try:
                txt = extract_text_from_pdf_bytes(up)
                if txt:
                    uploaded_texts.append(txt)
            except Exception:
                try:
                    df = pd.read_excel(up)
                    uploaded_texts.append(df.to_csv(index=False))
                except Exception:
                    logger.info("Unsupported file type or extraction failed for %s", up.name)

    prompt = f"""
You are an expert DPR writer for Indian MSME bank loan proposals. Produce a professional Detailed Project Report (DPR) in clear formal English.
Do NOT use raw Markdown symbols like '#', '##', '***', etc. Output plain text paragraphs and tables using clear labels.
Project inputs:
{json.dumps(inputs, indent=2)}
Generate: Executive Summary, Business Overview, Market Potential, Production Plan, Financial Estimates, Funding Pattern, SWOT Analysis, Conclusion, Annexures.
"""
    try:
        dpr_text = call_openai_chat(prompt)
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        st.stop()

    st.success("Draft DPR generated by LLM.")
    st.subheader("Preview (editable)")
    st.text_area("DPR Draft (you can edit before export)", value=dpr_text, height=400, key="dpr_edit")

    dpr_text_final = st.session_state.get("dpr_edit", dpr_text)

    # Forecast
    forecast = forecast_revenue(inputs["expected_annual_revenue"])
    st.subheader("AI Financial Forecast (3-year)")
    st.json(forecast)

    # Translation if Telugu
    if language == "Telugu":
        st.info("Translating DPR to Telugu...")
        dpr_text_final = translate_to_telugu(dpr_text_final)

    # HTML -> PDF
    formatted_html_body = clean_markdown_to_html(dpr_text_final)

    html_template = """<!doctype html>
<html><head><meta charset="utf-8"/>
<style>
body { font-family: "Times New Roman", serif; margin: 40px; line-height: 1.75; color: #111; font-size: 12.5pt; }
.cover { text-align: center; margin-top: 180px; }
.cover h1 { color: #0a3d62; font-size: 28pt; margin-bottom: 6px; }
.cover h2 { font-size: 18pt; margin-top: 6px; }
.section-title { color: #0a3d62; font-size: 16pt; margin-top: 28px; border-bottom: 1px solid #aaa; padding-bottom: 6px; }
p { margin-bottom: 12px; text-align: justify; }
table { width:100%; border-collapse: collapse; margin: 12px 0; }
th, td { border:1px solid #444; padding:8px 10px; vertical-align: top; }
th { background:#0a3d62; color:#fff; }
.footer { margin-top:40px; font-size:11pt; text-align:right; color:#555; }
.signature { margin-top:60px; text-align:right; font-style: italic; }
ul { margin-bottom:12px; margin-left:20px; }
hr { border:none; border-top:1px solid #ddd; margin:18px 0; }
</style></head><body>
<div class="cover">
  <h1>Detailed Project Report (DPR)</h1>
  <h2>{{ project_name }}</h2>
  <p><b>Promoter:</b> {{ entrepreneur_name }} &nbsp; | &nbsp; <b>Sector:</b> {{ sector }} &nbsp; | &nbsp; <b>Location:</b> {{ location }}</p>
  <p><b>Total Project Cost:</b> ₹{{ "{:,.2f}".format(project_cost) }}</p>
  <p><b>Generated on:</b> {{ today }}</p>
</div>
<hr/>
<div class="section">
  <div class="section-title">Executive Summary</div>
  <p>{{ brief_description }}</p>
</div>
<div class="section">
  <div class="section-title">Full Report</div>
  <div>{{ report_body | safe }}</div>
</div>
<div class="section">
  <div class="section-title">AI Financial Forecast (3-year)</div>
  {{ forecast_table | safe }}
</div>
<div class="signature">
  <p>Authorized Signatory,<br><b>{{ entrepreneur_name }}</b></p>
</div>
<div class="footer">Generated by AI-DPR Generator • Date: {{ today }}</div>
</body></html>
"""

    forecast_rows = "".join(f"<tr><td>{k}</td><td>₹{v:,.2f}</td></tr>" for k, v in forecast.items())
    forecast_table = f"<table><tr><th>Year</th><th>Projected Revenue (INR)</th></tr>{forecast_rows}</table>"

    html = Template(html_template).render(
        project_name=inputs["project_name"],
        entrepreneur_name=inputs["entrepreneur_name"],
        sector=inputs["sector"],
        location=inputs["location"],
        project_cost=inputs["project_cost"],
        brief_description=inputs["brief_description"],
        report_body=formatted_html_body,
        forecast_table=forecast_table,
        today=date.today().strftime("%d %B %Y")
    )

    tmp_dir = tempfile.gettempdir()
    safe_fname = re.sub(r"[^\w\s-]", "", inputs["project_name"]).strip().replace(" ", "_")
    pdf_path = os.path.join(tmp_dir, f"{safe_fname}_DPR.pdf")
    pdfkit.from_string(html, pdf_path, configuration=config, options={"quiet": ""})

    # Save log
    try:
        save_submission_log({
            "project_name": inputs["project_name"],
            "entrepreneur": inputs["entrepreneur_name"],
            "sector": inputs["sector"],
            "location": inputs["location"],
            "dpr_text": dpr_text_final
        })
    except Exception as e:
        logger.warning("Could not log submission: %s", e)

    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Final DPR (PDF)", f, file_name=f"{safe_fname}_DPR.pdf", mime="application/pdf")

    st.success("DPR generated successfully!")
