# File: app.py
import gradio as gr
import pdfplumber
import re
import os
import tempfile
import shutil
from transformers import pipeline

# Create temp dir for uploaded PDFs
TEMP_DIR = os.path.join(tempfile.gettempdir(), "TempFile")
os.makedirs(TEMP_DIR, exist_ok=True)

# Load lightweight summarizer (free HuggingFace model)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def save_temp_files(pdf_files):
    saved_paths = []
    for pdf_file in pdf_files:
        temp_path = os.path.join(TEMP_DIR, os.path.basename(pdf_file.name))
        shutil.copy(pdf_file.name, temp_path)
        saved_paths.append(temp_path)
    return saved_paths

def parse_pdf(pdf_paths):
    all_texts = []
    for path in pdf_paths:
        with pdfplumber.open(path) as pdf:
            text_pages = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_pages.append(f"--- Page {i+1} ---\n{page_text}")
            text = "\n".join(text_pages)
            all_texts.append((os.path.basename(path), text))
    return all_texts

def show_pdf_text(cv_files):
    if not cv_files:
        return "No files uploaded"
    pdf_paths = save_temp_files(cv_files)
    parsed = parse_pdf(pdf_paths)
    display_text = ""
    for filename, text in parsed:
        display_text += f"===== {filename} =====\n{text}\n\n"
    return display_text or "No text found in uploaded PDFs."

def keyword_match(cv_text, job_text):
    # Combined stopwords: Swedish + English (simple list)
    stopwords = {
        # Swedish
        "att", "av", "du", "din", "och", "för", "med", "i", "på", "den", "det", "en", "ett", "som",
        # English
        "the", "and", "for", "with", "from", "that", "this", "a", "an", "to", "in", "of", "on"
    }
    
    # Extract words, lowercase, remove stopwords and short words
    cv_words = set(w for w in re.findall(r'\b\w+\b', cv_text.lower()) if w not in stopwords and len(w) > 2)
    job_words = set(w for w in re.findall(r'\b\w+\b', job_text.lower()) if w not in stopwords and len(w) > 2)
    
    matched = sorted(cv_words & job_words)
    missing = sorted(job_words - cv_words)
    score = round(len(matched) / max(1, len(job_words)) * 100, 2)
    
    return matched, missing, score


def match_cvs_to_job(cv_files, job_description):
    if not cv_files:
        return [["No file", "", "", 0]]
    if not job_description.strip():
        return [["No job description", "", "", 0]]
    
    pdf_paths = save_temp_files(cv_files)
    parsed_cvs = parse_pdf(pdf_paths)
    
    results = []
    for filename, text in parsed_cvs:
        matched, missing, score = keyword_match(text, job_description)
        results.append([
            filename,
            ", ".join(matched[:15]) or "(none)",
            ", ".join(missing[:15]) or "(none)",
            score
        ])
    return results

def summarize_cv(cv_files):
    if not cv_files:
        return "No files uploaded"
    pdf_paths = save_temp_files(cv_files)
    parsed = parse_pdf(pdf_paths)
    summaries = ""
    for filename, text in parsed:
        if not text.strip():
            summaries += f"===== {filename} =====\n(No text)\n\n"
            continue
        short_text = text[:2000]  # limit length for speed
        summary = summarizer(short_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        summaries += f"===== {filename} =====\n{summary}\n\n"
    return summaries

with gr.Blocks() as demo:
    gr.Markdown("## ⚡ Instant CV Matcher + PDF Debug View + Quick Summary")
    
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste Job Description here...", label="Job Description")
    
    output_table = gr.Dataframe(
        headers=["CV Filename", "Matched Skills", "Missing Skills", "Match Score"],
        type="array"
    )
    output_text = gr.Textbox(lines=20, label="PDF Text Debug Output")
    output_summary = gr.Textbox(lines=10, label="Quick CV Summaries")
    
    analyze_button = gr.Button("Analyze CVs")
    debug_button = gr.Button("Show PDF Texts")
    summary_button = gr.Button("Summarize CVs")
    
    analyze_button.click(fn=match_cvs_to_job, inputs=[cv_input, job_input], outputs=[output_table])
    debug_button.click(fn=show_pdf_text, inputs=[cv_input], outputs=[output_text])
    summary_button.click(fn=summarize_cv, inputs=[cv_input], outputs=[output_summary])

demo.launch()
