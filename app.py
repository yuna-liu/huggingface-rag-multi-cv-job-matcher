# File: app.py
import gradio as gr
import pdfplumber
import re
import os

def parse_pdf(pdf_files):
    all_texts = []
    for pdf_file in pdf_files:
        file_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(
                [page.extract_text() or "" for page in pdf.pages]
            ).strip()
            print(f"Extracted from {file_path[:50]}: {text[:200]}...")  # debug
            all_texts.append((os.path.basename(file_path), text))
    return all_texts

def keyword_match(cv_text, job_text):
    cv_words = set(re.findall(r'\b\w+\b', cv_text.lower()))
    job_words = set(re.findall(r'\b\w+\b', job_text.lower()))
    matched = sorted(cv_words & job_words)
    missing = sorted(job_words - cv_words)
    score = round(len(matched) / max(1, len(job_words)) * 100, 2)
    return matched, missing, score

def match_cvs_to_job(cv_files, job_description):
    parsed_cvs = parse_pdf(cv_files)
    results = []
    for filename, text in parsed_cvs:
        if not text.strip():
            results.append({
                "CV Filename": filename,
                "Matched Skills": "No text found",
                "Missing Skills": "",
                "Match Score": 0
            })
            continue
        matched, missing, score = keyword_match(text, job_description)
        results.append({
            "CV Filename": filename,
            "Matched Skills": ", ".join(matched[:15]),
            "Missing Skills": ", ".join(missing[:15]),
            "Match Score": score
        })
    return results

with gr.Blocks() as demo:
    gr.Markdown("## ⚡ Instant CV Matcher (No Big Model Needed)")
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste Job Description here...", label="Job Description")
    output = gr.Dataframe(headers=["CV Filename", "Matched Skills", "Missing Skills", "Match Score"])
    run_button = gr.Button("Analyze CVs")

    run_button.click(fn=match_cvs_to_job, inputs=[cv_input, job_input], outputs=[output])

demo.launch()
