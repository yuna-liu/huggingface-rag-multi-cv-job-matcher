# File: app.py
import gradio as gr
import pdfplumber
import re
import os
import tempfile

# 临时目录存放上传文件
TEMP_DIR = os.path.join(tempfile.gettempdir(), "TempFile")
os.makedirs(TEMP_DIR, exist_ok=True)

def save_temp_files(pdf_files):
    saved_paths = []
    for pdf_file in pdf_files:
        temp_path = os.path.join(TEMP_DIR, pdf_file.name)
        with open(temp_path, "wb") as f:
            f.write(pdf_file.read())
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
    pdf_paths = save_temp_files(cv_files)
    parsed = parse_pdf(pdf_paths)
    display_text = ""
    for filename, text in parsed:
        display_text += f"===== {filename} =====\n{text}\n\n"
    return display_text or "No text found in uploaded PDFs."

def keyword_match(cv_text, job_text):
    cv_words = set(re.findall(r'\b\w+\b', cv_text.lower()))
    job_words = set(re.findall(r'\b\w+\b', job_text.lower()))
    matched = sorted(cv_words & job_words)
    missing = sorted(job_words - cv_words)
    score = round(len(matched) / max(1, len(job_words)) * 100, 2)
    return matched, missing, score

def match_cvs_to_job(cv_files, job_description):
    pdf_paths = save_temp_files(cv_files)
    parsed_cvs = parse_pdf(pdf_paths)
    
    results = []
    for filename, text in parsed_cvs:
        matched, missing, score = keyword_match(text, job_description)
        results.append({
            "CV Filename": filename,
            "Matched Skills": ", ".join(matched[:15]),
            "Missing Skills": ", ".join(missing[:15]),
            "Match Score": score
        })
    return results

with gr.Blocks() as demo:
    gr.Markdown("## ⚡ Instant CV Matcher with TempFile + PDF Debug View")
    
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste Job Description here...", label="Job Description")
    
    output_table = gr.Dataframe(headers=["CV Filename", "Matched Skills", "Missing Skills", "Match Score"])
    output_text = gr.Textbox(lines=20, label="PDF Text Debug Output")
    
    analyze_button = gr.Button("Analyze CVs")
    debug_button = gr.Button("Show PDF Texts")
    
    analyze_button.click(fn=match_cvs_to_job, inputs=[cv_input, job_input], outputs=[output_table])
    debug_button.click(fn=show_pdf_text, inputs=[cv_input], outputs=[output_text])

demo.launch()
