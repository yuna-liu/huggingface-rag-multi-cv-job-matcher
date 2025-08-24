# File: app.py
import gradio as gr
from PyPDF2 import PdfReader
from transformers import pipeline

# Load summarization model (free HF model)
summarizer = pipeline("summarization", model="google/flan-t5-large")  # smaller free model

def parse_pdf(pdf_files):
    texts = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file.name)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        texts.append((pdf_file.name, text))
    return texts

def summarize_cvs(cv_files):
    parsed_cvs = parse_pdf(cv_files)
    results = []

    for filename, text in parsed_cvs:
        # Limit text length for model
        snippet = text[:4000]  # cut long CVs
        summary = summarizer(snippet, max_length=150, min_length=60, do_sample=False)[0]['summary_text']
        results.append({
            "CV Filename": filename,
            "Summary": summary
        })

    return results

with gr.Blocks() as demo:
    gr.Markdown("## 📝 CV Summarizer (Free Model Version)")
    cv_input = gr.Files(label="Upload CV PDFs (.pdf)", file_types=[".pdf"])
    output = gr.Dataframe(headers=["CV Filename", "Summary"])
    run_button = gr.Button("Summarize CVs")
    
    run_button.click(fn=summarize_cvs, inputs=[cv_input], outputs=[output])

demo.launch()
