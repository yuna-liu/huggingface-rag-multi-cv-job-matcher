import gradio as gr
from PyPDF2 import PdfReader
from transformers import pipeline

# Load a small summarization model
summarizer = pipeline("summarization", model="google/flan-t5-small")

def parse_pdf(pdf_files):
    all_texts = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file.name)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        all_texts.append((pdf_file.name, text))
    return all_texts

def summarize_cvs(cv_files):
    parsed_cvs = parse_pdf(cv_files)
    results = []

    for filename, text in parsed_cvs:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        results.append({
            "CV Filename": filename,
            "Summary": summary[0]['summary_text']
        })
    return results

with gr.Blocks() as demo:
    gr.Markdown("## 🎯 Quick CV Summary Demo")
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    output = gr.Dataframe(headers=["CV Filename", "Summary"])
    run_button = gr.Button("Summarize CVs")
    run_button.click(fn=summarize_cvs, inputs=[cv_input], outputs=[output])

demo.launch()
