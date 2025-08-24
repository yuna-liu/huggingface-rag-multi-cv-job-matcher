import gradio as gr
from PyPDF2 import PdfReader
from collections import Counter

def parse_pdf_keywords(pdf_files, job_description):
    results = []
    job_words = set(job_description.lower().split())
    
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file.name)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        words = [w.lower() for w in text.split()]
        counter = Counter(words)
        matched = [w for w in counter if w in job_words]
        results.append({
            "CV Filename": pdf_file.name,
            "Matched Keywords": ", ".join(matched[:5]),
            "Total Matches": len(matched)
        })
    return results

with gr.Blocks() as demo:
    gr.Markdown("## Quick CV Matcher Demo")
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste Job Description here...", label="Job Description")
    output = gr.Dataframe(headers=["CV Filename", "Matched Keywords", "Total Matches"])
    run_button = gr.Button("Analyze CVs")
    
    run_button.click(fn=parse_pdf_keywords, inputs=[cv_input, job_input], outputs=[output])

demo.launch()
