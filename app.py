import gradio as gr
import pdfplumber

def parse_pdf(pdf_file):
    if not pdf_file:
        return "No file uploaded"
    text_all = ""
    with pdfplumber.open(pdf_file.name) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_all += f"--- Page {i+1} ---\n{text}\n"
    if not text_all:
        return "No text found in PDF"
    return text_all

with gr.Blocks() as demo:
    gr.Markdown("## PDF Text Extractor Demo")
    cv_input = gr.File(file_types=[".pdf"], label="Upload a CV PDF")
    output = gr.Textbox(lines=20)
    run_button = gr.Button("Extract Text")

    run_button.click(fn=parse_pdf, inputs=[cv_input], outputs=[output])

demo.launch()
