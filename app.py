import gradio as gr
from PyPDF2 import PdfReader

def parse_pdf_test(pdf_files):
    all_texts = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file.name)
        text_per_page = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text_per_page.append(f"Page {i+1}:\n{page_text}" if page_text else f"Page {i+1}: [No text found]")
        full_text = "\n\n".join(text_per_page)
        all_texts.append({
            "CV Filename": pdf_file.name,
            "Extracted Text": full_text
        })
    return all_texts

with gr.Blocks() as demo:
    gr.Markdown("## 📝 PDF Text Extraction Test")
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    output = gr.Dataframe(headers=["CV Filename", "Extracted Text"])
    run_button = gr.Button("Extract Text")
    run_button.click(fn=parse_pdf_test, inputs=[cv_input], outputs=[output])

demo.launch()
