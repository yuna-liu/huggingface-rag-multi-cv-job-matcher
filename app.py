from PyPDF2 import PdfReader

def parse_pdf_debug(pdf_files):
    all_texts = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file.name)
        print(f"Processing: {pdf_file.name}")
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            print(f"\n--- Page {i+1} ---\n{page_text}\n")
            all_texts.append({
                "CV Filename": pdf_file.name,
                "Page": i+1,
                "Extracted Text": page_text if page_text else "[No text found]"
            })
    return all_texts

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("## 🛠 PDF Debug Text Extraction")
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    output = gr.Dataframe(headers=["CV Filename", "Page", "Extracted Text"])
    run_button = gr.Button("Debug Extract")
    run_button.click(fn=parse_pdf_debug, inputs=[cv_input], outputs=[output])

demo.launch()
