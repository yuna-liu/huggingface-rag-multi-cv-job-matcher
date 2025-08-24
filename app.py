import gradio as gr
from pdf2image import convert_from_path
import pytesseract

def parse_pdf_ocr(pdf_files):
    all_texts = []
    for pdf_file in pdf_files:
        pages = convert_from_path(pdf_file.name)
        text_per_page = []
        for i, page in enumerate(pages):
            text = pytesseract.image_to_string(page)
            text_per_page.append(f"Page {i+1}:\n{text}" if text.strip() else f"Page {i+1}: [No text found]")
        full_text = "\n\n".join(text_per_page)
        all_texts.append({
            "CV Filename": pdf_file.name,
            "Extracted Text": full_text
        })
    return all_texts

with gr.Blocks() as demo:
    gr.Markdown("## 📝 OCR PDF Text Extraction Test")
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    output = gr.Dataframe(headers=["CV Filename", "Extracted Text"])
    run_button = gr.Button("Extract Text")
    run_button.click(fn=parse_pdf_ocr, inputs=[cv_input], outputs=[output])

demo.launch()
