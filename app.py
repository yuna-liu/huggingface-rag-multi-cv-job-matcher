import gradio as gr

def test_upload(pdf_files):
    print("PDF Files received:", pdf_files)
    if not pdf_files:
        return "No files received"
    for f in pdf_files:
        print(f"Processing file: {f.name}")
    return f"Received {len(pdf_files)} files"

with gr.Blocks() as demo:
    cv_input = gr.Files(file_types=[".pdf"], label="Upload PDFs")
    output = gr.Textbox()
    cv_input.change(fn=test_upload, inputs=[cv_input], outputs=[output])

demo.launch()
