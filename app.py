import gradio as gr
from PyPDF2 import PdfReader

def parse_pdf(pdf_files):
    texts = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file.name)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        texts.append((pdf_file.name, text.lower()))
    return texts

def match_points(cv_files, job_description):
    job_keywords = [kw.strip().lower() for kw in job_description.split(",")]
    parsed_cvs = parse_pdf(cv_files)
    results = []

    for filename, text in parsed_cvs:
        matched_count = sum(1 for kw in job_keywords if kw in text)
        total_count = len(job_keywords)
        score = round(matched_count / total_count * 100, 1) if total_count > 0 else 0

        results.append({
            "CV Filename": filename,
            "Match Points": f"{score}%",
            "Matched Keywords": ", ".join([kw for kw in job_keywords if kw in text])
        })
    return results

with gr.Blocks() as demo:
    gr.Markdown("## 📊 CV Matcher (Keyword-based, Fast Version)")
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=3, placeholder="Enter job keywords separated by commas", label="Job Description Keywords")
    output = gr.Dataframe(headers=["CV Filename", "Match Points", "Matched Keywords"], datatype=["str","str","str"])
    run_button = gr.Button("Calculate Match Points")

    run_button.click(fn=match_points, inputs=[cv_input, job_input], outputs=[output])

demo.launch()
