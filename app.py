import gradio as gr
from PyPDF2 import PdfReader
from transformers import pipeline

# Use a small, fast free model
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=300)

def parse_pdf(pdf_files):
    all_texts = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file.name)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        text = text[:1000]  # keep short for small models
        all_texts.append((pdf_file.name, text))
    return all_texts

def match_cvs_to_job(cv_files, job_description):
    job_description = job_description[:800]
    parsed_cvs = parse_pdf(cv_files)
    results = []
    
    for filename, text in parsed_cvs:
        prompt = f"""
Compare this CV to the job description.
Return matched skills, missing skills, and a match score.

CV:
{text}

Job:
{job_description}
"""
        response = generator(prompt)[0]["generated_text"]
        results.append({
            "CV Filename": filename,
            "Analysis": response
        })
    return results

with gr.Blocks() as demo:
    gr.Markdown("## 🎯 Multi-CV Job Matcher (Fast Free Version)")
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste the Job Description here...", label="Job Description")
    output = gr.Dataframe(headers=["CV Filename", "Analysis"], datatype=["str", "str"])
    run_button = gr.Button("Analyze CVs")

    run_button.click(fn=match_cvs_to_job, inputs=[cv_input, job_input], outputs=[output])

demo.launch()
