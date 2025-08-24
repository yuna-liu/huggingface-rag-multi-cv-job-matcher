import gradio as gr
from PyPDF2 import PdfReader
from transformers import pipeline

# Use free Mistral 7B Instruct v0.2 (open access)
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=512)

def parse_pdf(pdf_files):
    all_texts = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file.name)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        # limit size so model doesn't get overwhelmed
        text = text[:2000]
        all_texts.append((pdf_file.name, text))
    return all_texts

def match_cvs_to_job(cv_files, job_description):
    job_description = job_description[:1500]  # shorten for free models
    parsed_cvs = parse_pdf(cv_files)
    results = []
    
    for filename, text in parsed_cvs:
        prompt = f"""
You are a career assistant.
Compare the following CV with the job description.
List: matched skills, missing skills, and a match score (0-100%) with explanation.

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
    gr.Markdown("## 🎯 Multi-CV Job Matcher (Free Mistral Version)")
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste the Job Description here...", label="Job Description")
    output = gr.Dataframe(headers=["CV Filename", "Analysis"], datatype=["str", "str"])
    run_button = gr.Button("Analyze CVs")

    run_button.click(fn=match_cvs_to_job, inputs=[cv_input, job_input], outputs=[output])

demo.launch()
