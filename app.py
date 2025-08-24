# File: app.py
import gradio as gr
import pdfplumber
import json
from transformers import pipeline

# === Load a free HF model pipeline ===
# For demo, we use a small instruct model
matcher = pipeline(
    "text-generation", 
    model="OpenAssistant/oasst-sft-1-pythia-12b",  # you can replace with lighter free model
    device=0  # use GPU if available, or remove device argument for CPU
)

def parse_pdf(pdf_files):
    all_texts = []
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        with pdfplumber.open(pdf_file.name) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            all_texts.append((pdf_file.name, text))
    return all_texts

def match_cvs_to_job(cv_files, job_description):
    parsed_cvs = parse_pdf(cv_files)
    results = []

    for filename, text in parsed_cvs:
        prompt = f"""
You are a career assistant.

CV Text:
{text}

Job Description:
{job_description}

Please provide a simple summary:
1. List of skills/experiences in the CV that match the job description.
2. List of missing skills/experiences.
3. A match score from 0 to 100 (approximate).
Return it in JSON format with keys: "matched", "missing", "score".
"""
        response = matcher(prompt, max_length=512)[0]['generated_text']

        # try to convert to JSON
        try:
            answer_json = json.loads(response)
        except:
            answer_json = {
                "matched": response,
                "missing": "",
                "score": "0"
            }

        results.append({
            "CV Filename": filename,
            "Matched Skills": ", ".join(answer_json.get("matched")) if isinstance(answer_json.get("matched"), list) else str(answer_json.get("matched")),
            "Missing Skills": ", ".join(answer_json.get("missing")) if isinstance(answer_json.get("missing"), list) else str(answer_json.get("missing")),
            "Match Score": answer_json.get("score")
        })
    return results

# === Gradio Interface ===
with gr.Blocks() as demo:
    gr.Markdown("## 🎯 Multi-CV Job Matcher Demo (HF Free Model)")
    cv_input = gr.Files(label="Upload up to 5 CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste the Job Description here...", label="Job Description")
    output = gr.Dataframe(headers=["CV Filename", "Matched Skills", "Missing Skills", "Match Score"])
    run_button = gr.Button("Analyze CVs")
    
    run_button.click(fn=match_cvs_to_job, inputs=[cv_input, job_input], outputs=[output])

demo.launch()
