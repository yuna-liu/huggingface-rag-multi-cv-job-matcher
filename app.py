# app.py
import gradio as gr
from PyPDF2 import PdfReader
from transformers import pipeline
import json
import os
from huggingface_hub import login

# Log in with HF token from secrets
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

from transformers import pipeline
# Now the model load will work
generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=hf_token,  # pass token here too
    device_map="auto",
    max_new_tokens=512
)


def parse_pdf(pdf_files):
    all_texts = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file.name)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
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

Please provide:
1. List of skills, experiences, and qualifications from the CV that match the job requirements.
2. List of skills/experience that are missing.
3. An overall match score (0-100%) with a short explanation.

Return the result in JSON format with keys: "matched", "missing", "score", "explanation".
"""
        raw_output = generator(prompt)[0]["generated_text"]

        try:
            json_str = raw_output[raw_output.find("{"):raw_output.rfind("}")+1]
            answer_json = json.loads(json_str)
        except:
            answer_json = {
                "matched": raw_output,
                "missing": "",
                "score": "0",
                "explanation": ""
            }

        results.append({
            "CV Filename": filename,
            "Matched Skills": ", ".join(answer_json.get("matched")) if isinstance(answer_json.get("matched"), list) else str(answer_json.get("matched")),
            "Missing Skills": ", ".join(answer_json.get("missing")) if isinstance(answer_json.get("missing"), list) else str(answer_json.get("missing")),
            "Match Score": answer_json.get("score"),
            "Explanation": str(answer_json.get("explanation"))
        })

    return results

with gr.Blocks() as demo:
    gr.Markdown("## 📄 Multi-CV Job Matcher — Mistral-7B-Instruct-v0.3 (Free on HF)")

    cv_input = gr.Files(label="Upload up to 5 CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste the Job Description here...", label="Job Description")
    output = gr.Dataframe(headers=["CV Filename", "Matched Skills", "Missing Skills", "Match Score", "Explanation"])
    run_button = gr.Button("Analyze CVs")

    run_button.click(fn=match_cvs_to_job, inputs=[cv_input, job_input], outputs=[output])

demo.launch()
