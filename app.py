# File: app.py
import gradio as gr
from PyPDF2 import PdfReader
from openai import OpenAI
import json

# === OpenAI API key ===
client = OpenAI(api_key="YOUR_OPENAI_KEY")  # Replace with your key or use Secrets on Spaces

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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful career assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        answer_text = response.choices[0].message.content

        try:
            answer_json = json.loads(answer_text)
        except:
            answer_json = {
                "matched": answer_text,
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

# === Gradio Interface ===
with gr.Blocks() as demo:
    gr.Markdown("## 🎯 Multi-CV Job Matcher (Gradio Version)")
    with gr.Row():
        cv_input = gr.File(file_types=[".pdf"], file_types_multi=True, label="Upload up to 5 CV PDFs")
    job_input = gr.Textbox(lines=6, placeholder="Paste the Job Description here...", label="Job Description")
    output = gr.Dataframe(headers=["CV Filename", "Matched Skills", "Missing Skills", "Match Score", "Explanation"])
    run_button = gr.Button("Analyze CVs")

    run_button.click(fn=match_cvs_to_job, inputs=[cv_input, job_input], outputs=[output])

demo.launch()


