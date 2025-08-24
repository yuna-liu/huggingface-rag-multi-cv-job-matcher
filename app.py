# File: app.py
import gradio as gr
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Load a free HF model (Mistral-7B-Instruct if possible)
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def parse_pdf(pdf_files):
    texts = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file.name)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        texts.append((pdf_file.name, text))
    return texts

def extract_skills(cv_files, job_description):
    parsed_cvs = parse_pdf(cv_files)
    results = []

    for filename, text in parsed_cvs:
        prompt = f"""
You are a career assistant.
CV Text:
{text}

Job Description:
{job_description}

Extract:
1. Key skills
2. Key experiences
3. Overall match points (0-100)

Return as JSON with keys: "skills", "experience", "match_points"
"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=300)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            answer_json = json.loads(answer)
        except:
            answer_json = {"skills": answer, "experience": "", "match_points": 0}

        results.append({
            "CV Filename": filename,
            "Skills": answer_json.get("skills"),
            "Experience": answer_json.get("experience"),
            "Match Points": answer_json.get("match_points")
        })

    return results

with gr.Blocks() as demo:
    gr.Markdown("## 🏷️ CV Skill & Experience Extractor (Free HF Model)")
    cv_input = gr.Files(label="Upload CV PDFs (.pdf)", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste Job Description here...", label="Job Description")
    output = gr.Dataframe(headers=["CV Filename", "Skills", "Experience", "Match Points"])
    run_button = gr.Button("Analyze CVs")
    
    run_button.click(fn=extract_skills, inputs=[cv_input, job_input], outputs=[output])

demo.launch()
