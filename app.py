import gradio as gr
from PyPDF2 import PdfReader
from transformers import pipeline

# Load small, fast, free model
generator = pipeline("text2text-generation", model="google/flan-t5-base")

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
        1. Matched skills/experiences
        2. Missing skills/experiences
        3. Overall match score (0-100%) and explanation
        """
        response = generator(prompt, max_length=512, truncation=True)[0]["generated_text"]
        
        # Ensure display is plain text
        results.append({
            "CV Filename": filename,
            "Analysis": response
        })
    return results

with gr.Blocks() as demo:
    gr.Markdown("## 🎯 Multi-CV Job Matcher (Free Model Version)")
    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste the Job Description here...", label="Job Description")
    output = gr.Dataframe(headers=["CV Filename", "Analysis"], datatype=["str", "str"])
    run_button = gr.Button("Analyze CVs")

    run_button.click(fn=match_cvs_to_job, inputs=[cv_input, job_input], outputs=[output])

demo.launch()

