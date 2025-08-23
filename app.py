import gradio as gr
from PyPDF2 import PdfReader
import json

# Dummy function to simulate AI scoring
def score_cvs(cv_files, job_description):
    results = []
    for cv_file in cv_files:
        # Extract text from PDF
        reader = PdfReader(cv_file.name)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        # Here you would normally call a model/API
        # For testing, we simulate scores
        matched = ["SQL", "ETL"] if "SQL" in text else []
        missing = ["AI experience"] if "AI" not in text else []
        score = 75 if matched else 5
        explanation = f"{cv_file.name} has {len(matched)} matched skills."
        
        results.append({
            "CV Filename": cv_file.name,
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing),
            "Match Score": score,
            "Explanation": explanation
        })
    
    return json.dumps(results, indent=2)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Multi-CV Job Matcher (Test Version)")
    
    with gr.Row():
        cv_input = gr.File(file_types=[".pdf"], file_types_label="Upload CVs", file_types_multiple=True)
        job_desc_input = gr.Textbox(label="Job Description", placeholder="Paste the Job Description here...")
    
    output_box = gr.Code(label="Results (JSON)")

    run_btn = gr.Button("Run Matcher")
    run_btn.click(score_cvs, inputs=[cv_input, job_desc_input], outputs=[output_box])

demo.launch()

