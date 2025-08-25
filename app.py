# File: app.py
import gradio as gr
import pdfplumber
import re
import os
import tempfile
import shutil
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ----------------------------
# Setup Temp Directory
# ----------------------------
TEMP_DIR = os.path.join(tempfile.gettempdir(), "TempFile")
os.makedirs(TEMP_DIR, exist_ok=True)

# ----------------------------
# Models
# ----------------------------
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# Helpers
# ----------------------------
def save_temp_files(pdf_files):
    saved_paths = []
    for pdf_file in pdf_files:
        temp_path = os.path.join(TEMP_DIR, os.path.basename(pdf_file.name))
        shutil.copy(pdf_file.name, temp_path)
        saved_paths.append(temp_path)
    return saved_paths

def parse_pdf(pdf_paths):
    all_texts = []
    for path in pdf_paths:
        with pdfplumber.open(path) as pdf:
            text_pages = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_pages.append(f"--- Page {i+1} ---\n{page_text}")
            text = "\n".join(text_pages)
            all_texts.append((os.path.basename(path), text))
    return all_texts

def show_pdf_text(cv_files):
    if not cv_files:
        return "No files uploaded"
    pdf_paths = save_temp_files(cv_files)
    parsed = parse_pdf(pdf_paths)
    display_text = ""
    for filename, text in parsed:
        display_text += f"===== {filename} =====\n{text}\n\n"
    return display_text or "No text found in uploaded PDFs."

# ----------------------------
# Keyword Matching
# ----------------------------
stopwords = set([
    "and", "or", "the", "a", "an", "to", "of", "in", "on", "for", "with", "by", "at", "from",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "it", "as", "about"
])

def keyword_match(cv_text, job_text):
    cv_words = set(w for w in re.findall(r'\b\w+\b', cv_text.lower()) if w not in stopwords)
    job_words = set(w for w in re.findall(r'\b\w+\b', job_text.lower()) if w not in stopwords)
    matched = sorted(cv_words & job_words)
    missing = sorted(job_words - cv_words)
    score = round(len(matched) / max(1, len(job_words)) * 100, 2)  # percentage
    return matched, missing, score

def match_cvs_to_job(cv_files, job_description):
    if not cv_files:
        return [["No file", "", "", "0%"]]
    if not job_description.strip():
        return [["No job description", "", "", "0%"]]
    
    pdf_paths = save_temp_files(cv_files)
    parsed_cvs = parse_pdf(pdf_paths)
    
    results = []
    for filename, text in parsed_cvs:
        matched, missing, score = keyword_match(text, job_description)
        results.append([
            filename,
            ", ".join(matched[:15]) or "(none)",
            ", ".join(missing[:15]) or "(none)",
            f"{score}%"
        ])
    return results

# ----------------------------
# Summarization
# ----------------------------
def summarize_cv(cv_files):
    if not cv_files:
        return "No files uploaded"
    pdf_paths = save_temp_files(cv_files)
    parsed = parse_pdf(pdf_paths)
    summaries = ""
    for filename, text in parsed:
        if not text.strip():
            summaries += f"===== {filename} =====\n(No text)\n\n"
            continue
        short_text = text[:2000]
        summary = summarizer(short_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        summaries += f"===== {filename} =====\n{summary}\n\n"
    return summaries

# ----------------------------
# RAG Functions
# ----------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def build_faiss_index(cv_chunks):
    texts = []
    for filename, chunks in cv_chunks.items():
        for chunk in chunks:
            texts.append((filename, chunk))
    embeddings = embedder.encode([t[1] for t in texts], convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, texts

def retrieve(query, index, texts, top_k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    results = []
    for idx in I[0]:
        filename, chunk = texts[idx]
        results.append((filename, chunk))
    return results

def ask_question(cv_files, question):
    if not cv_files:
        return "No files uploaded"
    if not question.strip():
        return "Please enter a question"
    
    pdf_paths = save_temp_files(cv_files)
    parsed = parse_pdf(pdf_paths)

    cv_chunks = {fname: chunk_text(text) for fname, text in parsed}
    index, texts = build_faiss_index(cv_chunks)
    results = retrieve(question, index, texts)

    # Filter out empty or whitespace-only chunks
    valid_results = [(fname, chunk) for fname, chunk in results if chunk.strip()]
    if not valid_results:
        return "No relevant information found."

    summaries = []
    for filename, chunk in valid_results:
        try:
            summary = summarizer(chunk, max_length=80, min_length=20, do_sample=False)[0]['summary_text']
            summaries.append(f"**{filename}**: {summary}")
        except Exception:
            summaries.append(f"**{filename}**: (Unable to summarize this section)")

    return "\n".join(summaries) if summaries else "No relevant information found."


# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## ⚡ Instant CV Matcher + PDF Debug + Summary + RAG Search")

    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste Job Description here...", label="Job Description")

    with gr.Row():
        analyze_button = gr.Button("Analyze CVs")
        debug_button = gr.Button("Show PDF Texts")
        summary_button = gr.Button("Summarize CVs")

    output_table = gr.Dataframe(headers=["CV Filename", "Matched Skills", "Missing Skills", "Match Score"], type="array")
    output_text = gr.Textbox(lines=20, label="PDF Text Debug Output")
    output_summary = gr.Textbox(lines=10, label="Quick CV Summaries")

    analyze_button.click(fn=match_cvs_to_job, inputs=[cv_input, job_input], outputs=[output_table])
    debug_button.click(fn=show_pdf_text, inputs=[cv_input], outputs=[output_text])
    summary_button.click(fn=summarize_cv, inputs=[cv_input], outputs=[output_summary])

    gr.Markdown("### 🔍 Ask a Question about Candidates (RAG Search)")
    question_input = gr.Textbox(lines=2, placeholder="Example: Who has Azure ML pipeline experience?", label="Query")
    question_output = gr.Markdown()
    ask_button = gr.Button("Ask About Candidates")

    # Instant feedback message
    def show_processing_message():
        return "⏳ Processing your request, please wait..."

    def run_rag(cv_files, question):
        return ask_question(cv_files, question)

    ask_button.click(fn=show_processing_message, inputs=[], outputs=[question_output]) \
              .then(fn=run_rag, inputs=[cv_input, question_input], outputs=[question_output])

    gr.Markdown("""
    **💡 Sample Questions:**
    - Who has Azure ML pipeline experience?
    - Find people with RAG pipeline implementation experience.
    """)

demo.launch(share=True)
