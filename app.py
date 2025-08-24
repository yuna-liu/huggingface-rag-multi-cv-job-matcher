# File: app.py
import gradio as gr
import pdfplumber
import re
import os
import tempfile
import shutil
import math

# --- NEW: semantic embeddings + basic NLP ---
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk

# Ensure NLTK data (first run will download once)
try:
    _ = nltk.corpus.stopwords.words("english")
except:
    nltk.download("stopwords")
    nltk.download("punkt")

EN_SW_STOP = set(nltk.corpus.stopwords.words("english"))
# Minimal Swedish stoplist (add common fillers that kept showing up)
SE_STOP = {
    "och","att","det","som","en","i","på","för","med","till","av","är","ett","den","de","vi","du","din","dina",
    "använda","användning","analys","analysera","arbetslivserfarenhet","bidra","bred","del","denna","din","du",
    "erfarenhet","erfarenheter","förmåga","kommunicera","profil","roll","team","flexibelt","karriärutveckling",
    "innovativ","kund","kunder","värde","svenska","engelska","tal","skrift"
}
STOPWORDS = EN_SW_STOP | SE_STOP

# --- Keep your summarizer (same as before) ---
from transformers import pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# --- Load lightweight multilingual embedding model (CPU friendly) ---
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
emb_model = SentenceTransformer(EMB_MODEL_NAME)

# Temp dir for uploaded PDFs
TEMP_DIR = os.path.join(tempfile.gettempdir(), "TempFile")
os.makedirs(TEMP_DIR, exist_ok=True)

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

# ---- Helpers for semantic matching ----
def split_into_chunks(text, max_chars=1000):
    """Split long CV text into chunks for better semantic matching."""
    # split by paragraphs first
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + "\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    # fallback if somehow empty
    return chunks or [text[:max_chars]]

def tokenize_keep_terms(s):
    """Extract informative tokens, drop stopwords/digits, keep tech-like tokens."""
    raw = re.findall(r"[A-Za-zÅÄÖåäö0-9\+\#\.\-_/]+", s.lower())
    terms = []
    for w in raw:
        if w in STOPWORDS:
            continue
        if len(w) < 2:
            continue
        # keep techy tokens or acronyms (sql, dax, ai, ml, c#, .net)
        if any(ch.isdigit() for ch in w) or any(ch in "+#._/-" for ch in w) or w.isupper() or w in {"sql","dax","ai","ml","bi"}:
            terms.append(w)
            continue
        # otherwise alphas with length >=3
        if w.isalpha() and len(w) >= 3:
            terms.append(w)
    return terms

def extract_matched_missing(job_text, cv_text, top_k=20):
    job_terms = tokenize_keep_terms(job_text)
    cv_terms = tokenize_keep_terms(cv_text)

    # frequency dicts
    from collections import Counter
    jfreq = Counter(job_terms)
    cfreq = Counter(cv_terms)

    # rank job terms by frequency (proxy for importance)
    ranked_job = [t for t,_ in jfreq.most_common()]
    matched = [t for t in ranked_job if t in cfreq][:top_k]
    missing = [t for t in ranked_job if t not in cfreq][:top_k]
    return matched, missing

def sim_to_percent(sim, lo=0.35, hi=0.85):
    """
    Map cosine similarity ~[0,1] into 0-100.
    lo/hi are soft calibration points (tweakable).
    """
    x = (sim - lo) / (hi - lo)
    x = max(0.0, min(1.0, x))
    return round(x * 100, 1)

def semantic_score(cv_text, job_text):
    """
    Compute a robust semantic score:
    - chunk the CV
    - embed chunks + job
    - use a weighted blend of max and mean similarity
    """
    chunks = split_into_chunks(cv_text, max_chars=1000)
    job_emb = emb_model.encode([job_text], normalize_embeddings=True)
    cv_embs = emb_model.encode(chunks, normalize_embeddings=True)

    sims = (cv_embs @ job_emb.T).squeeze(-1)  # cosine because normalized
    if sims.ndim == 0:
        sims = np.array([float(sims)])

    max_sim = float(np.max(sims))
    mean_sim = float(np.mean(sims))
    # blend: emphasize best matching chunk but keep overall quality
    blended = 0.7 * max_sim + 0.3 * mean_sim
    score = sim_to_percent(blended)
    return score

# ---- Replace your keyword_match with semantic version ----
def match_cvs_to_job(cv_files, job_description):
    if not cv_files:
        return [["No file", "", "", 0]]
    if not job_description.strip():
        return [["No job description", "", "", 0]]

    pdf_paths = save_temp_files(cv_files)
    parsed_cvs = parse_pdf(pdf_paths)

    rows = []
    for filename, text in parsed_cvs:
        if not text.strip():
            rows.append([filename, "(none)", "(none)", 0])
            continue

        # semantic score
        score = semantic_score(text, job_description)

        # matched/missing terms (filtered)
        matched, missing = extract_matched_missing(job_description, text, top_k=15)

        rows.append([
            filename,
            ", ".join(matched) or "(none)",
            ", ".join(missing) or "(none)",
            score
        ])
    return rows

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
        short_text = text[:2000]  # limit length for speed
        summary = summarizer(short_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        summaries += f"===== {filename} =====\n{summary}\n\n"
    return summaries

# ---- UI (unchanged) ----
with gr.Blocks() as demo:
    gr.Markdown("## ⚡ Instant CV Matcher + PDF Debug View + Quick Summary")

    cv_input = gr.Files(label="Upload CV PDFs", file_types=[".pdf"])
    job_input = gr.Textbox(lines=6, placeholder="Paste Job Description here (Swedish or English)...", label="Job Description")

    output_table = gr.Dataframe(
        headers=["CV Filename", "Matched Skills", "Missing Skills", "Match Score"],
        type="array"
    )
    output_text = gr.Textbox(lines=20, label="PDF Text Debug Output")
    output_summary = gr.Textbox(lines=10, label="Quick CV Summaries")

    analyze_button = gr.Button("Analyze CVs")
    debug_button = gr.Button("Show PDF Texts")
    summary_button = gr.Button("Summarize CVs")

    analyze_button.click(fn=match_cvs_to_job, inputs=[cv_input, job_input], outputs=[output_table])
    debug_button.click(fn=show_pdf_text, inputs=[cv_input], outputs=[output_text])
    summary_button.click(fn=summarize_cv, inputs=[cv_input], outputs=[output_summary])

demo.launch(share=True)