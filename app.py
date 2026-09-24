from flask import Flask, request, render_template, send_file
import torch
from sentence_transformers import util
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras
from PyPDF2 import PdfReader
import google.generativeai as genai
import io

app = Flask(__name__)

# -------------------- Research Paper Recommendation Setup --------------------
embeddings = pickle.load(open('models/embeddings.pkl', 'rb'))
sentences = pickle.load(open('models/sentences.pkl', 'rb'))
rec_model = pickle.load(open('models/rec_model.pkl', 'rb'))
loaded_model = keras.models.load_model("models/model.h5")

with open("models/text_vectorizer_config.pkl", "rb") as f:
    saved_text_vectorizer_config = pickle.load(f)
loaded_text_vectorizer = TextVectorization.from_config(saved_text_vectorizer_config)

with open("models/text_vectorizer_weights.pkl", "rb") as f:
    weights = pickle.load(f)
    loaded_text_vectorizer.set_weights(weights)

with open("models/vocab.pkl", "rb") as f:
    loaded_vocab = pickle.load(f)

def recommendation(input_paper):
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)
    return [sentences[i.item()] for i in top_similar_papers.indices]

# -------------------- ATS Resume Analyzer --------------------
class ATSAnalyzer:
    @staticmethod
    def extract_text_from_pdf(file_stream):
        try:
            pdf_reader = PdfReader(file_stream)
            return "".join(page.extract_text() for page in pdf_reader.pages)
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"

    @staticmethod
    def get_gemini_response(api_key, prompt, pdf_text, job_description):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content([prompt, pdf_text, job_description])
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

# -------------------- Flask Routes --------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/research', methods=['GET', 'POST'])
def research():
    recommended_papers = []
    input_paper = ""
    if request.method == 'POST':
        input_paper = request.form.get('input_paper')
        if input_paper:
            recommended_papers = recommendation(input_paper)
    return render_template('research.html', recommended_papers=recommended_papers, input_paper=input_paper)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/ats", methods=["GET", "POST"])
def ats():
    response = ""
    if request.method == "POST":
        api_key = request.form.get("api_key")
        job_description = request.form.get("job_description")
        analysis_type = request.form.get("analysis_type")
        resume_file = request.files.get("resume")

        if not (api_key and job_description and resume_file):
            response = " Please fill in all fields and upload your resume."
        else:
            pdf_text = ATSAnalyzer.extract_text_from_pdf(resume_file)
            if "Error" in pdf_text:
                response = pdf_text
            else:
                if analysis_type == "Detailed Review":
                    prompt = (
                        "You are an experienced HR expert. Review the resume and compare it with the job description. "
                        "Identify strengths, areas for improvement, and give a final verdict (Hire, Consider, Reject) "
                        "along with detailed reasoning. Keep your tone professional and structured."
                    )
                else:
                    prompt = (
                        "You are acting as an advanced Applicant Tracking System (ATS). "
                        "Analyze the resume and the job description provided below. "
                        "Your task is to assess how well the candidate matches the role.\n\n"
                        "First, calculate a match percentage between 0–100% based on alignment in skills, experience, education, and keywords.\n\n"
                        "Then, break down the results into the following categories:\n"
                        "1. Match: XX%\n"
                        "2. Strengths: List 2–3 specific aspects from the resume that align strongly with the job description.\n"
                        "3. Gaps or Missing Elements: List key skills, tools, or experience the resume lacks compared to the job description.\n"
                        "4. Final Assessment: A 2–3 sentence summary explaining whether this candidate is a strong, average, or poor fit based on your evaluation.\n\n"
                        "Keep the output short and formatted as shown. Use bullet points for readability."
                    )
                response = ATSAnalyzer.get_gemini_response(api_key, prompt, pdf_text, job_description)
    return render_template("ats.html", response=response)

@app.route("/download", methods=["POST"])
def download():
    analysis_text = request.form.get("analysis_text")
    return send_file(io.BytesIO(analysis_text.encode()), mimetype="text/plain", as_attachment=True, download_name="resume_analysis.txt")

if __name__ == '__main__':
    app.run(debug=True)