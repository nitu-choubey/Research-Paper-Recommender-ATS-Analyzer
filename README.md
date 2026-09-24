# SkillPaper AI - Research Paper Recommender & ATS Resume Analyzer

SkillPaper AI is a web application that offers two primary features:
1. **Research Paper Recommender:** Suggests relevant research papers based on input text using Sentence Transformers.
2. **ATS Resume Analyzer:** Evaluates resumes against job descriptions using Google Gemini AI.

## Project Structure
- `app.py`: Flask application routes and backend logic.
- `templates/`: HTML templates for UI.
- `models/`: Model files and embeddings.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python app.py`
3. Open browser at `http://127.0.0.1:5000`