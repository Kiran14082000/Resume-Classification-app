import os
import re
import joblib
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import calibration_curve

# Load BERT model and tokenizer once, then reuse them
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

tokenizer, model = load_bert()

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + '\n'
    return text

def clean_text(text):
    if not isinstance(text, str):
        text = ""

    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word, tag in pos_tags
        if word not in stop_words and tag.startswith(('N', 'V', 'J'))
    ]

    return ' '.join(cleaned_tokens)

# Load saved models once and reuse them
@st.cache_resource
def load_saved_models():
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models"
    )

    label_encoder = joblib.load(
        os.path.join(models_dir, "label_encoder.pkl")
    )
    tfidf_vectorizer = joblib.load(
        os.path.join(models_dir, "tfidf_vectorizer.pkl")
    )
    ensemble_model = joblib.load(
        os.path.join(models_dir, "ensemble_model.pkl")
    )

    return label_encoder, tfidf_vectorizer, ensemble_model


label_encoder, tfidf_vectorizer, ensemble_model = load_saved_models()

# Streamlit UI
st.title("Resume Classification and Job Matching App")
st.write("Upload your resume as a PDF and enter the job description to predict the category and your probability of getting selected.")

# Resume and job description inputs
with st.form("resume_matching_form"):
    pdf_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"]
    )

    job_description = st.text_area(
        "Enter the job description"
    )

    st.caption(
        f"Job description length: {len(job_description)} characters"
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    if pdf_file is not None and job_description.strip() != "":
        # Extract and clean text from PDF
        resume_text = extract_text_from_pdf(pdf_file)
        cleaned_resume_text = clean_text(resume_text)
        
        # Vectorize the resume text with TF-IDF
        # Check for readable resume text
        if not cleaned_resume_text.strip():
            st.error("No readable text was extracted. Please upload a text-based PDF.")
            st.stop()

        vectorized_resume_text = tfidf_vectorizer.transform([cleaned_resume_text])

        # Experimental warning; threshold not validated
        probabilities = ensemble_model.predict_proba(vectorized_resume_text)[0]
        max_probability = float(np.max(probabilities))
        if max_probability < 0.12:
            st.warning(
                "This document may not match the resume categories supported by this app. "
                "Check that you uploaded a relevant resume; the category prediction may be unreliable."
            )
        
        # Predict the resume category using the ensemble model
        predicted_category = ensemble_model.predict(vectorized_resume_text)
        predicted_category_name = label_encoder.inverse_transform(predicted_category)[0]
        
        st.write(f"Predicted Resume Category: {predicted_category_name}")

        # Compare resume skills against job requirements
        resume_lower = resume_text.lower()
        job_lower = job_description.lower()

        skill_groups = {
            "Python": ["python", "python programming"],
            "SQL": ["sql", "structured query language", "t-sql", "pl/sql"],
            "Excel": ["excel", "microsoft excel", "spreadsheets", "pivot tables", "vlookup", "xlookup"],
            "Power BI": ["power bi", "powerbi", "dax"],
            "Tableau": ["tableau"],
            "Pandas": ["pandas"],
            "NumPy": ["numpy"],
            "R": ["r programming", "r language"],
            "Java": ["java programming"],
            "Data analysis": ["data analysis", "data analyst", "analyzing data", "analysing data", "analyze data", "analyse data"],
            "Data visualization": ["data visualization", "data visualisation", "visualizing data", "visualising data"],
            "Reporting": ["reporting", "reports", "report generation", "operational reporting"],
            "Dashboards": ["dashboard", "dashboards", "dashboard development"],
            "Statistics": ["statistics", "statistical analysis", "hypothesis testing", "regression analysis"],
            "Machine learning": ["machine learning", "ml models", "predictive modeling", "predictive modelling"],
            "Deep learning": ["deep learning", "neural networks"],
            "Natural language processing": ["natural language processing", "nlp", "text analytics"],
            "Generative AI": ["generative ai", "gen ai", "large language model", "large language models", "llm", "llms"],
            "Data cleaning": ["data cleaning", "cleaning data", "data cleansing", "data preprocessing", "preprocessing data"],
            "Data validation": ["data validation", "data quality", "data integrity"],
            "Data profiling": ["data profiling", "data assessment"],
            "Databases": ["database", "databases", "postgresql", "mysql", "sql server", "oracle database", "mongodb"],
            "ETL": ["etl", "extract transform load", "data pipelines", "data pipeline"],
            "Data warehousing": ["data warehouse", "data warehousing", "data warehouse design"],
            "Cloud computing": ["cloud computing", "aws", "amazon web services", "microsoft azure", "google cloud", "gcp"],
            "Big data": ["big data", "apache spark", "pyspark", "hadoop"],
            "Git": ["git", "github", "version control"],
            "APIs": ["api", "apis", "rest api", "restful api"],
            "Data modeling": ["data modeling", "data modelling", "dimensional modeling", "dimensional modelling"],
            "Business intelligence": ["business intelligence", "bi solutions", "bi reporting"],
            "Business analysis": ["business analysis", "business analyst", "requirements gathering", "business requirements"],
            "KPI tracking": ["kpi", "kpis", "key performance indicators", "performance metrics"],
            "Problem solving": ["problem solving", "problem-solving", "analytical thinking"],
            "Communication": ["communication", "communicate", "stakeholder communication", "presentation skills"],
            "Project management": ["project management", "project planning", "agile", "scrum"],
            "Data structures and algorithms": ["data structures", "algorithms", "data structures and algorithms"],
            "Scikit-learn": ["scikit-learn", "sklearn"],
            "TensorFlow": ["tensorflow"],
            "PyTorch": ["pytorch"],
            "XGBoost": ["xgboost"],
            "Model evaluation": ["model evaluation", "cross-validation", "cross validation", "precision", "recall", "f1 score"],
        }

        def find_skills(text):
            found = set()
            for skill, aliases in skill_groups.items():
                if any(
                    re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", text)
                    for alias in aliases
                ):
                    found.add(skill)
            return found

        resume_skills = find_skills(resume_lower)
        required_skills = find_skills(job_lower)
        matched_skills = sorted(resume_skills & required_skills)
        missing_skills = sorted(required_skills - resume_skills)

        st.subheader("Job Match Analysis")
        st.caption(
            "This is a basic keyword-coverage score, not a probability of getting hired."
        )

        if required_skills:
            match_score = round(
                len(matched_skills) / len(required_skills) * 100, 1
            )
            st.metric("Job skill coverage", f"{match_score}%")
            st.write(f"Skills detected in job description: {len(required_skills)}")

            if matched_skills:
                st.write("**Matching skills:** " + ", ".join(matched_skills))
            else:
                st.write("**Matching skills:** None detected.")

            if missing_skills:
                st.write("**Skills not found in resume:** " + ", ".join(missing_skills))
            else:
                st.success("All detected skills were found in the resume.")

            st.info(
                "This score checks only a limited built-in list of keywords. "
                "It does not evaluate experience quality, education requirements, "
                "or actual hiring outcomes."
            )
        else:
            st.warning(
                "No skills from the built-in list were detected in the job description. "
                "A skill-coverage score cannot be calculated."
            )

        st.warning(
            "Predicted category: " + predicted_category_name +
            ". This classifier may misclassify resumes whose categories were absent "
            "from its training dataset."
        )

        
      