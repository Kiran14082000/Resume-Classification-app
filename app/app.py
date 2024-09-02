import os
import re
import joblib
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import PyPDF2
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import calibration_curve

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower()
    return text

# Load saved models
label_encoder = joblib.load('../models/label_encoder.pkl')
tfidf_vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
ensemble_model = joblib.load('../models/ensemble_model.pkl')

# Streamlit UI
st.title("Resume Classification and Job Matching App")
st.write("Upload your resume as a PDF and enter the job description to predict the category and your probability of getting selected.")

# Resume PDF upload
pdf_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

# Job description input
job_description = st.text_area("Enter the job description")

if st.button("Predict"):
    if pdf_file is not None and job_description.strip() != "":
        # Extract and clean text from PDF
        resume_text = extract_text_from_pdf(pdf_file)
        cleaned_resume_text = clean_text(resume_text)
        
        # Vectorize the resume text with TF-IDF
        vectorized_resume_text = tfidf_vectorizer.transform([cleaned_resume_text])
        
        # Predict the resume category using the ensemble model
        predicted_category = ensemble_model.predict(vectorized_resume_text)
        predicted_category_name = label_encoder.inverse_transform(predicted_category)[0]
        
        st.write(f"Predicted Resume Category: {predicted_category_name}")
        
        # Clean the job description
        cleaned_job_description = clean_text(job_description)
        
        # Get BERT embeddings
        resume_embedding = get_bert_embeddings(cleaned_resume_text)
        job_description_embedding = get_bert_embeddings(cleaned_job_description)
        
        # Calculate cosine similarity
        similarity_score = cosine_similarity(resume_embedding, job_description_embedding)[0][0]

        # Dynamic threshold for probability adjustment
        if similarity_score > 0.9:
            probability_of_selection = round(similarity_score * 100, 2)
        elif similarity_score > 0.7:
            probability_of_selection = round(similarity_score * 75, 2)
        elif similarity_score > 0.5:
            probability_of_selection = round(similarity_score * 50, 2)
        else:
            probability_of_selection = 0
        
       # st.write(f"Similarity Score: {similarity_score}")
        st.write(f"Probability of getting selected: {probability_of_selection}%")
        
        # Provide feedback on the similarity score
        if probability_of_selection >= 80:
            st.success("Your resume is a strong match for this job description!")
        elif probability_of_selection>=60:
            st.warning("Your resume is litle bit ok, but change a few things")
        else:
            st.error("Your resume does not match well with the job description. You may need to update your resume.")
    else:
        st.write("Please upload a resume PDF and enter a job description.")
