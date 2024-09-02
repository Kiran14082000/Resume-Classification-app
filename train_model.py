import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# 1. Data Loading
df = pd.read_csv('data/UpdatedResumeDataSet.csv')

# 2. Text Preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_resume(text):
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower()
    
    # Tokenization and Lemmatization
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    # Keep only nouns, adjectives, and verbs
    cleaned_tokens = [lemmatizer.lemmatize(word) for word, tag in pos_tags if word not in stop_words and tag.startswith(('N', 'V', 'J'))]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

df['Cleaned_Resume'] = df['Resume'].apply(clean_resume)

# 3. Feature Extraction using TF-IDF with n-grams
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), analyzer='word')
X = tfidf.fit_transform(df['Cleaned_Resume']).toarray()

# Encode the target labels (Categories)
le = LabelEncoder()
y = le.fit_transform(df['Category'])

# Save the Label Encoder and TF-IDF Vectorizer
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

# 4. Model Training with Ensemble Methods
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
logistic_regression = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(n_estimators=100)

# Ensemble using Voting Classifier
ensemble_model = VotingClassifier(estimators=[('lr', logistic_regression), ('rf', random_forest)], voting='soft')

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'lr__C': [0.1, 1, 10],
    'rf__n_estimators': [100, 200]
}

grid_search = GridSearchCV(estimator=ensemble_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Save the best ensemble model
joblib.dump(best_model, 'models/ensemble_model.pkl')

# 5. Model Evaluation
y_pred = best_model.predict(X_test)
print("Ensemble Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
