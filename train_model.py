
import os
import re
import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "data", "UpdatedResumeDataSet.csv"
)
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Download required NLTK resources
for resource in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Use the same cleaning logic as app/app.py
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    if not isinstance(text, str):
        text = ""

    text = re.sub(r"\n|\r", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()

    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word, tag in pos_tags
        if word not in stop_words
        and tag.startswith(("N", "V", "J"))
    ]

    return " ".join(cleaned_tokens)


# 1. Load data and remove duplicate resume texts
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["Category", "Resume"]).copy()
df["Resume"] = df["Resume"].astype(str)
df = df.drop_duplicates(subset=["Resume"]).reset_index(drop=True)

print("Unique resumes:", len(df))
print("Categories:", df["Category"].nunique())
print("\nResumes per category:")
print(df["Category"].value_counts().sort_values().to_string())

# 2. Clean resume text consistently with the app
df["Cleaned_Resume"] = df["Resume"].apply(clean_text)

# Remove empty cleaned documents
df = df[df["Cleaned_Resume"].str.strip() != ""].reset_index(drop=True)

X = df["Cleaned_Resume"]
y_text = df["Category"]

# 3. Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

# Some categories have only 3 unique examples.
# A 3-fold stratified split is the maximum sensible CV here.
min_class_count = int(y_text.value_counts().min())
n_splits = min(3, min_class_count)

if n_splits < 2:
    raise ValueError(
        "Not enough unique resumes per category for stratified evaluation."
    )

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42,
)

# 4. Leakage-safe pipeline for cross-validation
def make_pipeline():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 1),
                min_df=2,
                max_features=5000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                C=1,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


print(f"\nRunning {n_splits}-fold cross-validation...")
cv_scores = cross_validate(
    make_pipeline(),
    X,
    y,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
    },
    n_jobs=-1,
    error_score="raise",
)

print(
    "Cross-validation accuracy: "
    f"{cv_scores['test_accuracy'].mean() * 100:.2f}%"
)
print(
    "Cross-validation macro F1: "
    f"{cv_scores['test_macro_f1'].mean() * 100:.2f}%"
)
print(
    "Macro F1 standard deviation: "
    f"{cv_scores['test_macro_f1'].std() * 100:.2f} percentage points"
)

# 5. Hold out a test set for an additional independent evaluation.
# Use stratification so categories are represented where possible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Fit vectorizer on training data only
final_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 1),
    min_df=2,
    max_features=5000,
    sublinear_tf=True,
)

X_train_vec = final_vectorizer.fit_transform(X_train)
X_test_vec = final_vectorizer.transform(X_test)

final_model = LinearSVC(
    C=1,
    class_weight="balanced",
    random_state=42,
)
final_model.fit(X_train_vec, y_train)

y_pred = final_model.predict(X_test_vec)

print("\nHeld-out test results")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro', zero_division=0) * 100:.2f}%")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)

# 6. Save compatible artifacts to temporary files first.
# Existing artifacts remain untouched until training and evaluation succeed.
joblib.dump(
    label_encoder,
    os.path.join(MODELS_DIR, "label_encoder_new.pkl"),
)
joblib.dump(
    final_vectorizer,
    os.path.join(MODELS_DIR, "tfidf_vectorizer_new.pkl"),
)
joblib.dump(
    final_model,
    os.path.join(MODELS_DIR, "ensemble_model_new.pkl"),
)

print("\nNew model artifacts saved:")
print("models/label_encoder_new.pkl")
print("models/tfidf_vectorizer_new.pkl")
print("models/ensemble_model_new.pkl")
print(
    "\nThe existing model files have NOT been replaced. "
    "Review the results before switching the app."
)
