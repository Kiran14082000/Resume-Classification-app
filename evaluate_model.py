
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# 1. Load data and remove duplicate resume texts
df = pd.read_csv("data/UpdatedResumeDataSet.csv")
df = df.drop_duplicates(subset=["Resume"]).reset_index(drop=True)

X = df["Resume"].fillna("")
y = df["Category"]

print("Unique resumes:", len(df))
print("Job categories:", y.nunique())

# 2. Configure cross-validation
min_class_count = y.value_counts().min()
n_splits = min(3, min_class_count)

if n_splits < 2:
    raise ValueError("Not enough unique resumes for cross-validation.")

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

# 3. Build a leakage-safe pipeline
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LinearSVC(random_state=42)
    )
])

# 4. Test different configurations
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [5000, 10000, None],
    "tfidf__min_df": [1, 2],
    "classifier__C": [0.5, 1, 2, 4],
    "classifier__class_weight": [None, "balanced"]
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    error_score="raise"
)

print(f"\nRunning {n_splits}-fold SVM hyperparameter tuning...")
search.fit(X, y)

# 5. Display the best results
print("\nBest cross-validation macro F1 (%):",
      round(search.best_score_ * 100, 2))

print("\nBest parameters:")
for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")

# 6. Show the top 10 configurations
results = pd.DataFrame(search.cv_results_)

top_results = results[
    [
        "mean_test_score",
        "std_test_score",
        "params"
    ]
].sort_values(
    by="mean_test_score",
    ascending=False
).head(10)

top_results = top_results.rename(columns={
    "mean_test_score": "Mean Macro F1",
    "std_test_score": "F1 Standard Deviation",
    "params": "Parameters"
})

top_results["Mean Macro F1 (%)"] = (
    top_results.pop("Mean Macro F1") * 100
).round(2)

top_results["F1 Standard Deviation"] = (
    top_results["F1 Standard Deviation"] * 100
).round(2)

print("\nTop 10 configurations:")
print(top_results.to_string(index=False))
