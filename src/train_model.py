import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/processed/cleaned_policies.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

X = df["clean_text"]
y = df["sentiment"]

print(f"Total records: {len(df)}")
print("\nSentiment distribution:")
print(y.value_counts())


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# TF-IDF Vectorizer
# --------------------------------------------------

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)


# --------------------------------------------------
# Models
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC(
        random_state=42
    )
}


# --------------------------------------------------
# Train Models
# --------------------------------------------------

results = []

best_model = None
best_model_name = None
best_f1 = -1


for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print("=" * 60)

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Select best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = model_name


# --------------------------------------------------
# Save Model Comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# --------------------------------------------------
# Save Best Model
# --------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "sentiment_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)


# --------------------------------------------------
# Save Test Dataset
# --------------------------------------------------

test_df = pd.DataFrame({
    "text": X_test,
    "actual_sentiment": y_test
})

test_df["predicted_sentiment"] = best_model.predict(
    X_test
)

test_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "test_predictions.csv"
    ),
    index=False
)


# --------------------------------------------------
# Final Output
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"\nBest Model: {best_model_name}")
print(f"Best F1 Score: {best_f1:.4f}")

print("\nModel comparison:")
print(
    results_df.to_string(
        index=False
    )
)

print("\nFiles created:")

print("✓ models/sentiment_model.pkl")
print("✓ models/model_comparison.csv")
print("✓ models/test_predictions.csv")