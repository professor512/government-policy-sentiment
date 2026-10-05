import joblib
import os
import re


MODEL_PATH = "models/sentiment_model.pkl"


def clean_text(text):
    """
    Clean input text before prediction.
    """

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Model not found. "
            "Run src/train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_sentiment(text):

    model = load_model()

    cleaned_text = clean_text(text)

    prediction = model.predict(
        [cleaned_text]
    )[0]

    return prediction


if __name__ == "__main__":

    print("\nGovernment Policy Sentiment Analyzer")
    print("-" * 45)

    text = input(
        "\nEnter a policy statement/opinion: "
    )

    result = predict_sentiment(text)

    print(
        f"\nPredicted Sentiment: {result.upper()}"
    )