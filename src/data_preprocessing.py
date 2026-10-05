import re
import pandas as pd


def clean_text(text):
    """
    Clean policy text for machine learning.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset(input_path, output_path):
    """
    Read raw dataset, clean text and save processed dataset.
    """

    df = pd.read_csv(input_path)

    required_columns = [
        "policy_id",
        "policy_name",
        "category",
        "text",
        "sentiment"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing text/sentiment
    df = df.dropna(subset=["text", "sentiment"])

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Remove empty text
    df = df[df["clean_text"].str.len() > 0]

    # Save processed dataset
    df.to_csv(output_path, index=False)

    print("Preprocessing completed successfully.")
    print(f"Total records: {len(df)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":

    preprocess_dataset(
        "data/raw/policies.csv",
        "data/processed/cleaned_policies.csv"
    )