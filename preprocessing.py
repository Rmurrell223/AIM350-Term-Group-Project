import re

import pandas as pd


def clean_comment_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_comments_dataframe(df):
    """Remove unusable rows and basic personal/contact information."""
    if df.empty:
        return df.copy()
    if "comment" not in df.columns:
        raise ValueError("The comment column is missing.")

    cleaned = df.dropna(subset=["comment"]).copy()
    cleaned["comment"] = cleaned["comment"].map(clean_comment_text)
    cleaned = cleaned[cleaned["comment"] != ""]
    cleaned = cleaned.drop_duplicates(subset=["comment"])
    return cleaned.reset_index(drop=True)

if __name__ == "__main__":
    raw_data = pd.DataFrame({
        "platform": ["YouTube", "YouTube", "YouTube"],
        "date": ["2026-10-12", "2026-10-12", "2026-10-12"],
        "comment": [
            "Check out my channel at http://example.com for more info! Email me at test@test.com",
            "This dishwasher is amazing!! So quiet. Call me at 123-456-7890.",
            "This dishwasher is amazing!! So quiet."
        ]
    })
    print(clean_comments_dataframe(raw_data))
