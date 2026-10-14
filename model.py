import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score


def train_and_evaluate_model(df):
    """
    Trains sentiment classifiers and compares Logistic Regression
    against Multinomial Naive Bayes using an 80/20 train-test split.
    """

    if len(df) < 5:
        return None, "Not enough data to train model yet. Need at least a few samples."

    # Remove unlabeled rows
    df = df.dropna(subset=["sentiment"])
    df = df[df["sentiment"].astype(str).str.lower() != "none"]

    if len(df) < 5:
        return None, "Not enough labeled data after filtering."

    X = df["comment"]
    y = df["sentiment"]

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english"
    )

    X_vectorized = vectorizer.fit_transform(X)

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)

    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(
        y_test,
        lr_pred,
        average="weighted",
        zero_division=0
    )

    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    nb_pred = nb_model.predict(X_test)

    nb_accuracy = accuracy_score(y_test, nb_pred)
    nb_f1 = f1_score(
        y_test,
        nb_pred,
        average="weighted",
        zero_division=0
    )

    # Keep original keys for compatibility
    results = {
        "accuracy": round(lr_accuracy * 100, 2),
        "f1_score": round(lr_f1, 2),

        "logistic_accuracy": round(lr_accuracy * 100, 2),
        "logistic_f1": round(lr_f1, 2),

        "naive_bayes_accuracy": round(nb_accuracy * 100, 2),
        "naive_bayes_f1": round(nb_f1, 2),

        "vectorizer": vectorizer,
        "model": lr_model,
        "naive_bayes_model": nb_model
    }

    return results