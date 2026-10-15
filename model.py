from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocessing import clean_comment_text


LABELS = ["positive", "neutral", "negative"]
SET_ASIDE_LABEL = "none"
DATA_PATH = Path("data/comments_500.csv")
MODEL_PATH = Path("models/sentiment_pipeline.joblib")
REPORT_DIR = Path("reports")


def prepare_labeled_comments(df):
    required = {"comment", "sentiment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    prepared = df[["comment", "sentiment"]].copy()
    prepared = prepared.dropna(subset=["comment", "sentiment"])
    prepared["comment"] = prepared["comment"].map(clean_comment_text)
    prepared["sentiment"] = prepared["sentiment"].astype(str).str.lower().str.strip()
    prepared = prepared[prepared["comment"] != ""]

    # "None" marks comments without a clear opinion, so it is not a sentiment class.
    prepared = prepared[prepared["sentiment"].isin(LABELS)]
    prepared = prepared.drop_duplicates(subset=["comment"])
    return prepared.reset_index(drop=True)


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Naive Bayes": MultinomialNB(),
    }


def build_vectorizer():
    return TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def train_and_evaluate_model(
    df,
    model_path=MODEL_PATH,
    report_dir=REPORT_DIR,
):
    prepared = prepare_labeled_comments(df)
    if len(prepared) < 15:
        raise ValueError("Not enough labeled comments to train and test the models.")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        prepared["comment"],
        prepared["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=prepared["sentiment"],
    )

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    test_results = []
    fitted_models = {}
    predictions = {}

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        fitted_models[model_name] = model
        predictions[model_name] = predicted
        test_results.append(
            {
                "model": model_name,
                "test_accuracy_percent": round(
                    accuracy_score(y_test, predicted) * 100,
                    2,
                ),
                "test_macro_f1": round(
                    f1_score(
                        y_test,
                        predicted,
                        average="macro",
                        zero_division=0,
                    ),
                    4,
                ),
                "test_weighted_f1": round(
                    f1_score(
                        y_test,
                        predicted,
                        average="weighted",
                        zero_division=0,
                    ),
                    4,
                ),
            }
        )

    comparison = pd.DataFrame(test_results).sort_values(
        ["test_macro_f1", "test_accuracy_percent"],
        ascending=False,
    ).reset_index(drop=True)
    best_model_name = comparison.loc[0, "model"]
    best_model = fitted_models[best_model_name]
    best_prediction = predictions[best_model_name]
    comparison.insert(
        1,
        "selected_model",
        comparison["model"].eq(best_model_name),
    )

    matrix = confusion_matrix(y_test, best_prediction, labels=LABELS)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in LABELS],
        columns=[f"predicted_{label}" for label in LABELS],
    )

    test_predictions = pd.DataFrame(
        {
            "comment": X_test_text.to_numpy(),
            "actual_sentiment": y_test.to_numpy(),
            "predicted_sentiment": best_prediction,
        }
    )

    model_path = Path(model_path)
    report_dir = Path(report_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": best_model,
            "model_name": best_model_name,
            "labels": LABELS,
        },
        model_path,
    )

    comparison_path = report_dir / "model_comparison.csv"
    matrix_path = report_dir / "confusion_matrix.csv"
    predictions_path = report_dir / "test_predictions.csv"
    comparison.to_csv(comparison_path, index=False)
    matrix_df.to_csv(matrix_path)
    test_predictions.to_csv(predictions_path, index=False)

    best_row = comparison[comparison["model"] == best_model_name].iloc[0]
    return {
        "total_rows": len(df),
        "used_rows": len(prepared),
        "excluded_rows": len(df) - len(prepared),
        "set_aside_none_rows": int(
            df["sentiment"].astype(str).str.lower().str.strip().eq(SET_ASIDE_LABEL).sum()
        ),
        "training_rows": len(X_train_text),
        "test_rows": len(X_test_text),
        "best_model": best_model_name,
        "test_accuracy_percent": float(best_row["test_accuracy_percent"]),
        "test_macro_f1": float(best_row["test_macro_f1"]),
        "comparison": comparison,
        "confusion_matrix": matrix_df,
        "model_path": str(model_path),
        "comparison_path": str(comparison_path),
        "matrix_path": str(matrix_path),
        "predictions_path": str(predictions_path),
    }


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    results = train_and_evaluate_model(data)

    print(results["comparison"].to_string(index=False))
    print()
    print("Selected model:", results["best_model"])
    print("Comments set aside as none:", results["set_aside_none_rows"])
    print("Training comments:", results["training_rows"])
    print("Testing comments:", results["test_rows"])
    print("Confusion matrix:")
    print(results["confusion_matrix"].to_string())
    print()
    print("Saved model:", results["model_path"])
    print("Saved reports:", REPORT_DIR)
