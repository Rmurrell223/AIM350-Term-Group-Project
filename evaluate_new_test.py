from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from preprocessing import clean_comment_text
from sentiment_predictor import load_sentiment_model, predict_sentiment


LABELS = ["positive", "neutral", "negative"]
ALLOWED_REVIEW_LABELS = LABELS + ["none"]
DEFAULT_INPUT = Path("data/live_test_round2.csv")
TRAINING_DATA = Path("data/comments_500.csv")
REPORT_DIR = Path("reports")


def normalize_comment(value):
    return clean_comment_text(value).lower()


def evaluate_new_test(input_path=DEFAULT_INPUT, report_dir=REPORT_DIR):
    input_path = Path(input_path)
    test_data = pd.read_csv(input_path)
    input_rows = len(test_data)

    required = {"comment", "correct_label"}
    missing = required.difference(test_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    test_data["comment"] = test_data["comment"].fillna("").astype(str).str.strip()
    test_data["correct_label"] = (
        test_data["correct_label"].fillna("").astype(str).str.lower().str.strip()
    )

    incomplete = test_data[
        (test_data["comment"].map(clean_comment_text) == "")
        | ~test_data["correct_label"].isin(ALLOWED_REVIEW_LABELS)
    ]
    if not incomplete.empty:
        rows = ", ".join(str(index + 2) for index in incomplete.index[:10])
        raise ValueError(
            "Every row must have a comment and one valid correct_label. "
            f"Check CSV row(s): {rows}. Use positive, neutral, negative, or none."
        )

    if TRAINING_DATA.exists():
        training_data = pd.read_csv(TRAINING_DATA)
        training_comments = set(
            training_data["comment"].fillna("").map(normalize_comment)
        )
        overlap = test_data["comment"].map(normalize_comment).isin(training_comments)
        if overlap.any():
            raise ValueError(
                f"{int(overlap.sum())} test comment(s) also appear in the training "
                "dataset. Replace them before running the separate test."
            )

    none_rows = test_data[test_data["correct_label"] == "none"].copy()
    scored_rows = test_data[test_data["correct_label"].isin(LABELS)].copy()
    if scored_rows.empty:
        raise ValueError(
            "There are no positive, neutral, or negative rows to score."
        )

    evaluated = predict_sentiment(scored_rows, load_sentiment_model())
    evaluated = evaluated.rename(columns={"sentiment": "predicted_sentiment"})

    test_accuracy = accuracy_score(
        evaluated["correct_label"],
        evaluated["predicted_sentiment"],
    )
    test_f1 = f1_score(
        evaluated["correct_label"],
        evaluated["predicted_sentiment"],
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        evaluated["correct_label"],
        evaluated["predicted_sentiment"],
        labels=LABELS,
    )
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in LABELS],
        columns=[f"predicted_{label}" for label in LABELS],
    )

    summary = pd.DataFrame(
        [
            {"metric": "input_rows", "value": input_rows},
            {
                "metric": "set_aside_none_rows",
                "value": len(none_rows),
            },
            {"metric": "scored_rows", "value": len(evaluated)},
            {
                "metric": "accuracy_percent",
                "value": round(test_accuracy * 100, 2),
            },
            {
                "metric": "macro_f1",
                "value": round(test_f1, 4),
            },
        ]
    )

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_name = input_path.stem
    summary_path = report_dir / f"{report_name}_summary.csv"
    matrix_path = report_dir / f"{report_name}_confusion_matrix.csv"
    predictions_path = report_dir / f"{report_name}_predictions.csv"

    summary.to_csv(summary_path, index=False)
    matrix_df.to_csv(matrix_path)
    evaluated.to_csv(predictions_path, index=False)

    return {
        "summary": summary,
        "confusion_matrix": matrix_df,
        "summary_path": str(summary_path),
        "matrix_path": str(matrix_path),
        "predictions_path": str(predictions_path),
    }


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    results = evaluate_new_test(source)
    print(results["summary"].to_string(index=False))
    print()
    print("Confusion matrix:")
    print(results["confusion_matrix"].to_string())
    print()
    print("Saved reports:", REPORT_DIR)
