import joblib


MODEL_PATH = "models/sentiment_pipeline.joblib"


def load_sentiment_model(model_path=MODEL_PATH):
    return joblib.load(model_path)


def predict_sentiment(df, model_bundle=None):
    if "comment" not in df.columns:
        raise ValueError("The comment column is missing.")

    if model_bundle is None:
        model_bundle = load_sentiment_model()

    vectorizer = model_bundle["vectorizer"]
    model = model_bundle["model"]

    result = df.copy()
    comment_vectors = vectorizer.transform(result["comment"].fillna(""))
    result["sentiment"] = model.predict(comment_vectors)
    return result
