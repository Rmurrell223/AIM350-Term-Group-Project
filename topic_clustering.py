import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_negative_comments(df, max_clusters=3):
    negative = df[
        df["sentiment"].astype(str).str.lower() == "negative"
    ].copy()

    if negative.empty:
        negative["topic_group"] = pd.Series(dtype="object")
        negative["topic_keywords"] = pd.Series(dtype="object")
        return negative

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)

    try:
        comment_vectors = vectorizer.fit_transform(negative["comment"].fillna(""))
    except ValueError:
        negative["topic_group"] = "Comment group 1"
        negative["topic_keywords"] = "No common keywords"
        return negative

    cluster_count = min(max_clusters, len(negative))
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    negative["cluster_number"] = model.fit_predict(comment_vectors)

    words = vectorizer.get_feature_names_out()
    group_keywords = {}
    for group_number, center in enumerate(model.cluster_centers_):
        top_words = words[center.argsort()[-3:][::-1]]
        group_keywords[group_number] = ", ".join(top_words)

    negative["topic_group"] = negative["cluster_number"].apply(
        lambda value: f"Comment group {value + 1}"
    )
    negative["topic_keywords"] = negative["cluster_number"].map(group_keywords)
    return negative
