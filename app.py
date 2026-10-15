from pathlib import Path

import streamlit as st
import pandas as pd
from model import train_and_evaluate_model
from youtube_fetch import search_videos, fetch_comments
from sentiment_predictor import load_sentiment_model
from sentiment_predictor import predict_sentiment
from topic_clustering import cluster_negative_comments

from preprocessing import clean_comments_dataframe


def fetch_youtube_comments(query_term, max_videos=10, comments_per_video=10):
    videos = search_videos(query_term, max_results=max_videos)
    all_comments = []
    for video in videos:
        video_comments = fetch_comments(
            video["video_id"],
            max_comments=comments_per_video,
        )
        if not video_comments.empty:
            all_comments.append(video_comments)
    if all_comments:
        return pd.concat(all_comments, ignore_index=True)
    return pd.DataFrame()


@st.cache_resource
def get_sentiment_model():
    try:
        return load_sentiment_model()
    except Exception:
        return None

st.set_page_config(page_title="Sentiment Tracker", layout="wide")
st.title("📊 YouTube Sentiment Tracker Dashboard")

with st.expander("Model performance test"):
    st.write(
        "Comments marked none are set aside because they do not contain a clear "
        "opinion. Of the remaining comments, 80% train both models and 20% test "
        "their predictions."
    )

    if st.button("Run model performance test"):
        try:
            labeled_data = pd.read_csv("data/comments_500.csv")
            st.session_state["training_results"] = train_and_evaluate_model(
                labeled_data
            )
            get_sentiment_model.clear()
        except Exception as error:
            st.error(f"The model test could not run: {error}")

    saved_comparison = Path("reports/model_comparison.csv")
    saved_matrix = Path("reports/confusion_matrix.csv")

    if "training_results" in st.session_state:
        performance = st.session_state["training_results"]
        comparison = performance["comparison"].copy()
        matrix = performance["confusion_matrix"]
        st.write(
            f"Usable comments: {performance['used_rows']} | "
            f"Set aside as none: {performance['set_aside_none_rows']} | "
            f"Training: {performance['training_rows']} | "
            f"Testing: {performance['test_rows']} | "
            f"Selected model: {performance['best_model']}"
        )
        st.dataframe(comparison, width="stretch")
        st.write("Confusion matrix: rows are actual labels; columns are predictions.")
        st.dataframe(matrix, width="stretch")
    elif saved_comparison.exists() and saved_matrix.exists():
        st.dataframe(pd.read_csv(saved_comparison), width="stretch")
        st.write("Confusion matrix: rows are actual labels; columns are predictions.")
        st.dataframe(pd.read_csv(saved_matrix, index_col=0), width="stretch")

model_bundle = get_sentiment_model()

st.sidebar.header("Search Parameters")
brand = st.sidebar.text_input("Brand Name", "Apple")
product = st.sidebar.text_input("Product Name", "iPhone 15")
search_button = st.sidebar.button("Run Analysis")

if search_button:
    query = f"{brand} {product}"
    st.info(f"Fetching comments for query: **{query}**...")

    try:
        df_comments = fetch_youtube_comments(query_term=query)
    except Exception as error:
        st.error(f"Comments could not be loaded: {error}")
        st.stop()

    if df_comments.empty:
        st.warning("No comments were returned for this search.")
    else:
        df_comments = clean_comments_dataframe(df_comments)

        if model_bundle is not None:
            df_comments = predict_sentiment(df_comments, model_bundle)
        else:
            st.error("The trained sentiment model could not be loaded.")
            st.stop()

        st.subheader("Comment Summary")
        pos_count = (df_comments["sentiment"] == "positive").sum()
        neg_count = (df_comments["sentiment"] == "negative").sum()
        neu_count = (df_comments["sentiment"] == "neutral").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Positive 🟢", pos_count)
        col2.metric("Negative 🔴", neg_count)
        col3.metric("Neutral ⚪", neu_count)

        st.divider()

        st.subheader("Classified YouTube Comments")
        display_columns = [
            column for column in ["date", "comment", "sentiment"]
            if column in df_comments.columns
        ]
        st.dataframe(df_comments[display_columns], width="stretch")

        review_columns = [
            column for column in ["date", "comment"]
            if column in df_comments.columns
        ]
        review_file = df_comments[review_columns].drop_duplicates(
            subset="comment"
        )
        review_file = review_file.sample(
            n=min(50, len(review_file)),
            random_state=42,
        ).reset_index(drop=True)
        review_file.insert(0, "search", query)
        review_file.insert(1, "row_number", range(1, len(review_file) + 1))
        review_file["correct_label"] = ""
        st.download_button(
            "Download a blind sample for a separate test",
            data=review_file.to_csv(index=False).encode("utf-8-sig"),
            file_name="live_test_round2.csv",
            mime="text/csv",
        )
        st.caption(
            "The model predicts positive, negative, or neutral. Comments without "
            "a clear product opinion were set aside when the training data was made."
        )

        negative_topics = cluster_negative_comments(df_comments)
        if not negative_topics.empty:
            st.subheader("Groups of Similar Negative Comments")
            st.dataframe(
                negative_topics[["topic_group", "topic_keywords", "comment"]],
                width="stretch"
            )
