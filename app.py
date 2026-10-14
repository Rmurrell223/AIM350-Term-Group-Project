import streamlit as st
import pandas as pd
from youtube_fetch import search_videos, fetch_comments
from sentiment_predictor import load_sentiment_model
from sentiment_predictor import predict_sentiment
from topic_clustering import cluster_negative_comments

# Safely import preprocessing from your team's file
try:
    from preprocessing import clean_comments_dataframe
except ImportError:
    def clean_comments_dataframe(df):
        return df

# Wrapper function to connect your team's YouTube functions to your dashboard
def fetch_youtube_comments(query_term, max_results=50):
    try:
        videos = search_videos(query_term, max_results=10)
        all_comments = []
        for v in videos:
            df_v = fetch_comments(v["video_id"], max_comments=10)
            if not df_v.empty:
                all_comments.append(df_v)
        if all_comments:
            return pd.concat(all_comments, ignore_index=True)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Set up page layout
st.set_page_config(page_title="Sentiment Tracker", layout="wide")
st.title("📊 YouTube Sentiment Tracker Dashboard")

# Load the trained model saved by your team
@st.cache_resource
def get_sentiment_model():
    try:
        return load_sentiment_model()
    except Exception:
        return None

model_bundle = get_sentiment_model()

# Keep the brand and product input fields in the sidebar
st.sidebar.header("Search Parameters")
brand = st.sidebar.text_input("Brand Name", "Apple")
product = st.sidebar.text_input("Product Name", "iPhone 15")
search_button = st.sidebar.button("Run Analysis")

if search_button:
    query = f"{brand} {product}"
    st.info(f"Fetching comments for query: **{query}**...")

    # Connect the input to the YouTube search
    df_comments = fetch_youtube_comments(query_term=query)

    if df_comments.empty:
        st.warning("No comments found or API key missing.")
    else:
        # Preprocess and clean the dataframe using your team's function
        df_comments = clean_comments_dataframe(df_comments)

        # Predict sentiments using the model
        if model_bundle is not None:
            df_comments = predict_sentiment(df_comments, model_bundle)
        else:
            st.error("The trained sentiment model could not be loaded.")
            st.stop()

        # Show positive, negative, and neutral totals
        st.subheader("Sentiment Summary")
        pos_count = (df_comments["sentiment"] == "positive").sum()
        neg_count = (df_comments["sentiment"] == "negative").sum()
        neu_count = (df_comments["sentiment"] == "neutral").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Positive 🟢", pos_count)
        col2.metric("Negative 🔴", neg_count)
        col3.metric("Neutral ⚪", neu_count)

        st.divider()

        # Display the classified comments table
        st.subheader("Classified YouTube Comments")
        display_columns = [
            column for column in ["date", "comment", "sentiment"]
            if column in df_comments.columns
        ]
        st.dataframe(df_comments[display_columns], use_container_width=True)

        negative_topics = cluster_negative_comments(df_comments)
        if not negative_topics.empty:
            st.subheader("Common Issues in Negative Comments")
            st.dataframe(
                negative_topics[["topic_group", "topic_keywords", "comment"]],
                use_container_width=True
            )
