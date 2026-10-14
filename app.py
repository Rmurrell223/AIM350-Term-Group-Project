import streamlit as st
import pandas as pd
import joblib
from youtube_fetch import search_videos, fetch_comments

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
def load_sentiment_model():
    try:
        model = joblib.load("models/sentiment_pipeline.joblib")
        return model
    except Exception as e:
        return None

model = load_sentiment_model()

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

        if "cleaned_comment" not in df_comments.columns and "comment" in df_comments.columns:
            df_comments["cleaned_comment"] = df_comments["comment"]

        # Predict sentiments using the model
        if model is not None and "cleaned_comment" in df_comments.columns:
            df_comments["sentiment"] = model.predict(df_comments["cleaned_comment"])
        else:
            df_comments["sentiment"] = "None"

        # Show positive, negative, neutral, and none totals
        st.subheader("Sentiment Summary")
        pos_count = (df_comments["sentiment"] == "Positive").sum()
        neg_count = (df_comments["sentiment"] == "Negative").sum()
        neu_count = (df_comments["sentiment"] == "Neutral").sum()
        none_count = (df_comments["sentiment"] == "None").sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Positive 🟢", pos_count)
        col2.metric("Negative 🔴", neg_count)
        col3.metric("Neutral ⚪", neu_count)
        col4.metric("None ❓", none_count)

        st.divider()

        # Display the classified comments table
        st.subheader("Classified YouTube Comments")
        st.dataframe(
            df_comments[["author", "date", "comment", "sentiment"]],
            use_container_width=True
        )