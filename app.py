import streamlit as st
import pandas as pd

# Page Title & Configuration
st.set_page_config(page_title="AI Sentiment & Trend Tracker", layout="wide")

st.title("📊 Real-Time Sentiment & Trend Tracker")
st.markdown("Analyze live customer reviews and sentiment patterns using AI.")

# Sidebar for User Inputs (matching your proposal's design)
st.sidebar.header("Search Parameters")
brand_name = st.sidebar.text_input("Brand Name", placeholder="e.g., GE, Apple")
product_name = st.sidebar.text_input("Product / Subject", placeholder="e.g., dishwasher, battery life")
timeframe = st.sidebar.selectbox("Timeframe", ["Past Week", "Past Month", "Past Year", "All Time"])

run_analysis = st.sidebar.button("Run Sentiment Analysis")

# Main Dashboard Area
if run_analysis:
    if not brand_name or not product_name:
        st.warning("⚠️ Please enter both a Brand Name and a Product/Subject to run the analysis.")
    else:
        st.success(f"Searching live data for **{brand_name} ({product_name})** over the **{timeframe.lower()}**...")
        
        # Placeholder for where the data & machine learning results will go later
        st.info("🔄 Connecting to data streams and running scikit-learn models...")
        
        # Example dummy table to show what the output will look like
        sample_data = pd.DataFrame({
            "Platform": ["YouTube", "YouTube", "Hacker News"],
            "Date": ["2026-10-12", "2026-10-11", "2026-10-10"],
            "Comment": [f"Really love the new {product_name} by {brand_name}, works great!", 
                        f"Had some issues with {product_name}, not super happy.", 
                        "Decent overall, standard tech performance."],
            "Sentiment": ["Positive", "Negative", "Neutral"]
        })
        
        st.subheader("Live Results & Classification")
        st.dataframe(sample_data, use_container_width=True)
else:
    st.info("👈 Enter your search parameters in the sidebar and click **Run Sentiment Analysis** to begin.")
