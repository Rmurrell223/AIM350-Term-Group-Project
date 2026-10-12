import pandas as pd
import re

def clean_comments_dataframe(df):
    """
    Cleans raw comments according to the project proposal:
    - Drops blank or missing comments
    - Removes duplicate comments
    - Strips out URLs/links, email addresses, phone numbers, and excess whitespace
    """
    if df.empty:
        return df

    # 1. Drop rows with missing comments
    df = df.dropna(subset=['comment'])
    
    # 2. Drop duplicate comments to keep unique insights
    df = df.drop_duplicates(subset=['comment'])
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        
        # Remove URLs (http/https/www)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone-like numbers (e.g., 555-123-4567 or similar patterns)
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
        
        # Remove extra whitespace, tabs, and newlines, then trim edges
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    # Apply the cleaning function to the 'comment' column
    df['comment'] = df['comment'].apply(clean_text)
    
    # Drop rows that became completely empty after cleaning
    df = df[df['comment'] != ""]
    
    return df

# --- Local Test ---
if __name__ == "__main__":
    # Test data mimicking messy API results
    raw_data = pd.DataFrame({
        "platform": ["YouTube", "YouTube", "YouTube"],
        "date": ["2026-10-12", "2026-10-12", "2026-10-12"],
        "comment": [
            "Check out my channel at http://example.com for more info! Email me at test@test.com",
            "This dishwasher is amazing!! So quiet. Call me at 123-456-7890.",
            "This dishwasher is amazing!! So quiet." # Duplicate test
        ]
    })
    
    print("--- BEFORE CLEANING ---")
    print(raw_data)
    
    cleaned_data = clean_comments_dataframe(raw_data)
    
    print("\n--- AFTER CLEANING ---")
    print(cleaned_data)
