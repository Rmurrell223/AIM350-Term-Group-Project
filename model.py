import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def train_and_evaluate_model(df):
    """
    Trains a Scikit-learn sentiment classifier and tests it against 
    the project's 75% accuracy and 0.70 F1-score benchmark.
    """
    if len(df) < 5:
        return None, "Not enough data to train model yet. Need at least a few samples."

    # For demonstration/training, if you don't have pre-labeled data yet, 
    # we can map basic keywords or use a small sample dataset.
    # In production, your team will label the pulled comments.
    
    X = df['comment']
    y = df['sentiment'] # Assumes a 'sentiment' column exists ('Positive', 'Negative', 'Neutral')

    # Convert text to numeric features using TF-IDF vectorizer (as proposed)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)

    # 80% training, 20% testing split (matching your proposal)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # Train a Logistic Regression classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Predict on test data
    y_pred = classifier.predict(X_test)

    # Calculate benchmarks
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results = {
        "accuracy": round(acc * 100, 2),
        "f1_score": round(f1, 2),
        "vectorizer": vectorizer,
        "model": classifier
    }

    return results
