# AIM350 Term Group Project

This project collects product comments, cleans them, classifies the comments as
positive, neutral, or negative, and groups negative comments by common wording.
Comments labeled `none` do not give a clear product opinion, so they are set
aside instead of being used as a fourth sentiment. The interface is made with
Streamlit.

## Main files

- `app.py`: Streamlit page and complete project flow
- `preprocessing.py`: cleans the comments
- `model.py`: trains, validates, and tests two sentiment models
- `sentiment_predictor.py`: applies the saved model to new comments
- `topic_clustering.py`: groups similar negative comments
- `evaluate_new_test.py`: checks the saved model on a separate labeled test
- `data/comments_500.csv`: labeled project dataset
- `reports/`: model scores, confusion matrix, and test predictions

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add the YouTube API key after `YOUTUBE_API_KEY=` in the new `.env` file.

## Train and test the models

```bash
python model.py
```

The file contains 500 reviewed comments. The 121 comments labeled `none` are
set aside. The remaining 379 comments are used in this order:

1. 80% trains Logistic Regression and Naive Bayes.
2. The remaining 20% tests both models on comments they did not train with.
3. Macro F1 selects the model that handled the three sentiments more evenly.
4. The program saves the accuracy, F1 scores, and confusion matrix.

The saved model goes to `models/sentiment_pipeline.joblib`. The test tables go
to `reports/`.

Current test result:

- Logistic Regression: 63.16% accuracy and 0.521 macro F1
- Naive Bayes: 55.26% accuracy and 0.2373 macro F1
- Selected model: Logistic Regression

The result is below the proposed 70% accuracy goal. The saved confusion matrix
shows that neutral comments are the most difficult group for this dataset.

## Open the Streamlit app

```bash
streamlit run app.py
```

Open **Model performance test** to see the model comparison and confusion
matrix. The regular analysis page shows the classified comments and the common
word groups found in negative comments.

## Run a separate test with new comments

1. Run an analysis in the app.
2. Select **Download a blind sample for a separate test**. The app randomly
   selects up to 50 comments and does not show the model's answers in this file.
3. Fill in `correct_label` for every downloaded row with `positive`, `neutral`,
   `negative`, or `none`. Keep every row, including the model's mistakes.
4. Save the completed file as `data/live_test_round2.csv`.
5. Run:

```bash
python evaluate_new_test.py data/live_test_round2.csv
```

This creates a new summary, confusion matrix, and row-by-row comparison in
`reports/`. Rows marked `none` are counted and set aside before the three
sentiments are scored. It does not add the new comments to the training data.
The program also stops if labels are missing or if a test comment already
exists in the training dataset.

Daniel's first manual review contains only mistakes that he noticed. It is
useful for describing errors but cannot be used as the overall accuracy rate.
It can be inspected with:

```bash
python evaluate_new_test.py data/model_error_review_round1.csv
```

## Optional: collect a new dataset

```bash
python youtube_fetch.py
python prepare_dataset.py 500
```

The second command creates a new unlabeled `data/comments_500.csv`. The labels
must be filled in before training again.
