# AIM350 Term Group Project

This project collects product comments, cleans them, classifies the comments as
positive, neutral, or negative, and groups negative comments by common wording.
In the training dataset, comments labeled `none` are set aside instead of
being used as a fourth sentiment. The interface is made with Streamlit.
New searches receive one of the three sentiment predictions; the app does
not automatically identify unrelated comments as `none`.

## Main files

- `app.py`: Streamlit page and complete project flow
- `youtube_fetch.py`: searches for videos and collects their comments
- `prepare_dataset.py`: creates the sample for manual labeling
- `preprocessing.py`: cleans the comments
- `model.py`: trains and tests Logistic Regression and Multinomial Naive Bayes
- `sentiment_predictor.py`: applies the saved model to new comments
- `topic_clustering.py`: groups similar negative comments
- `evaluate_new_test.py`: checks the saved model on a separate labeled test
- `data/comments_500.csv`: labeled project dataset
- `reports/`: model scores, confusion matrix, and test predictions

## Setup

Clone the repository and open its folder first:

```bash
git clone https://github.com/Rmurrell223/AIM350-Term-Group-Project.git
cd AIM350-Term-Group-Project
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to a new file named `.env` and add the YouTube API key
after `YOUTUBE_API_KEY=`. The key is needed for searches, not for training
with the included CSV.

## Train and test the models

```bash
python model.py
```

The file contains 500 reviewed comments. The 121 comments labeled `none` are
set aside. The remaining 379 comments are used in this order:

1. 303 comments train both models and 76 comments test them.
2. TF-IDF is fitted only on the training comments, then transforms the test comments.
3. Macro F1 selects the model that handled the three sentiments more evenly.
4. The program saves the accuracy, F1 scores, and confusion matrix.

The split is 80/20, stratified by sentiment, with `random_state=42`.
There is no separate validation set; both models are compared on the same
76 test comments. Those comments are not used to fit TF-IDF or the classifiers.

Outputs:

- `models/sentiment_pipeline.joblib`: fitted vectorizer and selected classifier
- `reports/model_comparison.csv`: accuracy, macro F1 and weighted F1 for both models
- `reports/confusion_matrix.csv`: the selected model's classification counts
- `reports/test_predictions.csv`: actual and predicted labels for the 76 test comments

Current test result:

- Logistic Regression: 63.16% accuracy, 0.5210 macro F1 and 0.6204 weighted F1
- Multinomial Naive Bayes: 55.26% accuracy, 0.2373 macro F1 and 0.3934 weighted F1
- Selected model: Logistic Regression

The result is below the proposal targets of 75% accuracy and 0.70 F1.
The saved confusion matrix shows that only 2 of the 12 neutral test comments
were classified correctly. These results come from the small appliance dataset
and do not establish the same performance for other products.

## Open the Streamlit app

```bash
streamlit run app.py
```

Open **Model performance test** to see the model comparison and confusion
matrix. The regular analysis page shows the classified comments and the common
word groups found in negative comments.

Enter a brand and product, then select **Run Analysis**. The app collects up
to 10 comments from each of up to 10 videos, cleans the text and applies the
saved model. KMeans groups only the negative predictions into up to three
groups. The displayed keywords help review similar wording; they do not
confirm a product defect. No Silhouette score is currently reported.

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

These commands overwrite `raw_comments.csv`, `videos_used.csv` and
`data/comments_500.csv`. Keep a copy of the current labeled dataset first.
The second command creates an unlabeled sample, so its `sentiment` column
must be filled in before training again. Do not run these commands just to
reproduce the included model results.
