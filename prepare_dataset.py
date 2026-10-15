import os
import sys

import pandas as pd

from preprocessing import clean_comments_dataframe


TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 500

df = pd.read_csv("raw_comments.csv")
print(f"Start: {len(df)} comments")

df = clean_comments_dataframe(df)
df = df[df["comment"].str.len() >= 15]
print(f"After cleaning: {len(df)} comments")

# Pick about the same number from each video.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
per_video = TOTAL // df["video_id"].nunique()
df["rank_in_video"] = df.groupby("video_id").cumcount()
picked = df[df["rank_in_video"] < per_video]
extra = df[df["rank_in_video"] >= per_video].head(TOTAL - len(picked))
sample = pd.concat([picked, extra]).sample(frac=1, random_state=1).reset_index(drop=True)
sample = sample.head(TOTAL)

sample["sentiment"] = ""
cols = ["platform", "date", "comment", "sentiment", "category", "video_id", "video_title"]

os.makedirs("data", exist_ok=True)
sample[cols].to_csv("data/comments_500.csv", index=False, encoding="utf-8-sig")

print(f"Saved data/comments_500.csv with {len(sample)} comments")
print(sample.groupby("video_title").size())
