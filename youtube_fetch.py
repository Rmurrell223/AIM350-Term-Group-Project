import os

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()  # loads YOUTUBE_API_KEY from your private .env file


def get_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not found - check your .env file")
    return build("youtube", "v3", developerKey=api_key)


def search_videos(query, max_results=10):
    """Search YouTube; return videos with their comment counts."""
    youtube = get_client()
    search = youtube.search().list(
        q=query, part="id", maxResults=max_results, type="video"
    ).execute()
    ids = [item["id"]["videoId"] for item in search.get("items", [])]
    if not ids:
        return []
    details = youtube.videos().list(
        part="snippet,statistics", id=",".join(ids)
    ).execute()
    return [
        {
            "video_id": item["id"],
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "comment_count": int(item["statistics"].get("commentCount", 0)),
        }
        for item in details.get("items", [])
    ]


def fetch_comments(video_id, max_comments=500):
    """Download top-level comments from ONE video into a table."""
    youtube = get_client()
    rows, page_token = [], None
    try:
        while len(rows) < max_comments:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=page_token,
                textFormat="plainText",
            ).execute()
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                rows.append({
                    "platform": "YouTube",
                    "date": snippet["publishedAt"][:10],
                    "comment": snippet["textDisplay"],
                    "video_id": video_id,
                })
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    except HttpError as e:
        print(f"Could not get comments (comments may be disabled): {e}")
    return pd.DataFrame(rows[:max_comments])


TOP_N = 5  # most-commented videos to use PER CATEGORY

CATEGORIES = {


    
    "Home appliance": [
        "dishwasher review",
        "washing machine review",
        "refrigerator review",
        "air fryer review",
        "GE Hotpoint dishwasher problems",
    ],
}

if __name__ == "__main__":
    frames, used = [], []
    for category, queries in CATEGORIES.items():
        found = {}
        for q in queries:
            for v in search_videos(q):
                found[v["video_id"]] = v
        ranked = sorted(found.values(), key=lambda v: v["comment_count"], reverse=True)
        print(f"\n=== {category} ===")
        for i, v in enumerate(ranked):
            print(f"{i}: {v['comment_count']:>6} comments | {v['title']} ({v['channel']})")

        for v in ranked[:TOP_N]:
            print(f"Downloading comments: {v['title']}")
            frame = fetch_comments(v["video_id"])
            frame["video_title"] = v["title"]
            frame["category"] = category
            frames.append(frame)
            used.append({**v, "category": category})

    df = pd.concat(frames, ignore_index=True)
    print(f"\nGot {len(df)} comments from {len(used)} videos")
    print(df.groupby("category").size())
    df.to_csv("raw_comments.csv", index=False)
    pd.DataFrame(used).to_csv("videos_used.csv", index=False)
    