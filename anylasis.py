import pandas as pd
from textblob import TextBlob

# Load data
df = pd.read_csv('tweets.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Remove retweets - we only want Trump's own words
df = df[df['isRetweet'] == 'f']

# Extract just the date (no time)
df['date_only'] = df['date'].dt.date

# Get sentiment score for each tweet (-1 = very negative, +1 = very positive)
def get_sentiment(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0

print("Calculating sentiment for 40,000+ tweets, this takes ~60 seconds...")
df['sentiment'] = df['text'].apply(get_sentiment)

# Define keywords to track
keywords = ['China', 'Fake News', 'Amazon', 'Russia', 'Mexico', 'NATO']

# For each keyword, find tweets containing it
for keyword in keywords:
    mask = df['text'].str.contains(keyword, case=False, na=False)
    count = mask.sum()
    print(f"{keyword}: {count} tweets")

print("\nDone! Saving cleaned data...")
df.to_csv('tweets_clean.csv', index=False)
print("Saved to tweets_clean.csv")