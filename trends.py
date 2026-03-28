import pandas as pd
from pytrends.request import TrendReq
import time

# Connect to Google Trends
pytrends = TrendReq(hl='en-US', tz=360)

# Keywords to search (must match what Google Trends understands)
keywords = ['China trade war', 'Fake News', 'Amazon stock', 'Russia investigation', 'Mexico wall', 'NATO']

# Date range covering Trump's first term
timeframe = '2016-01-01 2021-01-20'

all_trends = {}

for keyword in keywords:
    print(f"Fetching Google Trends for: {keyword}...")
    try:
        pytrends.build_payload([keyword], timeframe=timeframe)
        data = pytrends.interest_over_time()
        
        if not data.empty:
            all_trends[keyword] = data[keyword]
            print(f"  ✅ Got {len(data)} data points")
        else:
            print(f"  ⚠️ No data returned")
            
        # Be polite to Google's API - don't hammer it
        time.sleep(2)
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Combine all trends into one dataframe
trends_df = pd.DataFrame(all_trends)
trends_df.index = pd.to_datetime(trends_df.index)
trends_df.to_csv('trends_data.csv')
print("\nDone! Saved to trends_data.csv")
print(trends_df.head())