# Attention Shockwave Model
A behavioural signal model that correlates 56,571 presidential tweets with Google search behaviour — 2016 to 2021.
## What it does
- Correlates Trump tweets with Google Trends search spikes across 6 topics
- Invents the Attention Elasticity Index (AEI) — weights learned via Ridge regression
- Backtests a volatility-targeted trading strategy against the S&P 500
- Predicts search spikes and attention half-life for any input tweet
## Data
Download the Trump tweets dataset from Kaggle:
https://www.kaggle.com/datasets/codebreaker619/donald-trump-tweets-dataset
Save as tweets.csv in the project folder, then run the scripts in order.
## Setup
pip3 install dash plotly pandas numpy textblob scikit-learn yfinance pytrends
## Run in order
1. python3 anylasis.py
2. python3 trends.py
3. python3 get_market.py
4. python3 app.py
Then open http://127.0.0.1:8050
