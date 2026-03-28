import yfinance as yf
import pandas as pd

print("Downloading S&P 500 data 2016-2021...")
sp500 = yf.download('^GSPC', start='2016-01-01', end='2021-01-20')
sp500.to_csv('sp500.csv')
print("Done! Saved to sp500.csv")
print(sp500.head())