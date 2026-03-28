import pandas as pd

# Load the dataset - check what your CSV file is actually called first
df = pd.read_csv('tweets.csv')  

# First look at the data
print(df.shape)        # how many rows and columns
print(df.head())       # first 5 rows
print(df.columns)      # column names