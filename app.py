import pandas as pd
import os
from src.preprocessing import preprocess_data


raw_data = pd.read_csv('data/raw/IMDB Dataset.csv')

print(raw_data.head())
print(raw_data['sentiment'].value_counts())

preprocessed_reviews = raw_data['review'].apply(preprocess_data)
preprocessed_data = pd.DataFrame({'review': preprocessed_reviews, 'sentiment': raw_data['sentiment']})

preprocessed_data.to_csv('data/processed/IMDB Processed Data.csv', index=False)

