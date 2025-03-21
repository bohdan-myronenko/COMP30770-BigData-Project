import pandas as pd
import re

# Load Steam dataset from HuggingFace
from datasets import load_dataset

# Download dataset
dataset = load_dataset("FronkonGames/steam-games-dataset")
df = pd.DataFrame(dataset['train'])

# Select relevant columns for analysis
df = df[['AppID', 'Name', 'Release date', 'Estimated owners', 'Price',
         'DLC count', 'Positive', 'Negative', 'Average playtime forever',
         'Median playtime forever', 'Developers', 'Publishers', 'Categories', 'Genres', 'Tags']]

# Rename columns consistently
df.rename(columns={
    'Release date': 'Release_Date',
    'Estimated owners': 'Estimated_Owners',
    'Developers': 'Developer',
    'Publishers': 'Publisher',
    'DLC count': 'DLC_Count',
    'Average playtime forever': 'Avg_Playtime',
    'Median playtime forever': 'Median_Playtime'
}, inplace=True)

# Compute review score safely
df['Review_Score'] = df.apply(
    lambda row: row['Positive'] / (row['Positive'] + row['Negative']) * 100
    if (row['Positive'] + row['Negative']) > 0 else 0, axis=1)

# Standardize date format
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce').dt.date

# Categorize games by price range
df['Price_Category'] = pd.cut(df['Price'], bins=[-1, 0, 10, 30, 100, float('inf')], 
                             labels=['Free', 'Budget', 'Mid-Range', 'Premium', 'Luxury'])

# One-hot encode Genres and Categories (for ML usage)
df = pd.get_dummies(df, columns=['Genres', 'Categories'], prefix=['Genre', 'Category'])

# Remove any duplicates
aggregation_functions = {
    'Name': 'first',
    'Price': 'mean',
    'DLC_Count': 'mean',
    'Estimated_Owners': 'first',
    'Positive': 'sum',
    'Negative': 'sum',
    'Avg_Playtime': 'mean',
    'Median_Playtime': 'mean',
    'Developer': lambda x: ', '.join(x.dropna().unique()),
    'Publisher': lambda x: ', '.join(x.dropna().unique()),
    'Tags': lambda x: ', '.join(x.dropna().unique()),
    'Review_Score': 'mean',
    'Price_Category': 'first'
}

# Aggregate duplicates based on AppID
df_cleaned = df.groupby('AppID', as_index=False).agg(aggregation_functions)

# Recompute accurate Review Score
df_cleaned['Review_Score'] = df_cleaned.apply(
    lambda row: row['Positive'] / (row['Positive'] + row['Negative']) * 100
    if (row['Positive'] + row['Negative']) > 0 else 0, axis=1)

# Final column order for clarity
final_columns = ['AppID', 'Name', 'Price', 'Price_Category', 'DLC_Count',
                 'Estimated_Owners', 'Positive', 'Negative', 'Review_Score',
                 'Avg_Playtime', 'Median_Playtime', 'Developer', 'Publisher', 'Tags'] + \
                [col for col in df_cleaned.columns if col.startswith(('Genre_', 'Category_'))]

# Save the finalized cleaned and deduplicated dataset
df_cleaned = df_cleaned[final_columns]
df_cleaned.to_csv('data/steam_games_final_cleaned.csv', index=False)

print("Finalized and cleaned dataset saved to 'data/steam_games_final_cleaned.csv'")
