import pandas as pd

# Load the dataset
df = pd.read_csv('data/steam_games.csv')

# Display basic info about the dataset
print("Dataset Info:")
print(df.info())

# Handle missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing values
df['Price'].fillna(0, inplace=True)  # Free games
df['Positive'].fillna(0, inplace=True)
df['Negative'].fillna(0, inplace=True)

# Convert data types
df['Price'] = df['price'].astype(float)
df['Positive'] = df['Positive'].astype(int)
df['Negative'] = df['Negative'].astype(int)

# Calculate review score
df['review_score'] = df['Positive'] / (df['Positive'] + df['Negative']) * 100
df['review_score'].fillna(0, inplace=True)  # Handle cases where positive + negative = 0

# Save the cleaned dataset
df.to_csv('data/steam_games_cleaned.csv', index=False)
print("\nCleaned dataset saved to 'data/steam_games_cleaned.csv'")