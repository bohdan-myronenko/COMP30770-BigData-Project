import pandas as pd

# Load the cleaned dataset
df = pd.read_csv('data/steam_games_cleaned.csv')

# Categorize Games by Price Range
df['price_category'] = pd.cut(df['Price'], bins=[-1, 0, 10, 30, 100, float('inf')], 
                             labels=['Free', 'Budget', 'Mid-Range', 'Premium', 'Luxury'])

# One-Hot Encode Genres and Categories (if available)
if 'Genres' in df.columns:
    df = pd.get_dummies(df, columns=['Genres'], prefix=['Genre'])
if 'Categories' in df.columns:
    df = pd.get_dummies(df, columns=['Categories'], prefix=['Category'])

# Save the engineered dataset
df.to_csv('data/steam_games_engineered.csv', index=False)
print("Engineered dataset saved to 'data/steam_games_engineered.csv'")