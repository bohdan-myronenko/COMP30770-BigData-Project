import pandas as pd

# Load the combined dataset
df = pd.read_csv('data/steam_games_finalized.csv')

# Drop rows with negative discount percentages
negative_discounts = df[df['Discount_Percentage'] < 0]
if not negative_discounts.empty:
    print(f"Removing {len(negative_discounts)} entries with negative discounts.")
    df = df[df['Discount_Percentage'] >= 0]

# Merge duplicates based on AppID
aggregation_functions = {
    'Name': 'first',
    'Original_Price': 'mean',
    'Discount_Price': 'mean',
    'Discount_Percentage': 'mean',
    'Estimated_Owners': 'first',
    'Positive': 'sum',
    'Negative': 'sum',
    'Review_Score': 'mean',
    'Recent_Reviews': lambda x: ' | '.join(x.dropna().astype(str).unique()),
    'All_Reviews': lambda x: ' | '.join(x.dropna().astype(str).unique()),
    'Genres': lambda x: ', '.join(x.dropna().astype(str).unique()),
    'Categories': lambda x: ', '.join(x.dropna().astype(str).unique()),
    'Tags': lambda x: ', '.join(x.dropna().astype(str).unique()),
    'Achievements': 'max',
    'Developer': lambda x: ', '.join(x.dropna().astype(str).unique()),
    'Publisher': lambda x: ', '.join(x.dropna().astype(str).unique())
}

# Perform aggregation
df_cleaned = df.groupby('AppID', as_index=False).agg(aggregation_functions)

# Recompute Review Score accurately after summing Positive and Negative reviews
df_cleaned['Review_Score'] = df_cleaned.apply(
    lambda row: row['Positive'] / (row['Positive'] + row['Negative']) * 100
    if (row['Positive'] + row['Negative']) > 0 else 0, axis=1)

# Save the cleaned and deduplicated dataset
df_cleaned.to_csv('data/steam_games_finalized_clean.csv', index=False)

print(f"Deduplication completed. Final dataset saved to 'data/steam_games_finalized_clean.csv'")
