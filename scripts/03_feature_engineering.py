import pandas as pd

# Load the cleaned dataset
df = pd.read_csv('data/steam_games_final_cleaned.csv')

# Categorize Games by Price Range
df['Price_Category'] = pd.cut(df['Price'], bins=[-1, 0, 10, 30, 100, float('inf')], 
                              labels=['Free', 'Budget', 'Mid-Range', 'Premium', 'Luxury'])

# Split Tags into separate columns for one-hot encoding
tags_expanded = df['Tags'].str.get_dummies(sep=',')
tags_expanded.columns = ['Tag_' + tag.strip().replace(' ', '_') for tag in tags_expanded.columns]

# Concatenate original dataframe with tags
engineered_df = pd.concat([df, tags_expanded], axis=1)

# Drop original Tags column
engineered_df.drop(columns=['Tags'], inplace=True)

# Save the engineered dataset
engineered_df.to_csv('data/steam_games_engineered.csv', index=False)
print("Engineered dataset saved to 'data/steam_games_engineered.csv'")
