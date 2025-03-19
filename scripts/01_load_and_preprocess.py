import pandas as pd


# Load the original dataset
df_fronkon = pd.read_csv('data/steam_games.csv')

# Keep necessary columns only
df_fronkon = df_fronkon[['AppID', 'Name', 'Release date', 'Estimated owners', 'Price',
                         'Positive', 'Negative', 'Genres', 'Categories', 'Tags', 'Achievements',
                         'Developers', 'Publishers']]

# Rename columns for uniformity
df_fronkon.rename(columns={
    'Release date': 'Release_Date',
    'Estimated owners': 'Estimated_Owners',
    'Developers': 'Developer',
    'Publishers': 'Publisher'
}, inplace=True)

# Calculate review score
df_fronkon['Review_Score'] = df_fronkon.apply(
    lambda row: row['Positive'] / (row['Positive'] + row['Negative']) * 100
    if (row['Positive'] + row['Negative']) > 0 else 0, axis=1)

# Standardize Release_Date to ISO format
df_fronkon['Release_Date'] = pd.to_datetime(df_fronkon['Release_Date'], errors='coerce').dt.date


df_kaggle = pd.read_csv('data/steam_games_kaggle.csv')

# Extract AppID from URL
df_kaggle['AppID'] = df_kaggle['url'].str.extract(r'/app/(\d+)/')

# Check rows with NaN AppID
invalid_urls = df_kaggle[df_kaggle['AppID'].isnull()]
if not invalid_urls.empty:
    print(f"Found {len(invalid_urls)} invalid URLs:")
    print(invalid_urls['url'])
    # Drop or handle these rows explicitly
    df_kaggle = df_kaggle.dropna(subset=['AppID'])

# Now safely convert AppID to integer
df_kaggle['AppID'] = df_kaggle['AppID'].astype(int)


# Handle 'Free' and other non-numeric cases explicitly
df_kaggle['Original_Price'] = (
    df_kaggle['original_price']
    .replace('Free', '0', regex=False)  # Replace 'Free' with '0'
    .replace('[\$,]', '', regex=True)   # Remove dollar signs and commas
)

# Convert safely to float
df_kaggle['Original_Price'] = pd.to_numeric(df_kaggle['Original_Price'], errors='coerce').fillna(0.0)

# Repeat similar logic for 'discount_price'
df_kaggle['Discount_Price'] = (
    df_kaggle['discount_price']
    .replace('Free', '0', regex=False)
    .replace('[\$,]', '', regex=True)
)

df_kaggle['Discount_Price'] = pd.to_numeric(df_kaggle['Discount_Price'], errors='coerce').fillna(0.0)

df_kaggle['Discount_Percentage'] = (
    (df_kaggle['Original_Price'] - df_kaggle['Discount_Price']) /
    df_kaggle['Original_Price'] * 100
).round(2)

# Handle potential division by zero
df_kaggle['Discount_Percentage'].fillna(0, inplace=True)

# Keep necessary columns
df_kaggle = df_kaggle[['AppID', 'name', 'release_date', 'Original_Price', 'Discount_Price',
                       'Discount_Percentage', 'recent_reviews', 'all_reviews', 'popular_tags',
                       'achievements', 'genre', 'developer', 'publisher']]

# Rename columns for uniformity
df_kaggle.rename(columns={
    'name': 'Name',
    'release_date': 'Release_Date',
    'recent_reviews': 'Recent_Reviews',
    'all_reviews': 'All_Reviews',
    'popular_tags': 'Tags',
    'achievements': 'Achievements',
    'genre': 'Genres',
    'developer': 'Developer',
    'publisher': 'Publisher'
}, inplace=True)

# Standardize Release_Date to ISO format
df_kaggle['Release_Date'] = pd.to_datetime(df_kaggle['Release_Date'], errors='coerce').dt.date

# Merge datasets on AppID (use outer join to keep all records)
df_merged = pd.merge(df_fronkon, df_kaggle, on=['AppID', 'Name', 'Release_Date', 'Developer', 'Publisher',
                                                'Genres', 'Tags', 'Achievements'], how='outer')

# Reorder columns for clarity
final_columns_order = ['AppID', 'Name', 'Original_Price', 'Discount_Price',
                       'Discount_Percentage', 'Estimated_Owners', 'Positive', 'Negative', 'Review_Score',
                       'Recent_Reviews', 'All_Reviews', 'Genres', 'Categories', 'Tags', 'Achievements',
                       'Developer', 'Publisher']

df_final = df_merged[final_columns_order]

# Save finalized dataset
df_final.to_csv('data/steam_games_finalized.csv', index=False)
