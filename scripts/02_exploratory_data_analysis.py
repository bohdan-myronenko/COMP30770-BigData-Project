import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' (non-interactive)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv('data/steam_games_final_cleaned.csv')

# Distribution of Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('Distribution of Game Prices')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.savefig('outputs/price_distribution.png')
plt.close()

# Correlation Between Price and Review Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='Review_Score', data=df)
plt.title('Price vs Review Score')
plt.xlabel('Price (USD)')
plt.ylabel('Review Score (%)')
plt.savefig('outputs/price_vs_review_score.png')
plt.close()

# Distribution of Review Scores
plt.figure(figsize=(10, 6))
sns.histplot(df['Review_Score'], bins=50, kde=True)
plt.title('Distribution of Review Scores')
plt.xlabel('Review Score (%)')
plt.ylabel('Frequency')
plt.savefig('outputs/review_score_distribution.png')
plt.close()

# Average Review Score by Price Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='Price_Category', y='Review_Score', data=df, order=['Free', 'Budget', 'Mid-Range', 'Premium', 'Luxury'])
plt.title('Average Review Score by Price Category')
plt.xlabel('Price Category')
plt.ylabel('Review Score (%)')
plt.savefig('outputs/review_score_by_price_category.png')
plt.close()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df[['Price', 'DLC_Count', 'Positive', 'Negative', 'Review_Score', 'Avg_Playtime', 'Median_Playtime']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Features')
plt.savefig('outputs/correlation_matrix.png')
plt.close()

# Relationship between Average Playtime and Review Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Avg_Playtime', y='Review_Score', data=df)
plt.title('Average Playtime vs Review Score')
plt.xlabel('Average Playtime (minutes)')
plt.ylabel('Review Score (%)')
plt.savefig('outputs/playtime_vs_review_score.png')
plt.close()

print("Visualizations successfully generated and saved to 'outputs/' directory.")
