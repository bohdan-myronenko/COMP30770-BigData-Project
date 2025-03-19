import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' (non-interactive)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv('data/steam_games_cleaned.csv')

# Distribution of Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('Distribution of Game Prices')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.savefig('outputs/price_distribution.png')  # Save the plot
plt.close()  # Close the plot to free up memory

# Correlation Between Price and Review Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='review_score', data=df)
plt.title('Price vs Review Score')
plt.xlabel('Price (USD)')
plt.ylabel('Review Score (%)')
plt.savefig('outputs/price_vs_review_score.png')  # Save the plot
plt.close()  # Close the plot to free up memory

# Impact of Discounts (if discount data is available)
if 'discount_percentage' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='discount_percentage', y='review_score', data=df)
    plt.title('Discount Percentage vs Review Score')
    plt.xlabel('Discount Percentage')
    plt.ylabel('Review Score (%)')
    plt.savefig('outputs/discount_vs_review_score.png')  # Save the plot
    plt.close()  # Close the plot to free up memory
else:
    print("No discount data available.")