import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' (non-interactive)
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import seaborn as sns

# Load the engineered dataset
df = pd.read_csv('data/steam_games_engineered.csv')

# Load the trained model
model = joblib.load('outputs/review_score_predictor.pkl')

# Select all numeric features except the target
target = 'Review_Score'
X = df.drop(columns=['AppID', 'Name', 'Estimated_Owners', 'Developer', 'Publisher', 'Price_Category', target])
y = df[target]

# Make predictions
df['Predicted_Review_Score'] = model.predict(X)

# Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y, df['Predicted_Review_Score'], alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Actual vs Predicted Review Scores')
plt.xlabel('Actual Review Score (%)')
plt.ylabel('Predicted Review Score (%)')
plt.savefig('outputs/actual_vs_predicted_review_scores.png')
plt.close()

# Residual Plot
plt.figure(figsize=(10, 6))
residuals = y - df['Predicted_Review_Score']
plt.scatter(df['Predicted_Review_Score'], residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Review Score (%)')
plt.ylabel('Residuals')
plt.savefig('outputs/residual_plot.png')
plt.close()

# DLC Impact Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x=pd.cut(df['DLC_Count'], bins=[-1, 0, 5, 20, 50, float('inf')], 
                    labels=['No DLC', 'Few DLCs', 'Moderate', 'Many', 'Tons']),
            y='Review_Score', data=df)
plt.title('DLC Count Impact on Review Score')
plt.xlabel('DLC Group')
plt.ylabel('Review Score (%)')
plt.savefig('outputs/dlc_impact_on_reviews.png')
plt.close()

# Playtime vs Review Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Avg_Playtime', y='Review_Score', data=df)
plt.title('Average Playtime vs Review Score')
plt.xlabel('Average Playtime (minutes)')
plt.ylabel('Review Score (%)')
plt.savefig('outputs/playtime_vs_review_score.png')
plt.close()

print("Evaluation plots generated and saved to 'outputs/' directory.")
