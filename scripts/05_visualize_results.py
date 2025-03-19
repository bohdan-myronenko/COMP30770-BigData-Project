import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' (non-interactive)
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Load the engineered dataset
df = pd.read_csv('data/steam_games_engineered.csv')

# Load the trained model
model = joblib.load('outputs/review_score_predictor.pkl')

# Select features and target
X = df[['Price', 'Positive', 'Negative']]
y = df['review_score']

# Make predictions
df['predicted_review_score'] = model.predict(X)

# Plot Actual vs Predicted Review Scores
plt.figure(figsize=(10, 6))
plt.scatter(y, df['predicted_review_score'], alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.title('Actual vs Predicted Review Scores')
plt.xlabel('Actual Review Score (%)')
plt.ylabel('Predicted Review Score (%)')
plt.savefig('outputs/actual_vs_predicted_review_scores.png')  # Save the plot

plt.close()  # Close the plot to free up memory