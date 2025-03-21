import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load the engineered dataset
df = pd.read_csv('data/steam_games_engineered.csv')

# Define features (excluding non-numeric and target columns)
X = df.drop(columns=['AppID', 'Name', 'Estimated_Owners', 'Review_Score', 'Developer', 'Publisher', 'Price_Category'])

# Define target variable
y = df['Review_Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Save the model
joblib.dump(model, 'outputs/review_score_predictor.pkl')
print("Model saved to 'outputs/review_score_predictor.pkl'")
