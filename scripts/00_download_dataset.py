from datasets import load_dataset

# Load the dataset
dataset = load_dataset("FronkonGames/steam-games-dataset")

# Save the dataset to a CSV file
dataset['train'].to_csv('data/steam_games.csv', index=False)
print("Dataset downloaded and saved to 'data/steam_games.csv'")