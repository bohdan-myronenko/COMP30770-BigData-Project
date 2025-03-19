import os
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate with Kaggle API
api = KaggleApi()
api.authenticate()

# Kaggle dataset details
dataset_owner_slug = 'trolukovich'
dataset_slug = 'steam-games-complete-dataset'
data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')

# Ensure the data directory exists
os.makedirs(data_folder, exist_ok=True)

# Download dataset ZIP to your data folder
api.dataset_download_files(
    dataset=f"{dataset_owner_slug}/{dataset_slug}",
    path=data_folder,
    unzip=False
)

# Path to downloaded ZIP
zip_file_path = os.path.join(data_folder, f"{dataset_slug}.zip")

# Extract dataset into a temporary subfolder first
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    temp_extract_path = os.path.join(data_folder, "temp_extracted")
    zip_ref.extractall(temp_extract_path)

# Rename the extracted CSV to steam_games_kaggle.csv
# Assumes there's exactly one CSV file in the extracted folder
for filename in os.listdir(temp_extract_path):
    if filename.endswith('.csv'):
        src = os.path.join(temp_extract_path, filename)
        dst = os.path.join(data_folder, 'steam_games_kaggle.csv')
        shutil.move(src, dst)
        print(f"Renamed and moved file to {dst}")

# Clean up temporary extracted folder and zip file
shutil.rmtree(temp_extract_path)
os.remove(zip_file_path)

print(f"Dataset successfully downloaded and saved as 'steam_games_kaggle.csv' in '{data_folder}'")
