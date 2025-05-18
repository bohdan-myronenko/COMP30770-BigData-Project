# COMP30770-BigData-Project
# Project Description

This project builds a predictive model for *Steam game review scores*
(defined as the percentage of positive user reviews) using a
comprehensive dataset of Steam game metadata. The goal is to uncover
which factors most influence a game’s reception and to leverage those
features to accurately predict review scores, offering insights for
developers, publishers, and players.

# Dataset Description

The dataset originates from a public Kaggle dataset by FronkonGames,
accessed via the Hugging Face `datasets` library. It contains thousands
of game entries with both numerical and categorical attributes:

- **Numerical features:** Price, DLC count, positive and negative review
  counts, average and median playtime, estimated owners.

- **Categorical features:** Genres, categories, developers, publishers,
  user-defined tags.

The target variable, *Review Score*, was computed as:
$$\text{Review Score} = \frac{\text{Positive}}{\text{Positive} + \text{Negative}} \times 100$$
All categorical fields (genres, categories, tags) were one-hot encoded.
Cleaned and engineered datasets are saved under `data/` as
`steam_games_final_cleaned.csv` and `steam_games_engineered.csv`.

# Project Objectives

- Develop a model that accurately predicts game review scores from
  metadata.

- Help developers identify influential design and pricing factors.

- Provide market insights for publishers on pricing and feature
  strategies.

- Assist players in discovering games aligned with their preferences.

# Project Structure

## 01_load_and_preprocess.py

- Load raw dataset via Hugging Face `datasets`.

- Select and rename relevant columns (AppID, Name, Release Date, Owners,
  Price, DLC count, reviews, playtime, developers, publishers,
  categories, genres, tags).

- Compute *Review Score*.

- Standardize dates and bucket prices into categories (Free, Budget,
  Mid-Range, Premium, Luxury).

- One-hot encode genres and categories.

- Aggregate duplicates and save cleaned data to
  `data/steam_games_final_cleaned.csv`.

## 02_exploratory_data_analysis.py

- Generate visualizations: price distribution, review score
  distribution, price vs. review score, review score by price category,
  correlation heatmap, playtime vs. review score.

- Save plots (e.g. `price_distribution.png`, `correlation_matrix.png`)
  for data insights.

## 03_feature_engineering.py

- Further price bucketing via `pandas.cut`.

- Expand tags into binary columns with `get_dummies`.

- Concatenate new features and save to
  `data/steam_games_engineered.csv`.

## 04_train_model.py

- Split engineered data: 80% train, 20% test.

- Train a Random Forest regressor with hyperparameter tuning via
  `GridSearchCV` (grid over `n_estimators`, `max_depth`,
  `min_samples_split`).

- Use `n_jobs=-1` for parallel processing.

- Evaluate on test set (MAE, MSE, R<sup>2</sup>) and save model to
  `outputs/final_review_score_predictor.pkl`.

## 05_visualize_results.py

- Plot Actual vs. Predicted review scores and residuals.

- Compute and display feature importances.

- Save evaluation plots to `outputs/`.

## 06_deploy_model.py

- Deploy the trained model as a REST API using Flask.

- Expose `/predict` endpoint for JSON input and return predicted scores.

- Example request:

      curl -X POST -H "Content-Type: application/json" \
        -d '[{
              "Price": 19.99,
              "DLC_Count": 3,
              "Positive": 5000,
              "Negative": 200,
              "Avg_Playtime": 1200,
              "Median_Playtime": 800,
              "Genre_Action": 1,
              "Genre_Adventure": 0,
              "Category_Single-player": 1,
              "Category_Multi-player": 0,
              "Tag_Indie": 1,
              "Tag_Casual": 0
            }]' \
        http://127.0.0.1:5000/predict

# Modeling Approach

A Random Forest regressor was chosen for its robustness with mixed data
types and interpretability. Hyperparameters were optimized via grid
search with 5-fold cross-validation. The final model demonstrated high
predictive accuracy (R<sup>2</sup> ≈ 1.0) and was serialized for
deployment.

# Optimization Techniques

Grid search is computationally intensive. Initially intended to leverage
Apache Spark MLlib for distributed tuning, the project instead used
Joblib-based parallelism (`n_jobs=-1`) to fully utilize an 8-core CPU,
significantly reducing runtime.

# Deployment

The model is served via a Flask API. Run `06_deploy_model.py` to start
the server (default: <http://127.0.0.1:5000>). Send POST requests with
JSON-formatted game features to `/predict` for real‐time score
predictions.

# How to Run the Project

1.  Clone the repository: `git clone ...`

2.  Install Python 3.12 and dependencies:

        pip install pandas numpy scikit-learn flask datasets matplotlib

3.  Run preprocessing: `python 01_load_and_preprocess.py`

4.  (Optional) EDA: `python 02_exploratory_data_analysis.py`

5.  Feature engineering: `python 03_feature_engineering.py`

6.  Train model: `python 04_train_model.py`

7.  Visualize results: `python 05_visualize_results.py`

8.  Deploy API: `python 06_deploy_model.py`

# Future Work

- **Distributed Training:** Integrate Apache Spark MLlib for
  cluster-based hyperparameter tuning.

- **Alternative Frameworks:** Evaluate Dask or Ray for parallel
  computing.

- **GPU Acceleration:** Use RAPIDS cuML or similar for GPU‐accelerated
  training.

- **Model Enhancements:** Experiment with gradient boosting or neural
  network models.

- **Feature Enrichment:** Incorporate temporal or external data sources;
  refine tag processing with TF–IDF or clustering.

# Authors and Acknowledgements

**Authors:** Oleksii Shvets, Bohdan Myronenko.  
**Acknowledgements:** The Steam Games Dataset (FronkonGames), Hugging
Face `datasets`, Pandas, NumPy, Scikit‐learn, Matplotlib, Seaborn,
Flask, and the University College Dublin COMP30770 module instructors.
