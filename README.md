# housing-teller

## Before cloning

- Run `git lfs install` on your console, we're using this to manage our large dataset files exceeding GitHub's file limit, read more [here](https://git-lfs.com/).

## Setting up

1. After cloning the repo, run `git lfs pull` to make sure csv files are pulled.

2. Run `pip install -r requirements.txt` to download all libraries used.

# Team 2: Housing Price Prediction Project

## Overview
This repository contains all code, notebooks, and data for Team 2’s housing price prediction project. We develop machine learning models to predict home sale prices using the Boston and Chicago housing datasets, and provide a Flask web app for interactive prediction.

## Prerequisites
- Python 3.8+
- Git (to clone repository)

## Requirements
- pandas
- numpy
- scikit-learn
- xgboost
- Flask
- joblib
- matplotlib
- lightgbm
- hyperopt
- seaborn
- requests

## Setup
1. **Clone the repo**
  
2. **Install dependencies**

## Data

- `BostonHousing.csv` (Boston dataset from Kaggle)
- `Chicago_Housing.csv` (Chicago dataset from public records)
- `chicago_data_final.csv` – Cleaned and processed Chicago dataset (post-preprocessing).
- `chicago_housing_data.csv` – Combined Chicago housing features and sale prices.
- `chicago_residential_sales_data.csv` – Detailed Chicago sales transaction records.
- `geoid_chicago_residential_sales.csv` – Geographic identifiers (GEOIDs) for sales entries.
- `updt_chicago_residential_sales.csv` – Updated Chicago sales dataset with additional fields and corrections.

## Notebooks
- **preprocess.ipynb**: Data cleaning, missing value handling, encoding, and feature engineering.
- **chicago_housing.ipynb**: Exploratory analysis and model training on Chicago data.
- **Hamza_BuObaid.ipynb**: Individual experiment notebook.

## Scripts

### Boston_model.py
Trains an XGBoost regressor on the Boston dataset:
`
python Boston_model.py
`
This script will:
- Load and clean `datasets/BostonHousing.csv`
- Train-test split and model fitting
- Output MSE and R² scores
- Save the trained model as `xgb_boston_model.pkl`
- Display a feature importance plot

### chicago_model.py
Trains a LightGBM regressor on the Chicago dataset:
`
python chicago_model.py
`
This script will:
- Load `chicago_data_final.csv`
- Train-test split and model fitting
- Output MSE and R² scores
- Save the trained model as `lgbm_chicago_model.pkl`

### app.py
Runs a Flask web app for interactive predictions:
`
python app.py
`
- The app loads `xgb_boston_model.pkl` and `lgbm_chicago_model.pkl`
- Accepts input features via a web form
- Displays predicted sale price
- Default route: http://localhost:5000/
- Chicago model route: http://localhost:5000/chicago

## Running the Application
1. Make sure `xgb_boston_model.pkl` and `lgbm_chicago_model.pkl` is on the same folder of `app.py`, otherwise use scripts to generate
2. Start the Flask server:
   `
   python app.py
   `
3. Open your browser to `http://localhost:5000/` and enter feature values to get predictions.



