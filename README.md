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
All Python dependencies are listed in `requirements.txt`, including:
- pandas
- numpy
- scikit-learn
- xgboost
- Flask
- joblib
- matplotlib

## Setup
1. **Clone the repo**:
   `
   git clone https://github.com/your-org/housing-price-prediction.git
   cd housing-price-prediction
   `
2. **Install dependencies**:
   `
   python3 -m venv venv
   source venv/bin/activate      # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   `

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

### app.py
Runs a Flask web app for interactive predictions:
`
python app.py
`
- The app loads `xgb_boston_model.pkl`
- Accepts input features via a web form
- Displays predicted sale price
- Default route: http://localhost:5000/

### main.py
A placeholder entry script. Currently prints a message:
`
python main.py
`

## Running the Application
1. Train or load the Boston model:
   `
   python Boston_model.py
   `
2. Start the Flask server:
   `
   python app.py
   `
3. Open your browser to `http://localhost:5000/` and enter feature values to get predictions.



