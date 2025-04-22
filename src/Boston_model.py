import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and clean data
data = pd.read_csv("BostonHousing.csv")
data.dropna(inplace=True)  # Drop rows with any null values

# Use 7 most correlated features with 'medv'
features = ["lstat", "rm", "ptratio", "indus", "tax", "nox", "crim"]
X = data[features]
y = data["medv"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure input is float for XGBoost
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Define and train the XGBoost model
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10,
    reg_alpha=1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Predictions
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Evaluation
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

joblib.dump(xgb_model, "xgb_boston_model.pkl")
# Results
print(f"Training Mean Squared Error: {train_mse:.4f}")
print(f"Testing Mean Squared Error: {test_mse:.4f}")
print(f"Training R-squared Score: {train_r2:.4f}")
print(f"Testing R-squared Score: {test_r2:.4f}")

# Feature importance plot
xgb.plot_importance(xgb_model)
plt.title("Feature Importance in XGBoost")
plt.show()
