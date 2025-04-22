import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgbm
import numpy as np

df = pd.read_csv('chicago_data_final.csv')
X = df.drop('Sale Price', axis=1)
y = df['Sale Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
params = {
    'feature_fraction': np.float64(0.5900267817235574), 
    'lambda_l1': np.float64(0.24531083913245863), 
    'lambda_l2': np.float64(0.008500002679858797), 
    'learning_rate': np.float64(0.006964452798892195), 
    'max_bin': np.int64(311), 
    'min_data_in_leaf': np.int64(30), 
    'min_split_gain': np.float64(1700.2192095065927), 
    'n_estimators': np.int64(2365), 
    'num_leaves': np.int64(169)}

lgbm_model = lgbm.LGBMRegressor(**params, random_state=42)
lgbm_model.fit(X_train, y_train)

# joblib.dump(lgbm_model, 'lgbm_chicago_model.pkl')

pred = lgbm_model.predict(X_test)
lgbm_mse = mean_squared_error(y_test, pred)
lgbm_rmse = np.sqrt(lgbm_mse)
lgbm_mae = mean_absolute_error(y_test, pred)
lgbm_r2 = r2_score(y_test, pred)

print(f"LightGBM Results:")
print(f"Mean Squared Error: ${lgbm_mse:.2f}")
print(f"Root Mean Squared Error: ${lgbm_rmse:.2f}")
print(f"Mean Absolute Error: ${lgbm_mae:.2f}")
print(f"R² Score: {lgbm_r2:.4f}")

ref = pd.read_csv("chicago_ref_table.csv")

keys = ['EKW_2024','INC_2019-2023','CZM_2023','EDB_2019-2023']
ref_unique = ref.drop_duplicates(subset=keys)

remake = X_test.merge(
    ref_unique,
    on=keys,
    how='left'
)

final = pd.concat([
    remake.reset_index(drop=True),
    y_test.reset_index(drop=True).rename('True Sale Price'),
], axis=1)


final.to_csv("test_vals.csv", index=False)
