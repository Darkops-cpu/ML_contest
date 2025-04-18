import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from math import sqrt

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Drop ID columns
test_ids = test["id"]
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)

# Fill missing values
train = train.ffill()
test = test.ffill()

# Add car age feature
train["car_age"] = 2025 - train["model_year"]
test["car_age"] = 2025 - test["model_year"]
train.drop("model_year", axis=1, inplace=True)
test.drop("model_year", axis=1, inplace=True)

# Drop noisy or highly redundant columns
drop_cols = ["engine_displacement", "engine_desc", "drivetrain", "doors"]
train.drop(columns=[col for col in drop_cols if col in train.columns], inplace=True)
test.drop(columns=[col for col in drop_cols if col in test.columns], inplace=True)

# Encode categorical variables
cat_cols = train.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    combined_vals = pd.concat([train[col], test[col]], axis=0).astype(str)
    encoder = LabelEncoder()
    encoder.fit(combined_vals)
    train[col] = encoder.transform(train[col].astype(str))
    test[col] = encoder.transform(test[col].astype(str))

# Define features and target
X = train.drop("price", axis=1)
y = train["price"]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model - Using LightGBM Regressor
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=14,
    num_leaves=31,
    min_data_in_leaf=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=100)

# Evaluate
val_preds = model.predict(X_val)
print("Validation RMSE:", sqrt(mean_squared_error(y_val, val_preds)))

# Predict on test set
test_preds = model.predict(test)

# Save submission
submission = pd.DataFrame({
    "id": test_ids,
    "price": test_preds.astype(int)
})
submission.to_csv("submission.csv", index=False)
