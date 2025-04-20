import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Backup test IDs
test_ids = test["id"]

# Drop ID
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)

# Fill missing values
train.ffill(inplace=True)
test.ffill(inplace=True)

# Feature engineering
train["car_age"] = 2025 - train["model year"]
test["car_age"] = 2025 - test["model_year"]

# Align column names
train.rename(columns={"model year": "model_year"}, inplace=True)

# Drop original year column
train.drop("model_year", axis=1, inplace=True)
test.drop("model_year", axis=1, inplace=True)

# Categorical encoding
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
cat_cols.remove("price") if "price" in cat_cols else None

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Split features/target
X = train.drop("price", axis=1)
y = train["price"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Predict and evaluate
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {rmse:.2f}")

# Predict on test set
test_preds = model.predict(test)

# Save submission
submission = pd.DataFrame({
    "id": test_ids,
    "price": test_preds.astype(int)
})
submission.to_csv("submission.csv", index=False)

