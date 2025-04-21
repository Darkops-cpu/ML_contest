import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Feature engineering - focused only on the most important features
def engineer_features(df):
    df = df.copy()
    
    # Top features from previous model
    df['car_age'] = 2020 - df['model_year']
    df['log_milage'] = np.log1p(df['milage'])
    df['milage_squared'] = df['milage'] ** 2
    
    # Extract only horsepower from engine 
    df['engine_hp'] = df['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)
    
    # Add milage_bin
    df['milage_bin'] = pd.qcut(df['milage'], 10, labels=False, duplicates='drop')
    
    # Add brand_model
    df['brand_model'] = df['brand'] + '_' + df['model']
    
    # New features
    df['milage_per_hp'] = df['milage'] / (df['engine_hp'] + 1)
    df['age_milage_interaction'] = df['car_age'] * df['milage']
    
    # Fix: Avoid division by zero by adding a small epsilon and handling negative ages
    df['car_age_safe'] = np.maximum(df['car_age'], 0) + 0.1  # At least 0.1 to avoid division by zero
    df['milage_per_age'] = df['milage'] / df['car_age_safe']
    
    df['brand_freq'] = df['brand'].map(df['brand'].value_counts())
    df['model_freq'] = df['model'].map(df['model'].value_counts())
    luxury_brands = ['BMW', 'Mercedes', 'Audi', 'Lexus', 'Jaguar', 'Porsche', 'Land Rover']
    df['is_luxury_brand'] = df['brand'].isin(luxury_brands).astype(int)
    df['car_age_bin'] = pd.cut(df['car_age'], bins=[-1,2,5,10,100], labels=[0,1,2,3])
    
    return df

# Process data
print("Engineering features...")
train = engineer_features(train)
test = engineer_features(test)

# Drop extreme price outliers (0.05% and 99.95%)
if 'price' in train.columns:
    low, high = train['price'].quantile([0.01, 0.99])
    train_clean = train[(train['price'] >= low) & (train['price'] <= high)]
    print(f"Removed {len(train) - len(train_clean)} outliers")
    train = train_clean

# Fill missing values
print("Handling missing values...")
for df in [train, test]:
    # Categorical
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')
    # Numeric
    for col in df.select_dtypes(include=['number']).columns:
        if col != 'price':  # Don't impute target
            df[col] = df[col].fillna(df[col].median())

# Label encode brand_model
le = LabelEncoder()
all_vals = pd.concat([train['brand_model'], test['brand_model']], axis=0).astype(str)
le.fit(all_vals)
train['brand_model'] = le.transform(train['brand_model'].astype(str))
test['brand_model'] = le.transform(test['brand_model'].astype(str))

# Target encoding for only the most important categorical features
cat_cols = ['brand', 'model', 'transmission', 'accident', 'clean_title']
cat_cols = [c for c in cat_cols if c in train.columns and c in test.columns]

# Simple mean encoding
print("Encoding categorical features...")
encoded_cols = []
for col in cat_cols:
    mean_enc = train.groupby(col)['price'].mean()
    train[f'{col}_enc'] = train[col].map(mean_enc)
    test[f'{col}_enc'] = test[col].map(mean_enc).fillna(train['price'].mean())
    encoded_cols.append(f'{col}_enc')

# Pruned feature set
numeric_cols = ['log_milage', 'milage_squared', 'milage', 'milage_bin', 'car_age', 'brand_model', 'engine_hp', 'milage_per_hp', 'age_milage_interaction', 'milage_per_age', 'brand_freq', 'model_freq', 'is_luxury_brand', 'car_age_bin']
feature_cols = numeric_cols + encoded_cols

X = train[feature_cols]
y = train['price']
X_test = test[feature_cols]

printf"Usin…