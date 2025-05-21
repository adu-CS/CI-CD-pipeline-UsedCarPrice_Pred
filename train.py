# train.py
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("car_data.csv")

# Filter out bikes if needed (example condition)
df = df[df["Car_Name"].str.lower().str.contains("bike") == False]

# Encode categorical columns
categorical_cols = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "label_encoder.pkl")

# Features & Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = XGBRegressor()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
