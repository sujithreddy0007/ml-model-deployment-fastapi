import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_diabetes
import joblib

# Load sample diabetes dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Convert target to binary: 1 if target > 140 else 0
y = (diabetes.target > 140).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model to a .pkl file
joblib.dump(model, "model.pkl")

print("✅ model.pkl created successfully.")
