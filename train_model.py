import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("BankNote_Authentication.csv")  # Replace with your CSV file path

# Split features and labels
X = df.drop("class", axis=1)
y = df["class"]

# Train-test split (optional, here just to fit)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open("classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to classifier.pkl ✅")
