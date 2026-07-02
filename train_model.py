import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("heart.csv")

# Features (X) and Target (y)
X = df.drop("target", axis=1)
y = df["target"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("=" * 40)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("=" * 40)

# Save the trained model
joblib.dump(model, "model.pkl")

print("Model saved successfully as model.pkl")