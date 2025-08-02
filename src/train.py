# src/train.py

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

# 1. Load the dataset
data = fetch_california_housing()
X, y = data.data, data.target

# 2. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# 5. Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/linear_regression_model.joblib")
import os

model_path = "models/linear_regression_model.joblib"
file_size_bytes = os.path.getsize(model_path)
file_size_kb = file_size_bytes / 1024

print(f"\nModel saved to: {model_path}")
print(f"Model file size: {file_size_kb:.2f} KB")
