import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Step 1: Load trained model
model_path = "models/linear_regression_model.joblib"
model = joblib.load(model_path)
print("Loaded trained model.")

# Step 2: Load dataset and split
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Step 3: Run predictions
predictions = model.predict(X_test)

# Step 4: Print sample predictions
print("\nSample predictions:")
for i in range(5):
    print(f"Predicted: {predictions[i]:.3f} | Actual: {y_test[i]:.3f}")
