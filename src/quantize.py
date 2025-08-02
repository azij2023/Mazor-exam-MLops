import numpy as np
import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

print("Starting quantization...")

# Step 1: Load trained model
model_path = "models/linear_regression_model.joblib"
model = joblib.load(model_path)
coef = model.coef_
intercept = model.intercept_

# Step 2: Save raw parameters
raw_params = {
    "coef": coef,
    "intercept": intercept
}
joblib.dump(raw_params, "models/unquant_params.joblib")
print("Saved unquantized parameters to models/unquant_params.joblib")

# Step 3: Quantize weights manually using symmetric int8 quantization
def symmetric_quantize(arr):
    max_val = np.max(np.abs(arr))
    scale = max_val / 127.0 if max_val != 0 else 1.0
    q_arr = np.round(arr / scale).astype(np.int8)
    return q_arr, scale

def symmetric_dequantize(q_arr, scale):
    return q_arr.astype(np.float32) * scale

q_coef, coef_scale = symmetric_quantize(coef)
q_intercept, intercept_scale = symmetric_quantize(np.array([intercept]))

# Step 4: Save quantized parameters
quant_params = {
    "q_coef": q_coef,
    "q_intercept": q_intercept,
    "coef_scale": coef_scale,
    "intercept_scale": intercept_scale
}
joblib.dump(quant_params, "models/quant_params.joblib")
print("Saved quantized parameters to models/quant_params.joblib")

# Step 5: Dequantize for inference
dq_coef = symmetric_dequantize(q_coef, coef_scale)
dq_intercept = symmetric_dequantize(q_intercept, intercept_scale)[0]

# Step 6: Load test data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Step 7: Predict using dequantized weights
y_pred = X_test @ dq_coef + dq_intercept

# Step 8: Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
quant_file_size = os.path.getsize("models/quant_params.joblib") / 1024

print("\nEvaluation of quantized model:")
print(f"R2 Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"Quantized model file size: {quant_file_size:.2f} KB")

