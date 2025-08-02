# California Housing Price Prediction (MLOps Pipeline)

This project demonstrates a complete MLOps pipeline using the California Housing dataset with:

- Linear Regression model training
-  Manual quantization of model weights
-  Model evaluation (R², MSE, size)
-  Dockerized prediction
-  CI/CD pipeline using GitHub Actions

---

##  Comparison Table

| Metric             | Original Model | Quantized Model |
|--------------------|----------------|------------------|
| R² Score           | 0.5758          | 0.5566           |
| MSE                | 0.5559          | 0.5810            |
| Model File Size    | 68 KB           | 46 KB            |

>  Note: Quantized model shows degraded performance due to naive manual quantization (int8 conversion without proper scaling). Optimization techniques can improve this.

### Virtual environment

python -m venv mlops-env

mlops-env\Scripts\activate 


## Project Structure

mlops_major/
├── src/
│ ├── train.py 
│ ├── predict.py 
│ ├── quantize.py 
│ └── utils.py 
├── tests/
│ └── test_train.py 
├── models/ # Trained and quantized models
├── Dockerfile 
├── requirements.txt 
├── .github/workflows/
│ └── ci.yml 
└── README.md 


---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt

Train the Model
python src/train.py

Predict with Trained Model

python src/predict.py

Test with Trained Model

python test/test.py

Manual Quantization

python src/quantize.py

### Docker Inference

docker build -t housing-predict .
docker run --rm housing-predict
### predict.py for model verification

Predicted: 0.552 | Actual: 0.477
Predicted: 1.613 | Actual: 0.458
Predicted: 2.616 | Actual: 5.000
Predicted: 2.658 | Actual: 2.186
Predicted: 2.461 | Actual: 2.780

### CI/CD (GitHub Actions)

 Automatically runs on every push to main
 Performs:

Linting & tests

Docker build check

Workflow file: .github/workflows/ci.yml

## Notes
Quantization here is manually done using int8 conversion (no real scaling/compression).

R² becomes negative post-quantization due to poor preservation of model weights.