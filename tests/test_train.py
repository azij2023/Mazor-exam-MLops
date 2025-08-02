# tests/test_train.py

import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

# Fixture: load and split dataset
@pytest.fixture
def data():
    dataset = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_dataset_loading(data):
    X_train, X_test, y_train, y_test = data
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

def test_model_training(data):
    X_train, X_test, y_train, y_test = data
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Check if model is correct type
    assert isinstance(model, LinearRegression)
    
    # Check if model is trained (coef_ assigned)
    assert hasattr(model, "coef_")
    
    # Check performance
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.5, f"R² score too low: {r2}"
