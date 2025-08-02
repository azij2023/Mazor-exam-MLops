# Use a small official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy script and model
COPY src/predict.py src/predict.py
COPY models/ models/

# Run prediction script
CMD ["python", "src/predict.py"]
