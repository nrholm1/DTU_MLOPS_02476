# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY mlops_mnist_classifier/ src/
# data is copied
COPY data/ data/ 

# create logging directories
RUN mkdir src/trained_models

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/predict_model.py", "predict", "src/trained_models/trained_model.pt", "data/processed/predict.npy"]