#!/bin/bash

# Ativa o virtualenv (se estiver usando)
source .venv/bin/activate

# Inicia MLflow em http://localhost:5000
python -m mlflow server --backend-store-uri file:./shared/mlflow --host 0.0.0.0 &

sleep 3
python dl_models/dl_house_objects/dl_training/train_compare_yolos.py
wait