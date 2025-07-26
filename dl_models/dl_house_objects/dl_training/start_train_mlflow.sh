#!/bin/bash

# Ativa o virtualenv
source .venv/bin/activate

# Inicia o servidor MLflow em um novo terminal
gnome-terminal -- bash -c "source .venv/bin/activate; python -m mlflow server --backend-store-uri file:./shared/mlflow --host 0.0.0.0; exec bash"
sleep 3

python dl_models/dl_house_objects/dl_training/train_compare_yolos.py
wait