import mlflow
from ultralytics import YOLO
import torch
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent
data_yaml_path = project_root / "dl_models" / "dl_house_objects" / "dl_model" / "df_unified" / "data.yaml"
mlflow_path = project_root / "shared" / "mlflow"
yolo_runs_path = project_root / "shared" / "runs"  # Novo diretório para runs do YOLO

mlflow_path.mkdir(parents=True, exist_ok=True)
yolo_runs_path.mkdir(parents=True, exist_ok=True)


class CustomTrainer:
    def __init__(self):
        self.model_name = 'house_objects_yolov9_custom'
        self.yolo_version_weight = 'yolov9c.pt'  #yolov8s yolov9c.pt'

        self.config = {
            'data': str(data_yaml_path),
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'freeze':5,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'name': self.model_name,
            'optimizer': 'AdamW',
            'lr0': 3e-5, #1e-4
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'augment': True,
            'mosaic': 0.5,
            'mixup': 0.2,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'patience': 10,
            'workers': 8,
            'project': str(yolo_runs_path),  # Diretório para salvar os resultados
        }

    def train(self):
        mlflow.set_tracking_uri(f"file://{mlflow_path.resolve()}")
        mlflow.set_experiment("DL_House_Objects_YOLOv")
        
        with mlflow.start_run(run_name='YOLOv9_100_EPOCHS'):
            mlflow.log_params(self.config)
            model = YOLO(self.yolo_version_weight)
            results = model.train(**self.config)

            mlflow.set_tags({
                "task": "house-objects-detection",
                "framework": "YOLOv9",
                "dataset": Path(self.config['data']).stem,
                })
            
            try:
                # Métricas
                mlflow.log_metrics({
                    "mAP50": results.results_dict.get("metrics/mAP50(B)", 0.0),
                    "mAP50_95": results.results_dict.get("metrics/mAP50-95(B)", 0.0), 
                    "precision": results.results_dict.get("metrics/precision(B)", 0.0),
                    "recall": results.results_dict.get("metrics/recall(B)", 0.0),
                    "f1": results.results_dict.get("metrics/F1(B)", 0.0),
                    "inference_speed_ms": results.results_dict.get("speed/inference", 0.0),
                    "nms_speed_ms": results.results_dict.get("speed/nms", 0.0),
                    "preprocess_speed_ms": results.results_dict.get("speed/preprocess", 0.0)
                })

            except:
                # Problema pra gerar as metricas
                print("Problema pra gerar as metricas")
                pass
            
            # Salva o melhor modelo
            try:
                best_model_path = Path(results.save_dir) / "weights" / "best.pt"
                if best_model_path.exists():
                    mlflow.log_artifact(str(best_model_path), "weights")
                else:
                    print("Arquivo best.pt não encontrado")
            except Exception as e:
                print(f"Erro ao salvar modelo: {e}")

if __name__ == '__main__':
    trainer = CustomTrainer()
    trainer.train()