import mlflow
from ultralytics import YOLO
import torch
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent
data_yaml_path = project_root / "dl_models" / "dl_house_objects" / "dl_model" / "df_unified" / "data.yaml"
mlflow_path = project_root / "shared" / "mlflow"

model_name = 'house_objects_yolov8_custom'
yolo_version_weight = 'yolov8s.pt'  #yolov9c.pt'

class CustomTrainer:
    def __init__(self):
        self.config = {
            'weights': yolo_version_weight,  # Modelo pré-treinado YOLOv8
            'data': str(data_yaml_path),
            'epochs': 100,
            'batch_size': 16,
            'imgsz': 640,
            'freeze':5,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'name': model_name,
            'optimizer': 'AdamW',
            'lr0': 0.0001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'augment': True,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'patience': 10,
            'workers': 8
        }

    def train(self):
        #mlflow.set_tracking_uri("file:" + str(Path("/shared/mlflow").absolute()))
        mlflow.set_tracking_uri("file:" + str(mlflow_path))
        mlflow.set_experiment("DL_House_Objects_YOLOv8-Custom")
        
        with mlflow.start_run():
            mlflow.log_params(self.config)
            model = YOLO(self.config['weights'])
            
            results = model.train(
                data=self.config['data'],
                epochs=self.config['epochs'],
                batch=self.config['batch_size'],
                imgsz=self.config['imgsz'],
                freeze=self.config['freeze'],
                device=self.config['device'],
                name=self.config['name'],
                optimizer=self.config['optimizer'],
                lr0=self.config['lr0'],
                lrf=self.config['lrf'],
                momentum=self.config['momentum'],
                weight_decay=self.config['weight_decay'],
                warmup_epochs=self.config['warmup_epochs'],
                hsv_h=self.config['hsv_h'],
                hsv_s=self.config['hsv_s'],
                hsv_v=self.config['hsv_v'],
                degrees=self.config['degrees'],
                translate=self.config['translate'],
                scale=self.config['scale'],
                workers=self.config['workers']
            )
            
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
            
            # Salva o melhor modelo
            mlflow.log_artifact(
                Path(results.save_dir) / "weights" / "best.pt",
                "model"
            )

if __name__ == '__main__':
    trainer = CustomTrainer()
    trainer.train()