import os
from pathlib import Path
import cv2
from detect_object import CustomDetector

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent

# Configs
MODEL_PATH = project_root / "shared" / "runs" / "house_objects_yolov9_custom4" / "weights" / "best.pt"  # Caminho do seu modelo
CONF_THRES = 0.6
SAVE_DIR = project_root / "dl_models" / "dl_house_objects" / "dl_model" / "imgs_to_predict"
MODO = "webcam"    # "imagem" ou "webcam"
FILENAME = "house_interior.jpg"     

os.makedirs(SAVE_DIR, exist_ok=True)

def find_webcam(max_index=5):
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            
            if ret:
                return index
    return None  # Nenhuma webcam encontrada


detector = CustomDetector(model_path=MODEL_PATH, conf_thres=CONF_THRES)
# Modo de predicao
if MODO == "imagem":
    SOURCE = SAVE_DIR / FILENAME
    detector.predict(modo="imagem", source=SOURCE, save_dir=SAVE_DIR)

elif MODO == "webcam":
    webcam_index = find_webcam()
    if webcam_index is None:
        print("Nenhuma webcam encontrada!")
    else:
        print(f"Usando webcam no índice {webcam_index}")
        try:
            detector.predict(modo="webcam", source=webcam_index, save_dir=SAVE_DIR)
        finally:
            cv2.destroyAllWindows()

else:
    raise ValueError("Modo inválido. Use 'imagem' ou 'webcam'")

print("Execução concluída!")

