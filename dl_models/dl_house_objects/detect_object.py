import cv2
import torch
from ultralytics import YOLO
from datetime import datetime

class CustomDetector:
    def __init__(self, model_path, conf_thres=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.names = self.model.names
        self.window_name = "Detecção de Objetos"
    
    def predict(self, modo, source, save_dir=None):
        # Modo imagem única
        if modo == "imagem":
            img = cv2.imread(source)
            if img is None:
                raise FileNotFoundError(f"Imagem não encontrada: {source}")
            
            results = self.model.predict(
                source=img,
                conf=self.conf_thres,
                device=self.device,
                verbose=False
            )[0]
            
            # Salva/exibe resultado
            result_img = results.plot()
            if save_dir:
                file = source.name                
                filename = f"predict_{file}.jpg"
                save_path = f"{save_dir}/{filename}"
                cv2.imwrite(save_path, result_img)
                print(f"Resultado salvo em: {save_path}")
            else:
                cv2.imshow('Resultado', result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return results
        
        # Modo webcam
        elif modo == "webcam":
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Câmera não encontrada: índice {source}")
            print("Webcam ativa. Pressione 'q' para sair | 's' para salvar frame")

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Erro na captura")
                        break
                    
                    # Inferência
                    results = self.model.predict(
                        source=frame,
                        conf=self.conf_thres,
                        device=self.device,
                        verbose=False
                    )[0]
                    
                    # Mostra resultado
                    result_frame = results.plot()
                    cv2.imshow(self.window_name, result_frame)
                    
                    # Controles
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and save_dir:
                        filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        save_path = f"{save_dir}/{filename}"
                        cv2.imwrite(save_path, result_frame)
                        print(f"Frame salvo: {save_path}")
            finally:
                cap.release()
                cv2.destroyWindow(self.window_name)
        else:
            raise ValueError("Modo inválido. Use 'imagem' ou 'webcam'")