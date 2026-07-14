import torch
from ultralytics import YOLO

def main():
    print("[1] Dang su dung dataset Udacity da giai nen...")
    dataset_yaml = r"Udacity Self Driving Car Dataset.v1i.yolov11\data.yaml"
    
    print("[2] Bat dau huan luyen voi RTX 3050...")
    # Load model gốc tốt nhất
    model = YOLO("yolo11n.pt")
    
    # Chạy train 50 epochs
    model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        name="fast_adas_model",
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=2,
        patience=10
    )
    print("HOAN THANH! Ban da co the lay file runs/detect/fast_adas_model/weights/best.pt de su dung!")

if __name__ == "__main__":
    main()
