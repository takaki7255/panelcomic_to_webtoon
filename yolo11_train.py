from ultralytics import YOLO
import os

data_path = './datasets/data.yaml'

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")
# 学習途中のmodelを使って学習再開する場合のパス
# model = YOLO('./runs/detect/train12/weights/last.pt')

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(
    data = data_path,
    epochs = 300,
    patience = 50,
    imgsz = 640
    # device = '0',
)