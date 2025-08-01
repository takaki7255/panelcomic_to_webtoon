from ultralytics import YOLO

#load the model
model = YOLO('./runs/detect/train5/weights/best.pt')

#detect the object
results = model.predict('./image/comic_dataset/images/ARMS_008.jpg', imgsz=640, conf=0.250)

#display the result
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
