from ultralytics import YOLO
import cv2
import os

def detect_panel(image):
    """
    検出結果を指定形式に変換して返す関数。

    Args:
        image (numpy.ndarray): OpenCVで読み込んだ入力漫画画像。
        model_path (str): 学習済みモデル (YOLO) のパス。
        conf_threshold (float): 信頼値の閾値。

    Returns:
        list: 検出結果のリスト。各結果は次の形式で出力されます：
            - {"type": "frame", "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
    """
    # YOLOモデルのロード
    model = YOLO('./runs/detect/train5/weights/best.pt')

    # 推論を実行
    results = model.predict(image, imgsz=640, conf=0.45)

    # 検出結果をフォーマット
    detections = []
    for result in results:
        for box in result.boxes:
            # 座標を取得
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # 指定形式で結果を保存
            detections.append({
                "type": "frame",
                "xmin": int(x1),
                "ymin": int(y1),
                "xmax": int(x2),
                "ymax": int(y2)
            })

    return detections

# 使用例
if __name__ == "__main__":
    # 入力画像とモデルのパスを指定
    input_image = cv2.imread("./image/comic_dataset/images/ARMS_009.jpg")
    model_path = "./runs/detect/train/weights/best.pt"

    # 検出関数を実行
    results = detect_panel(input_image)

    # 結果を表示
    for detection in results:
        print(detection)
