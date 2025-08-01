import cv2
import numpy as np

def translate_layout_center(image, bounding_boxes):
    """
    指定された画像をバウンディングボックスでマスクして切り抜き、
    コマの中心を揃えて縦に結合する。

    Parameters:
        image (np.ndarray): 入力画像のNumPy配列。
        bounding_boxes (list of dict): 'xmin', 'ymin', 'xmax', 'ymax' をキーに持つバウンディングボックスのリスト。

    Returns:
        np.ndarray: 縦に結合された画像を表すNumPy配列。
    """
    # 元の画像の幅を取得
    original_width = image.shape[1]

    # 切り抜いた画像を格納するリストを初期化
    cropped_images = []

    for box in bounding_boxes:
        # バウンディングボックスの座標を取得
        xmin, ymin, xmax, ymax = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])

        # バウンディングボックスを使用して画像を切り抜く
        cropped_image = image[ymin:ymax, xmin:xmax]

        # 幅や高さが0の場合は無視
        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            continue

        cropped_images.append(cropped_image)

    # 結合画像の合計高さを計算
    total_height = sum(img.shape[0] for img in cropped_images)

    # 結合画像用真っ白の空白キャンバスを作成
    # combined_image = np.zeros((total_height, original_width, 3), dtype=np.uint8)
    combined_image = np.ones((total_height, original_width, 3), dtype=np.uint8) * 255

    # 中心を揃えて縦に結合
    current_y = 0
    for img in cropped_images:
        height, width = img.shape[:2]
        # コマをキャンバスの中心に配置するためのx座標を計算
        center_x = (original_width - width) // 2
        # コマをキャンバスに貼り付け
        combined_image[current_y:current_y + height, center_x:center_x + width, :] = img
        current_y += height

    return combined_image

if __name__ == '__main__':
    # テスト用の画像を読み込む
    image = cv2.imread('./image/comic_dataset/images/ARMS_000.jpg')
    output_file = 'ARMS_000_centered'

    # バウンディングボックスのリストを作成
    bounding_boxes = [
        {'xmin': 100, 'ymin': 50, 'xmax': 200, 'ymax': 150},
        {'xmin': 300, 'ymin': 200, 'xmax': 400, 'ymax': 300}
    ]

    # 画像を縦に結合
    combined_image = translate_layout_center(image, bounding_boxes)

    # 結合画像を表示
    cv2.imshow('Combined Image', combined_image)
    cv2.imwrite(f'./translated_imgs/{output_file}.jpg', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
