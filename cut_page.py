from get_frame_bbox import get_framebbox

def cut_page(image):
    """
    見開きページを左右に分割する
    Args:
        image (np.ndarray): 見開きページの画像
    Returns:
        left_image (np.ndarray): 右ページの画像
        right_image (np.ndarray): 左ページの画像
    """
    if image is None:
        raise ValueError("image is None")
    pageImg = []
    if image.shape[1] > image.shape[0]:  # 縦 < 横の場合: 見開きだと判断し真ん中で切断
        cut_img_left = image[:, : image.shape[1] // 2]  # 右ページ
        cut_img_right = image[:, image.shape[1] // 2 :]  # 左ページ
        pageImg.append(cut_img_right)
        pageImg.append(cut_img_left)
    else:  # 縦 > 横の場合: 単一ページ画像だと判断しそのまま保存
        pageImg.append(image)
    return pageImg

def get_right_left_frame_list(frames,img_width):
    """
    コマリストを左右に分割する
    Args:
        frames (list of dict): コマのバウンディングボックス情報
            [{'type': 'frame', 'id': '...', 'xmin': '...', 'ymin': '...', 'xmax': '...', 'ymax': '...'}, ...]
        img_width (int): 画像の横幅
    Returns:
        list: 右ページのコマリスト，左ページのコマリスト
    """
    right_frames = []
    left_frames = []
    for frame in frames:
        if int(frame['xmin']) > img_width // 2:
            frame['xmin'] = int(frame['xmin']) - (img_width // 2)
            frame['xmax'] = int(frame['xmax']) - (img_width // 2)
            frame['xmin'] = str(frame['xmin'])
            frame['xmax'] = str(frame['xmax'])
            right_frames.append(frame)
        else:
            left_frames.append(frame)
    return right_frames, left_frames
    

import cv2
if __name__ ==  '__main__':
    # テストデータ
    image = cv2.imread('./../Manga109_released_2023_12_07/images/ARMS/004.jpg')
    ano = './../Manga109_released_2023_12_07/annotations/ARMS.xml'
    frame_list = get_framebbox(ano)
    img_width = image.shape[1]
    right_frame_list, left_frame_list = get_right_left_frame_list(frame_list[4],img_width)
    right_image, left_image = cut_page(image)
    for frame in right_frame_list:
        x1 = frame['xmin']
        y1 = frame['ymin']
        x2 = frame['xmax']
        y2 = frame['ymax']
        cv2.rectangle(right_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.imshow('right', right_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    for frame in left_frame_list:
        x1 = frame['xmin']
        y1 = frame['ymin']
        x2 = frame['xmax']
        y2 = frame['ymax']
        cv2.rectangle(left_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.imshow('left', left_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()