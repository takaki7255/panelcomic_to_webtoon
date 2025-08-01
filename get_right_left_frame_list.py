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
    if not frames:
        print("No frames detected.")
        return [], []
    right_frames = []
    left_frames = []
    for frame in frames:
        if int(frame['xmin']) > (img_width // 2) - 10:
            frame['xmin'] = int(frame['xmin']) - (img_width // 2)
            frame['xmax'] = int(frame['xmax']) - (img_width // 2)
            frame['xmin'] = str(frame['xmin'])
            frame['xmax'] = str(frame['xmax'])
            right_frames.append(frame)
        else:
            left_frames.append(frame)
    return right_frames, left_frames