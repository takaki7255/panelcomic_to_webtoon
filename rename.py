import os

def rename_images(base_dir):
    """
    manga109の画像ディレクトリ内のファイルを title_000.jpg の形式にリネームする。

    Args:
        base_dir (str): manga109のimagesディレクトリのパス
    """
    for title_dir in os.listdir(base_dir):
        title_path = os.path.join(base_dir, title_dir)
        if not os.path.isdir(title_path):
            continue  # ディレクトリでない場合はスキップ

        for file_name in os.listdir(title_path):
            if not file_name.endswith(".jpg"):
                continue  # jpg以外のファイルはスキップ

            # 新しいファイル名を生成
            new_name = f"{title_dir}_{file_name}"
            old_path = os.path.join(title_path, file_name)
            new_path = os.path.join(base_dir, new_name)

            # ファイル名を変更
            os.rename(old_path, new_path)

        # 元のディレクトリを削除（空になるため）
        os.rmdir(title_path)

# 使用例
base_dir = "./image/comic_dataset/images/"  # 画像のベースディレクトリ
rename_images(base_dir)
