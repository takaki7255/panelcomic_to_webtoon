import os
import shutil
import random

# 抽出するデータセットがあるディレクトリパスを指定　ここに後述のディレクトリ名で新しく作成される
dataset_dir = './image/comic_dataset/'
now_dataset_dir = './datasets/'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
new_dataset_dir = os.path.join(dataset_dir, 'new_500datasets')# 抽出後のデータセットを格納するディレクトリ名を指定

train_dir = os.path.join(new_dataset_dir, 'train')
valid_dir = os.path.join(new_dataset_dir, 'valid')
test_dir = os.path.join(new_dataset_dir, 'test')

for split in [train_dir, valid_dir, test_dir]:
    os.makedirs(os.path.join(split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(split, 'labels'), exist_ok=True)

image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
print(f"データセットの総数: {len(image_files)}")

# すでにデータセット内に存在するファイルを削除
now_train_files = [f for f in os.listdir(os.path.join(now_dataset_dir, 'train', 'images')) if f.endswith('.jpg')]
now_valid_files = [f for f in os.listdir(os.path.join(now_dataset_dir, 'valid', 'images')) if f.endswith('.jpg')]
now_test_files = [f for f in os.listdir(os.path.join(now_dataset_dir, 'test', 'images')) if f.endswith('.jpg')]
now_files = now_train_files + now_valid_files + now_test_files
new_files = [f for f in image_files if f not in now_files]

# ランダムに500ファイルを抽出
selected_files = random.sample(now_files, 100)

train_split = int(0.6 * len(selected_files)) # 抽出した500ファイルのうち60%をtrainディレクトリに格納
valid_split = int(0.1 * len(selected_files))# 抽出した500ファイルのうち10%をvalidディレクトリに格納
test_split = len(selected_files) - train_split - valid_split #残りをtestディレクトリに格納

train_files = selected_files[:train_split]
valid_files = selected_files[train_split:train_split + valid_split]
test_files = selected_files[train_split + valid_split:]

def copy_files(files, split_dir):
    for file in files:
        image_src = os.path.join(images_dir, file)
        label_src = os.path.join(labels_dir, file.replace('.jpg', '.txt'))
        image_dst = os.path.join(split_dir, 'images', file)
        label_dst = os.path.join(split_dir, 'labels', file.replace('.jpg', '.txt'))
        
        shutil.copy(image_src, image_dst)
        shutil.copy(label_src, label_dst)

copy_files(train_files, train_dir)
copy_files(valid_files, valid_dir)
copy_files(test_files, test_dir)

print("データセットの分割が完了しました。")