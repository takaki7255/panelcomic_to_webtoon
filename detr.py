from manga109api import Parser
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class Manga109Dataset(Dataset):
    def __init__(self, root_dir, parser, transform=None):
        """
        Manga109用データセットクラス。
        Args:
            root_dir (str): Manga109データセットのルートディレクトリ
            parser (Parser): Manga109APIのParserインスタンス
            transform (callable, optional): 画像に適用する変換関数
        """
        self.root_dir = root_dir
        self.parser = parser
        self.books = parser.books
        self.transform = transform

    def __len__(self):
        return len(self.books)

    def __getitem__(self, idx):
        book = self.books[idx]
        book_data = self.parser.get_annotation(book)
        page_data = book_data['page'][0]  # 1つのページを例として取得
        image_path = os.path.join(self.root_dir, "images", book, page_data['filename'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 境界ボックスを取得
        boxes = []
        labels = []
        for obj in page_data['annotation']:
            bbox = obj['bbox']  # [xmin, ymin, xmax, ymax]
            boxes.append(bbox)
            labels.append(obj['type'])  # ラベル名

        # 画像変換（例: 正規化）
        if self.transform:
            image = self.transform(image)

        # Tensor化
        image = F.to_tensor(image)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        return image, {"boxes": boxes, "labels": labels}

# データセット初期化
manga109_root = "./../Manga109_released_2023_12_07/"
parser = Parser(root_dir=manga109_root)

dataset = Manga109Dataset(root_dir=manga109_root, parser=parser)

# サンプルを取得
image, target = dataset[0]
print("画像の形状:", image.shape)
print("ターゲット:", target)
