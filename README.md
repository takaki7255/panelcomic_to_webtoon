# Panel Comic to Webtoon Converter

漫画のパネル（コマ）を自動検出し、Webtoon形式に変換するプロジェクトです。YOLOv11を使用してコマの検出を行い、検出されたコマの順序を推定して縦スクロール形式に再配置します。

## 概要

このプロジェクトは以下の処理を自動化します：

1. **コマ検出**: YOLOv11を使用して漫画ページからコマを自動検出
2. **ページ分割**: 左右のページを分離
3. **コマ順序推定**: 読む順序に基づいてコマを並び替え
4. **レイアウト変換**: Webtoon形式（縦スクロール）に変換

## 機能

- 🔍 **自動コマ検出**: 学習済みYOLOモデルによる高精度コマ検出
- 📄 **ページ分割**: 見開きページの左右自動分離
- 🔄 **順序推定**: 日本語漫画の読み順に基づく自動コマ並び替え
- 📱 **Webtoon変換**: モバイル向け縦スクロール形式への変換
- 🎨 **レイアウト調整**: リサイズあり/なしの選択可能

## 必要要件

### Python環境
- Python 3.8+
- OpenCV
- Ultralytics YOLO
- NumPy

### インストール

```bash
pip install ultralytics opencv-python numpy
```

## プロジェクト構成

```
panelcomic_to_webtoon/
├── main.py                          # メイン実行ファイル
├── main_rule.py                     # ルールベース処理版
├── detect_panel.py                  # パネル検出モジュール
├── cut_page.py                      # ページ分割モジュール
├── get_right_left_frame_list.py     # 左右フレーム分離
├── panel_order_estimater.py         # パネル順序推定
├── translate_layout.py              # レイアウト変換（リサイズあり）
├── translate_layout_noresize.py     # レイアウト変換（リサイズなし）
├── modules.py                       # 共通モジュール
├── yolo11_train.py                  # YOLOモデル学習
├── yolo11_detect.py                 # YOLO推論
├── datasets/                        # 学習データセット
├── runs/                           # 学習結果
├── test/                           # テスト用画像
├── translated_imgs/                # 変換結果画像
└── models/
    ├── yolo11n.pt                  # YOLOv11 Nanoモデル
    ├── yolov8n.pt                  # YOLOv8 Nanoモデル
    └── rtdetr-l.pt                 # RT-DETR Largeモデル
```

## 使用方法

### 1. 基本的な使用方法

```bash
python main.py
```

テストディレクトリ内の全ての漫画画像を処理し、変換結果を`translated_imgs/`に保存します。

### 2. モデル学習

新しいデータセットでモデルを学習する場合：

```bash
python yolo11_train.py
```

学習設定:
- エポック数: 300
- 画像サイズ: 640x640
- 早期停止: 50エポック
- データセット: COCOフォーマット

### 3. データセット準備

#### データセット形式
```
datasets/
├── data.yaml          # データセット設定
├── train/
│   ├── images/        # 学習用画像
│   └── labels/        # YOLOフォーマットアノテーション
├── valid/
│   ├── images/        # 検証用画像
│   └── labels/        # YOLOフォーマットアノテーション
└── test/
    ├── images/        # テスト用画像
    └── labels/        # YOLOフォーマットアノテーション
```

#### アノテーション形式
YOLOフォーマット（クラス x_center y_center width height）
```
0 0.5 0.3 0.4 0.6
0 0.2 0.7 0.3 0.5
```

## モジュール説明

### `detect_panel.py`
- YOLOv11を使用してコマを検出
- 信頼度閾値: 0.45
- 戻り値: `{"type": "frame", "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}`

### `cut_page.py`
- 見開きページを左右に分割
- 画像の中央で分離

### `panel_order_estimater.py`
- 検出されたコマの読み順を推定
- 日本語漫画の右から左、上から下の読み順に対応

### `translate_layout.py` / `translate_layout_noresize.py`
- コマを縦スクロール形式に再配置
- リサイズあり版とリサイズなし（中央配置）版を提供

## パラメータ調整

### 検出パラメータ
- `conf_threshold`: 検出信頼度閾値（デフォルト: 0.45）
- `imgsz`: 推論時画像サイズ（デフォルト: 640）

### 学習パラメータ
- `epochs`: 学習エポック数（デフォルト: 300）
- `patience`: 早期停止エポック数（デフォルト: 50）
- `imgsz`: 学習時画像サイズ（デフォルト: 640）

## 入力・出力

### 入力
- 漫画画像ファイル（JPG, PNG対応）
- サポート形式: 単ページ、見開きページ

### 出力
- Webtoon形式変換画像
- ファイル名形式: `{元ファイル名}_{left/right}_translated.jpg`

## トラブルシューティング

### よくある問題

1. **モデルファイルが見つからない**
   ```
   FileNotFoundError: ./runs/detect/train5/weights/best.pt
   ```
   解決策: モデルを学習するか、事前学習済みモデルパスを変更

2. **メモリエラー**
   - 画像サイズを小さくする
   - バッチサイズを調整する

3. **検出精度が低い**
   - 信頼度閾値を調整
   - より多くのデータで再学習

## ライセンス

このプロジェクトは研究目的で開発されています。

## 更新履歴

- v1.0: 初期リリース
- YOLOv11対応
- Webtoon変換機能追加

## 今後の改善予定

- [ ] より高精度な順序推定アルゴリズム
- [ ] GUI界面の追加
- [ ] バッチ処理の最適化
- [ ] 多言語漫画対応
- [ ] クラウドデプロイメント対応

## 貢献

バグ報告や機能改善の提案は Issues でお知らせください。

## 参考文献

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [Manga109 Dataset](http://www.manga109.org/)
- OpenCV Documentation
