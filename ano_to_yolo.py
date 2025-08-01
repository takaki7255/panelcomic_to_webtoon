import os
import xml.etree.ElementTree as ET

def convert_to_yolo_format(xmin, ymin, xmax, ymax, image_width, image_height):
    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    return x_center, y_center, width, height

def process_annotation_file(xml_file_path, output_dir):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    file_name = os.path.splitext(os.path.basename(xml_file_path))[0]

    for page in root.findall('.//page'):
        page_index = int(page.get('index'))
        page_width = int(page.get('width'))
        page_height = int(page.get('height'))

        yolo_annotations = []
        for body in page.findall('.//frame'):
            xmin = int(body.get('xmin'))
            ymin = int(body.get('ymin'))
            xmax = int(body.get('xmax'))
            ymax = int(body.get('ymax'))

            x_center, y_center, width, height = convert_to_yolo_format(xmin, ymin, xmax, ymax, page_width, page_height)

            class_id = 0  # 任意のクラスID
            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

        output_file_name = f"{file_name}_{page_index:03}.txt"
        output_file_path = os.path.join(output_dir, output_file_name)
        with open(output_file_path, 'w') as f:
            for annotation in yolo_annotations:
                f.write(annotation + '\n')

    print(f"{file_name}のYOLO形式のアノテーションファイル保存完了")

def process_all_xml_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(input_dir):
        if xml_file.endswith('.xml'):
            xml_file_path = os.path.join(input_dir, xml_file)
            process_annotation_file(xml_file_path, output_dir)

# 入力XMLファイルのディレクトリのパス
input_dir = './../Manga109_released_2023_12_07/annotations/'
# 出力ディレクトリのパス
output_dir = './image/comic_dataset/labels'

process_all_xml_files(input_dir, output_dir)