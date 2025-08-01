import os
import json
import yaml
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# Function to convert Manga109 XML to COCO format
def convert_to_coco(images_dir, annotations_dir, output_file):
    coco = {
        "info": {
            "description": "Manga109 Dataset in COCO Format",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "frame", "supercategory": "frame"}
        ],
    }

    annotation_id = 1
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        
        # Map title to images
        title = root.attrib['title']
        for page in root.find('pages'):
            page_index = int(page.attrib['index']) + 1
            image_file = os.path.join(images_dir, title, f"{page_index:03}.jpg")
            
            # Add image info
            image_id = len(coco['images']) + 1
            coco['images'].append({
                "id": image_id,
                "file_name": image_file,
                "width": int(page.attrib['width']),
                "height": int(page.attrib['height']),
            })

            # Add annotations
            for frame in page.findall('frame'):
                xmin = int(frame.attrib['xmin'])
                ymin = int(frame.attrib['ymin'])
                xmax = int(frame.attrib['xmax'])
                ymax = int(frame.attrib['ymax'])

                coco['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0,
                })
                annotation_id += 1

    # Write COCO JSON
    with open(output_file, 'w') as f:
        json.dump(coco, f, indent=4)

# Function to save COCO data in YAML format
def save_coco_as_yaml(coco_data, output_file):
    with open(output_file, 'w') as f:
        yaml.dump(coco_data, f, allow_unicode=True, sort_keys=False)

# Dataset splitting function
def split_dataset(coco_json_path, train_ratio=0.8, val_ratio=0.1):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    image_ids = [img['id'] for img in coco['images']]
    train_ids, test_ids = train_test_split(image_ids, test_size=(1 - train_ratio))
    val_ids, test_ids = train_test_split(test_ids, test_size=val_ratio / (1 - train_ratio))

    def filter_coco_by_ids(image_ids):
        filtered = {
            "info": coco['info'],
            "licenses": coco['licenses'],
            "images": [img for img in coco['images'] if img['id'] in image_ids],
            "annotations": [anno for anno in coco['annotations'] if anno['image_id'] in image_ids],
            "categories": coco['categories'],
        }
        return filtered

    train_coco = filter_coco_by_ids(train_ids)
    val_coco = filter_coco_by_ids(val_ids)
    test_coco = filter_coco_by_ids(test_ids)

    # Save splits in JSON
    with open('train_coco.json', 'w') as f:
        json.dump(train_coco, f, indent=4)
    with open('val_coco.json', 'w') as f:
        json.dump(val_coco, f, indent=4)
    with open('test_coco.json', 'w') as f:
        json.dump(test_coco, f, indent=4)

    # Save splits in YAML
    save_coco_as_yaml(train_coco, 'train_coco.yaml')
    save_coco_as_yaml(val_coco, 'val_coco.yaml')
    save_coco_as_yaml(test_coco, 'test_coco.yaml')

# Example usage
images_dir = "./../Manga109_released_2023_12_07/images/"
annotations_dir = "./../Manga109_released_2023_12_07/annotations/"
output_file = "manga109_coco.json"
convert_to_coco(images_dir, annotations_dir, output_file)
split_dataset(output_file)
