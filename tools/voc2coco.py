"""Convert BCCD (Pascal VOC XML) dataset to COCO JSON format.

Usage:
    python tools/voc2coco.py

Output:
    date/blood_cell/annotations/instances_train.json
    date/blood_cell/annotations/instances_val.json
    date/blood_cell/annotations/instances_test.json
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET

# === Path config ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BCCD_ROOT = os.path.join(PROJECT_ROOT, 'BCCD')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'date', 'blood_cell')

# === Class config ===
CLASSES = {'RBC': 0, 'WBC': 1, 'Platelets': 2}


def parse_xml(xml_path):
    """Parse a single VOC XML file, return image info and annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASSES:
            continue
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        # VOC bbox → COCO bbox: [x, y, w, h]
        w = xmax - xmin
        h = ymax - ymin
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, w, h],
            'area': w * h,
        })

    return filename, width, height, objects


def convert_split(split_name):
    """Convert a split (train/val/test) to COCO JSON and copy images."""
    # Read split file
    split_file = os.path.join(BCCD_ROOT, 'ImageSets', 'Main', f'{split_name}.txt')
    with open(split_file, 'r') as f:
        image_ids = [line.strip() for line in f if line.strip()]

    # Create output directories
    img_out_dir = os.path.join(OUTPUT_ROOT, 'images', split_name)
    ann_out_dir = os.path.join(OUTPUT_ROOT, 'annotations')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(ann_out_dir, exist_ok=True)

    coco = {
        'info': {
            'description': 'BCCD Blood Cell Detection Dataset',
            'version': '1.0',
            'year': 2025,
        },
        'licenses': [{'id': 1, 'name': 'Unknown', 'url': ''}],
        'categories': [
            {'id': 0, 'name': 'RBC', 'supercategory': 'cell'},
            {'id': 1, 'name': 'WBC', 'supercategory': 'cell'},
            {'id': 2, 'name': 'Platelets', 'supercategory': 'cell'},
        ],
        'images': [],
        'annotations': [],
    }

    ann_id = 0
    for img_id, stem in enumerate(image_ids):
        xml_path = os.path.join(BCCD_ROOT, 'Annotations', f'{stem}.xml')
        if not os.path.exists(xml_path):
            print(f'  [跳过] {xml_path} 不存在')
            continue

        filename, width, height, objects = parse_xml(xml_path)

        # Copy image
        src_img = os.path.join(BCCD_ROOT, 'JPEGImages', filename)
        dst_img = os.path.join(img_out_dir, filename)
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

        # Add image info
        coco['images'].append({
            'id': img_id,
            'file_name': f'{split_name}/{filename}',
            'width': width,
            'height': height,
        })

        # Add annotations
        for obj in objects:
            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': CLASSES[obj['name']],
                'bbox': obj['bbox'],
                'area': obj['area'],
                'iscrowd': 0,
                'segmentation': [],
            })
            ann_id += 1

    # Save JSON
    out_path = os.path.join(ann_out_dir, f'instances_{split_name}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, indent=2)

    print(f'  [OK] {split_name}: {len(coco["images"])} images, '
          f'{len(coco["annotations"])} annotations -> {out_path}')
    return coco


if __name__ == '__main__':
    print('=' * 50)
    print('BCCD VOC -> COCO Converter')
    print('=' * 50)

    for split in ['train', 'val', 'test']:
        print(f'\nProcessing {split} set...')
        convert_split(split)

    print('\n' + '=' * 50)
    print('Done!')
    print(f'Output: {OUTPUT_ROOT}')
