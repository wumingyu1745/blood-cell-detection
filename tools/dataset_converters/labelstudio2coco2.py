import argparse
import io
import json
import logging
import pathlib
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.model_selection import train_test_split

import numpy as np

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Label Studio JSON file to COCO format JSON File')
    parser.add_argument('config', help='Labeling Interface xml code file path')
    parser.add_argument('input', help='Label Studio format JSON file path')
    parser.add_argument('output', help='The output COCO format JSON file path')
    parser.add_argument('--output_dir', help='The output directory for COCO format JSON files')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of test set')
    args = parser.parse_args()
    return args


class LSConverter:

    def __init__(self, config: str):
        # get label info from config file
        tree = ET.parse(config)
        root = tree.getroot()
        labels = root.findall('.//KeyPointLabels/Label') + \
            root.findall('.//PolygonLabels/Label') + \
            root.findall('.//RectangleLabels/Label')
        label_values = [label.get('value') for label in labels]

        self.categories = list()
        self.category_name_to_id = dict()
        for i, value in enumerate(label_values):
            # category id start with 1
            self.categories.append({'id': i + 1, 'name': value})
            self.category_name_to_id[value] = i + 1

    def convert_to_coco(self, input_json: str, output_json: str):
        """Convert `input_json` to COCO format and save in `output_json`.

        Args:
            input_json (str): The path of Label Studio format JSON file.
            output_json (str): The path of the output COCO JSON file.
        """

        def add_image(images, width, height, image_id, image_path):
            images.append({
                'width': width,
                'height': height,
                'id': image_id,
                'file_name': image_path,
            })
            return images
    

        output_path = pathlib.Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        images = list()
        annotations = list()

        with open(input_json, 'r') as f:
            ann_list = json.load(f)

        for item_idx, item in enumerate(ann_list):
            # each image is an item
            image_name = item['data']
            image_id = len(images)
            width, height = None, None

            # skip tasks without annotations
            if not item['annotations']:
                logger.warning('No annotations found for item #' +
                               str(item_idx))
                continue

            instance_annotations = {}

            for i, label in enumerate(item['annotations'][0]['result']):
                category_name = None

                for key in [
                        'rectanglelabels', 'polygonlabels', 'labels',
                        'keypointlabels'
                ]:
                    if key == label['type'] and len(label['value'][key]) > 0:
                        category_name = label['value'][key][0]
                        break

                if category_name is None:
                    logger.warning('Unknown label type or labels are empty')
                    continue

                if not height or not width:
                    if 'original_width' not in label or \
                            'original_height' not in label:
                        logger.debug(
                            f'original_width or original_height not found'
                            f'in {image_name}')
                        continue

                    width, height = label['original_width'], label[
                        'original_height']
                    image_name = image_name['image']
                    image_name = image_name.split("/")[-1]
                    images = add_image(images, width, height, image_id,
                                       image_name)

                category_id = self.category_name_to_id[category_name]

                annotation_id = len(annotations)

                instance_id = label.get('instanceId', 0)

                if instance_id not in instance_annotations:
                    instance_annotations[instance_id] = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'keypoints': [],
                        'ignore': 0,
                        'iscrowd': 0,
                        'bbox': [],
                        'area': 0,
                        'segmentation': []
                    }

                if'rectanglelabels' == label['type'] or 'labels' == label['type']:
                    x = label['value']['x']
                    y = label['value']['y']
                    w = label['value']['width']
                    h = label['value']['height']

                    x = x * label['original_width'] / 100
                    y = y * label['original_height'] / 100
                    w = w * label['original_width'] / 100
                    h = h * label['original_height'] / 100

                    instance_annotations[instance_id]['bbox'] = [x, y, w, h]
                    instance_annotations[instance_id]['area'] = w * h
                    instance_annotations[instance_id]['num_keypoints'] = len(
                        instance_annotations[instance_id]['keypoints']) // 3

                elif 'polygonlabels' == label['type']:
                    points_abs = [(x / 100 * width, y / 100 * height)
                                  for x, y in label['value']['points']]
                    x, y = zip(*points_abs)

                    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)

                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    area = float(0.5 * np.abs(
                        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

                    instance_annotations[instance_id]['segmentation'] = [[
                        coord for point in points_abs for coord in point
                    ]]
                    instance_annotations[instance_id]['bbox'] = bbox
                    instance_annotations[instance_id]['area'] = area
                    instance_annotations[instance_id]['num_keypoints'] = len(
                        instance_annotations[instance_id]['keypoints']) // 3

                elif 'keypointlabels' == label['type']:
                    x = label['value']['x'] * label['original_width'] / 100
                    y = label['value']['y'] * label['original_height'] / 100

                    if x == y == 0:
                        current_kp = [x, y, 0]
                    else:
                        current_kp = [x, y, 2]

                    instance_annotations[instance_id]['keypoints'].extend(current_kp)
                    instance_annotations[instance_id]['num_keypoints'] = len(
                        instance_annotations[instance_id]['keypoints']) // 3

            for instance_id in instance_annotations:
                annotations.append(instance_annotations[instance_id])

        with io.open(output_json, mode='w', encoding='utf8') as fout:
            json.dump(
                {
                    'images': images,
                    'categories': self.categories,
                    'annotations': annotations,
                    'info': {
                        'year': datetime.now().year,
                        'version': '1.0',
                        'description': '',
                        'contributor': 'Label Studio',
                        'url': '',
                        'date_created': str(datetime.now()),
                    },
                },
                fout,
                indent=2,
            )
        return images, annotations

    def split_and_save(self, images, annotations, output_dir, train_ratio, val_ratio, test_ratio):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_ids = [image['id'] for image in images]
        train_ids, test_val_ids = train_test_split(image_ids, test_size=1 - train_ratio, random_state=42)
        val_ratio_relative = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(test_val_ids, test_size=1 - val_ratio_relative, random_state=42)

        def filter_data(ids):
            filtered_images = [image for image in images if image['id'] in ids]
            filtered_annotations = [ann for ann in annotations if ann['image_id'] in ids]
            return filtered_images, filtered_annotations

        train_images, train_annotations = filter_data(train_ids)
        val_images, val_annotations = filter_data(val_ids)
        test_images, test_annotations = filter_data(test_ids)

        def save_data(images, annotations, file_name):
            file_path = output_dir / file_name
            with io.open(file_path, mode='w', encoding='utf8') as fout:
                json.dump(
                    {
                        'images': images,
                        'categories': self.categories,
                        'annotations': annotations,
                        'info': {
                            'year': datetime.now().year,
                            'version': '1.0',
                            'description': '',
                            'contributor': 'Label Studio',
                            'url': '',
                            'date_created': str(datetime.now()),
                        },
                    },
                    fout,
                    indent=2,
                )

        save_data(train_images, train_annotations, 'train.json')
        save_data(val_images, val_annotations, 'val.json')
        save_data(test_images, test_annotations, 'test.json')

        print(f"Number of images in train set: {len(train_images)}")
        print(f"Number of images in val set: {len(val_images)}")
        print(f"Number of images in test set: {len(test_images)}")


def main():
    args = parse_args()
    config = args.config
    input_json = args.input
    output_json = args.output
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio

    converter = LSConverter(config)
    images, annotations = converter.convert_to_coco(input_json, output_json)
    converter.split_and_save(images, annotations, output_dir, train_ratio, val_ratio, test_ratio)


if __name__ == '__main__':
    main()
    