"""ONNX Runtime inference for RTMDet blood cell detection.

Usage:
    python tools/deploy/onnx_inference.py --image date/blood_cell/images/test/BloodImage_00007.jpg

Output:
    deploy/results/onnx_result_*.jpg
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mmdet.apis import init_detector
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


CLASSES = ('RBC', 'WBC', 'Platelets')
PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def preprocess(img_path, img_size=640):
    """Preprocess image for ONNX inference."""
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]

    # Resize
    img_resized = cv2.resize(img, (img_size, img_size))

    # Normalize (same as mmdet config)
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    img_norm = (img_resized.astype(np.float32) - mean) / std

    # HWC -> CHW, add batch dim
    img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

    return img_input.astype(np.float32), img, (orig_w, orig_h)


def decode_predictions(cls_scores, bbox_preds, img_size=640,
                       conf_thr=0.3, iou_thr=0.45):
    """Decode raw model outputs to bounding boxes."""
    # Generate anchors for each FPN level
    strides = [8, 16, 32]
    anchors_all = []
    for stride in strides:
        h = img_size // stride
        w = img_size // stride
        yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        anchor_x = (xv.float() + 0.5) * stride
        anchor_y = (yv.float() + 0.5) * stride
        anchors = torch.stack([anchor_x, anchor_y], dim=-1).reshape(-1, 2)
        anchors_all.append(anchors)
    anchors_all = torch.cat(anchors_all, dim=0).cuda()  # [N, 2]

    # Decode predictions from each level
    all_boxes = []
    all_scores = []

    for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
        # cls_score: [1, C, H, W] -> [H*W, C]
        # bbox_pred: [1, 4, H, W] -> [H*W, 4]
        B, C, H, W = cls_score.shape
        cls_score = cls_score.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C)
        bbox_pred = bbox_pred.reshape(B, 4, -1).permute(0, 2, 1).reshape(-1, 4)

        # Apply sigmoid to cls_score (QualityFocalLoss uses sigmoid)
        cls_score = cls_score.sigmoid()

        all_boxes.append(bbox_pred)
        all_scores.append(cls_score)

    # Concatenate all levels
    all_boxes = torch.cat(all_boxes, dim=0)  # [N, 4]
    all_scores = torch.cat(all_scores, dim=0)  # [N, num_classes]

    # Filter by confidence
    max_scores, labels = all_scores.max(dim=1)
    mask = max_scores > conf_thr
    all_boxes = all_boxes[mask]
    max_scores = max_scores[mask]
    labels = labels[mask]
    anchor_indices = torch.where(mask)[0]

    if len(all_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Decode boxes: distance -> xyxy
    # bbox_pred format: [left, top, right, bottom] (distances from anchor)
    anchors = anchors_all[anchor_indices]
    x1 = anchors[:, 0] - all_boxes[:, 0]
    y1 = anchors[:, 1] - all_boxes[:, 1]
    x2 = anchors[:, 0] + all_boxes[:, 2]
    y2 = anchors[:, 1] + all_boxes[:, 3]
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # Scale to image size
    boxes = boxes.clamp(0, img_size)

    # NMS per class
    from torchvision.ops import batched_nms
    keep = batched_nms(boxes, max_scores, labels, iou_thr)
    boxes = boxes[keep].cpu().numpy()
    scores = max_scores[keep].cpu().numpy()
    label_ids = labels[keep].cpu().numpy()

    return boxes, scores, label_ids


def draw_results(img, boxes, scores, label_ids, orig_size, img_size=640):
    """Draw detection results on image."""
    orig_w, orig_h = orig_size
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size

    for box, score, lid in zip(boxes, scores, label_ids):
        x1, y1, x2, y2 = box
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        color = PALETTE[lid]
        label = f'{CLASSES[lid]} {score:.2f}'

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


def run_inference(onnx_path, img_path, output_dir, conf_thr=0.3):
    """Run ONNX inference on a single image."""
    print(f'Loading ONNX model: {onnx_path}')
    session = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    print(f'Processing: {img_path}')
    img_input, img_orig, orig_size = preprocess(img_path)

    # Run inference
    start = time.time()
    outputs = session.run(None, {input_name: img_input})
    infer_time = time.time() - start
    print(f'  Inference time: {infer_time*1000:.1f} ms')

    # Decode outputs
    cls_scores = [torch.from_numpy(o).cuda() for o in outputs[:3]]
    bbox_preds = [torch.from_numpy(o).cuda() for o in outputs[3:]]

    boxes, scores, label_ids = decode_predictions(
        cls_scores, bbox_preds, conf_thr=conf_thr)

    print(f'  Detected {len(boxes)} objects')
    for i, cls_name in enumerate(CLASSES):
        count = (label_ids == i).sum()
        if count > 0:
            print(f'    {cls_name}: {count}')

    # Draw results
    result_img = draw_results(img_orig.copy(), boxes, scores, label_ids, orig_size)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, f'onnx_{basename}.jpg')
    cv2.imwrite(output_path, result_img)
    print(f'  Saved: {output_path}')

    return boxes, scores, label_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--onnx', type=str, default=None,
                        help='ONNX model path')
    parser.add_argument('--out-dir', type=str, default='deploy/results',
                        help='Output directory')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold')
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), '..', '..')

    if args.onnx is None:
        args.onnx = os.path.join(project_root, 'deploy', 'bccd_rtmdet.onnx')

    run_inference(args.onnx, args.image, args.out_dir, args.conf)
