"""血液细胞检测系统 - 桌面应用

使用方法:
    python tools/deploy/app.py

然后在浏览器打开 http://127.0.0.1:7860
"""

import os
import sys
import time

import cv2
import gradio as gr
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mmdet.apis import init_detector, inference_detector


# === Config ===
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
CONFIG_PATH = os.path.join(
    PROJECT_ROOT, 'configs', 'rtmdet', 'bccd_rtmdet_tiny_8xb32-300e_coco.py')
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, 'work_dirs', 'bccd_rtmdet_tiny_8xb32-300e_coco', 'epoch_280.pth')

CLASSES = ('RBC', 'WBC', 'Platelets')
CLASSES_CN = {'RBC': '红细胞', 'WBC': '白细胞', 'Platelets': '血小板'}
PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


# === Load model (only once) ===
print('正在加载模型...')
model = init_detector(CONFIG_PATH, CHECKPOINT_PATH, device='cuda:0')
model.eval()
print('模型加载完成！')


def detect(image, conf_threshold):
    """对上传的图片进行检测。"""
    if image is None:
        return None, "请先上传一张图片。"

    # Convert RGB (Gradio) to BGR (OpenCV)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save temp image
    tmp_path = os.path.join(PROJECT_ROOT, 'deploy', 'tmp_input.jpg')
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    cv2.imwrite(tmp_path, img_bgr)

    # Run inference
    start = time.time()
    result = inference_detector(model, tmp_path)
    infer_time = time.time() - start

    # Draw results
    img_show = image.copy()
    pred_instances = result.pred_instances

    # Filter by confidence
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    bboxes = pred_instances.bboxes.cpu().numpy()

    mask = scores > conf_threshold
    scores = scores[mask]
    labels = labels[mask]
    bboxes = bboxes[mask]

    # Count by class
    counts = {cls: 0 for cls in CLASSES}

    for bbox, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = bbox.astype(int)
        cls_name = CLASSES[label]
        cls_cn = CLASSES_CN[cls_name]
        color = PALETTE[label]
        counts[cls_name] += 1

        # Draw bbox
        cv2.rectangle(img_show, (x1, y1), (x2, y2), color, 2)

        # Draw label (Chinese + English + score)
        text = f'{cls_cn}({cls_name}) {score:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_show, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img_show, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Build info text
    total = sum(counts.values())
    info = f"### 🔍 检测结果\n\n"
    info += f"| 类别 | 数量 |\n|------|------|\n"
    for cls_name, count in counts.items():
        emoji = '🔴' if cls_name == 'RBC' else ('🟢' if cls_name == 'WBC' else '🔵')
        cls_cn = CLASSES_CN[cls_name]
        info += f"| {emoji} {cls_cn} ({cls_name}) | **{count}** |\n"
    info += f"| 📊 **总计** | **{total}** |\n\n"
    info += f"**推理耗时:** {infer_time*1000:.1f} ms\n\n"
    info += f"**置信度阈值:** {conf_threshold}\n"

    return img_show, info


# === Gradio UI ===
title = "🔬 血液细胞智能检测系统"
description = """
基于 **RTMDet-Tiny** 深度学习模型，自动检测和分类显微镜血液涂片图像中的细胞。

**检测类别：**
- 🔴 **红细胞 (RBC)** — 数量最多的细胞类型
- 🟢 **白细胞 (WBC)** — 免疫系统的重要组成部分
- 🔵 **血小板 (Platelets)** — 参与凝血过程

**模型性能：** mAP@50 = **88.4%**
"""

with gr.Blocks(title="血液细胞检测系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📷 上传血液细胞图片",
                type="numpy",
                height=400,
            )
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.3,
                step=0.05,
                label="🎚️ 置信度阈值",
            )
            detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")

            # Example images
            example_dir = os.path.join(PROJECT_ROOT, 'date', 'blood_cell', 'images', 'test')
            examples = []
            if os.path.exists(example_dir):
                for f in sorted(os.listdir(example_dir))[:6]:
                    if f.endswith('.jpg'):
                        examples.append(os.path.join(example_dir, f))

            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=input_image,
                    label="📷 示例图片（点击使用）",
                )

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="✅ 检测结果",
                height=400,
            )
            output_info = gr.Markdown(label="📊 检测信息")

    detect_btn.click(
        fn=detect,
        inputs=[input_image, conf_slider],
        outputs=[output_image, output_info],
    )

    # Auto-detect on image upload
    input_image.change(
        fn=detect,
        inputs=[input_image, conf_slider],
        outputs=[output_image, output_info],
    )

    gr.Markdown("""
---
**项目：** 基于 RTMDet-Tiny + MMDetection v3 | **数据集：** BCCD（364张图片，3个类别）| **mAP@50：** 88.4%
    """)


if __name__ == '__main__':
    print('\n' + '=' * 50)
    print('血液细胞智能检测系统')
    print('浏览器访问: http://127.0.0.1:7860')
    print('=' * 50 + '\n')

    demo.launch(
        server_name='127.0.0.1',
        server_port=7860,
        share=False,
        inbrowser=True,
    )
