"""Benchmark PyTorch vs ONNX inference speed and accuracy.

Usage:
    python tools/deploy/benchmark.py

Output:
    Console output with comparison table.
"""

import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def benchmark_pytorch(config_path, checkpoint_path, img_dir, warmup=3, repeats=10):
    """Benchmark PyTorch inference speed."""
    from mmdet.apis import init_detector, inference_detector

    model = init_detector(config_path, checkpoint_path, device='cuda:0')
    model.eval()

    # Get test images
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                 if f.endswith(('.jpg', '.png'))][:5]

    # Warmup
    print(f'  PyTorch warmup ({warmup} runs)...')
    for img_path in img_files[:1]:
        for _ in range(warmup):
            with torch.no_grad():
                inference_detector(model, img_path)

    # Benchmark
    print(f'  PyTorch benchmark ({repeats} runs per image)...')
    times = []
    for img_path in img_files:
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                inference_detector(model, img_path)
            torch.cuda.synchronize()
            times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    fps = 1000 / avg_time
    print(f'  PyTorch: {avg_time:.1f} ms/frame, {fps:.1f} FPS')

    return avg_time, fps


def benchmark_onnx(onnx_path, img_dir, warmup=3, repeats=10):
    """Benchmark ONNX Runtime inference speed."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Get test images
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                 if f.endswith(('.jpg', '.png'))][:5]

    # Preprocess all images
    def preprocess(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
        img = (img.astype(np.float32) - mean) / std
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        return img.astype(np.float32)

    # Warmup
    print(f'  ONNX warmup ({warmup} runs)...')
    dummy = preprocess(img_files[0])
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # Benchmark
    print(f'  ONNX benchmark ({repeats} runs per image)...')
    times = []
    for img_path in img_files:
        img_input = preprocess(img_path)
        for _ in range(repeats):
            start = time.time()
            session.run(None, {input_name: img_input})
            times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    fps = 1000 / avg_time
    print(f'  ONNX FP32: {avg_time:.1f} ms/frame, {fps:.1f} FPS')

    return avg_time, fps


def benchmark_onnx_fp16(onnx_path, img_dir, warmup=3, repeats=10):
    """Benchmark ONNX Runtime FP16 inference speed."""
    import onnxruntime as ort
    from onnxruntime.transformers import float16

    # Load and convert to FP16
    import onnx
    model = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model)

    # Save FP16 model
    fp16_path = onnx_path.replace('.onnx', '_fp16.onnx')
    onnx.save(model_fp16, fp16_path)

    session = ort.InferenceSession(
        fp16_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Get test images
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                 if f.endswith(('.jpg', '.png'))][:5]

    def preprocess(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
        img = (img.astype(np.float32) - mean) / std
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        return img.astype(np.float16)

    # Warmup
    print(f'  ONNX FP16 warmup ({warmup} runs)...')
    dummy = preprocess(img_files[0])
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # Benchmark
    print(f'  ONNX FP16 benchmark ({repeats} runs per image)...')
    times = []
    for img_path in img_files:
        img_input = preprocess(img_path)
        for _ in range(repeats):
            start = time.time()
            session.run(None, {input_name: img_input})
            times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    fps = 1000 / avg_time
    print(f'  ONNX FP16: {avg_time:.1f} ms/frame, {fps:.1f} FPS')

    return avg_time, fps


def print_table(results):
    """Print comparison table."""
    print('\n' + '=' * 60)
    print('Performance Benchmark Results')
    print('=' * 60)
    print(f'{"Method":<20} {"Time (ms)":<15} {"FPS":<15} {"Speedup":<10}')
    print('-' * 60)

    base_time = results[0][1]
    for name, avg_time, fps in results:
        speedup = base_time / avg_time
        print(f'{name:<20} {avg_time:<15.1f} {fps:<15.1f} {speedup:<10.2f}x')
    print('=' * 60)


if __name__ == '__main__':
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')

    config_path = os.path.join(
        project_root, 'configs', 'rtmdet', 'bccd_rtmdet_tiny_8xb32-300e_coco.py')
    checkpoint_path = os.path.join(
        project_root, 'work_dirs', 'bccd_rtmdet_tiny_8xb32-300e_coco', 'epoch_280.pth')
    onnx_path = os.path.join(project_root, 'deploy', 'bccd_rtmdet.onnx')
    img_dir = os.path.join(project_root, 'date', 'blood_cell', 'images', 'test')

    results = []

    print('=' * 60)
    print('Benchmarking...')
    print('=' * 60)

    # PyTorch
    print('\n[1/3] PyTorch Inference')
    t, f = benchmark_pytorch(config_path, checkpoint_path, img_dir)
    results.append(('PyTorch (FP32)', t, f))

    # ONNX FP32
    print('\n[2/3] ONNX FP32 Inference')
    t, f = benchmark_onnx(onnx_path, img_dir)
    results.append(('ONNX (FP32)', t, f))

    # ONNX FP16
    print('\n[3/3] ONNX FP16 Inference')
    t, f = benchmark_onnx_fp16(onnx_path, img_dir)
    results.append(('ONNX (FP16)', t, f))

    print_table(results)
