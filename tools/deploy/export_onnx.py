"""Export RTMDet model to ONNX format.

Usage:
    python tools/deploy/export_onnx.py

Output:
    deploy/bccd_rtmdet.onnx
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mmdet.apis import init_detector
from mmengine.config import Config


def export_onnx(config_path, checkpoint_path, output_path, img_size=640):
    """Export mmdet model to ONNX format.

    Args:
        config_path: Path to config file.
        checkpoint_path: Path to checkpoint file.
        output_path: Path to save ONNX model.
        img_size: Input image size.
    """
    print(f'Loading model from {checkpoint_path}...')
    model = init_detector(config_path, checkpoint_path, device='cuda:0')
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size).cuda()

    # Get model from runner
    pytorch_model = model

    print(f'Exporting to ONNX...')
    print(f'  Input shape: {dummy_input.shape}')
    print(f'  Output: {output_path}')

    # Export using torch.onnx
    # We need to create a wrapper that handles the forward pass
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # Extract backbone features
            feat = self.model.backbone(x)
            # Extract neck features
            feat = self.model.neck(feat)
            # Get predictions from head
            # The head returns cls_scores and bbox_preds
            cls_scores, bbox_preds = self.model.bbox_head(feat)
            # Concatenate outputs
            # cls_scores: list of [B, num_classes, H, W] at each level
            # bbox_preds: list of [B, 4, H, W] at each level
            return cls_scores, bbox_preds

    wrapper = ONNXWrapper(pytorch_model).cuda()
    wrapper.eval()

    # Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=['input'],
        output_names=['cls_scores', 'bbox_preds'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'cls_scores': {0: 'batch_size'},
            'bbox_preds': {0: 'batch_size'},
        },
    )

    # Verify
    import onnx
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print(f'\nONNX model saved to: {output_path}')
    print(f'Model size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB')

    return output_path


if __name__ == '__main__':
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')

    config_path = os.path.join(
        project_root, 'configs', 'rtmdet', 'bccd_rtmdet_tiny_8xb32-300e_coco.py')
    checkpoint_path = os.path.join(
        project_root, 'work_dirs', 'bccd_rtmdet_tiny_8xb32-300e_coco', 'epoch_280.pth')
    output_path = os.path.join(
        project_root, 'deploy', 'bccd_rtmdet.onnx')

    export_onnx(config_path, checkpoint_path, output_path)
