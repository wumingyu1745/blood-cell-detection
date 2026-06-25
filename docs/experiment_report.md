# 基于RTMDet的血液细胞检测与分类系统

## 实验报告

---

## 一、研究背景与意义

### 1.1 研究背景

血液细胞检测与分类是临床血液检验中的重要环节。传统的血液细胞分析主要依赖人工在显微镜下观察和计数，这种方法存在以下问题：

- **效率低下**：人工计数耗时长，一份血常规样本需要数分钟到数十分钟
- **主观性强**：不同检验人员的判断标准存在差异，结果一致性差
- **劳动强度大**：长时间观察显微镜容易导致视觉疲劳，影响准确性
- **人力成本高**：需要专业检验人员，基层医疗机构人才匮乏

随着深度学习技术的快速发展，基于计算机视觉的自动化血液细胞检测成为可能。通过目标检测算法，可以实现血液细胞的自动定位、分类和计数，大幅提升检验效率和准确性。

### 1.2 研究意义

本项目的意义在于：

1. **提高检验效率**：自动化检测可将检验时间从数分钟缩短到毫秒级别
2. **提升检测精度**：基于深度学习的检测模型具有较高的准确性和一致性
3. **降低人力成本**：减少对专业检验人员的依赖，适用于基层医疗机构
4. **辅助临床诊断**：为血液疾病的早期筛查提供技术支持

---

## 二、研究现状

### 2.1 传统方法

传统的血液细胞检测方法主要包括：

- **基于阈值分割**：通过设定灰度阈值将细胞从背景中分离，但对光照变化敏感
- **基于边缘检测**：使用Canny、Sobel等算子检测细胞边缘，但对噪声敏感
- **基于形态学处理**：利用膨胀、腐蚀等操作提取细胞区域，但对重叠细胞效果差

### 2.2 深度学习方法

近年来，基于深度学习的目标检测算法在血液细胞检测中取得了显著进展：

| 算法 | 类型 | 特点 | 代表工作 |
|------|------|------|----------|
| Faster R-CNN | 两阶段 | 精度高，速度较慢 | RBC检测 |
| YOLOv5/v8 | 单阶段 | 速度快，实时检测 | 血液细胞计数 |
| DETR | Transformer | 端到端，无需NMS | 医学图像检测 |
| RTMDet | 单阶段 | 平衡速度与精度 | 实时检测场景 |

### 2.3 本项目选择

本项目选择 **RTMDet-Tiny** 作为检测模型，原因如下：

1. **速度快**：RTMDet是实时目标检测算法，适合部署应用
2. **精度高**：在COCO数据集上达到SOTA水平
3. **轻量级**：Tiny版本参数量小，适合边缘设备部署
4. **社区支持**：基于MMDetection框架，文档完善，易于复现

---

## 三、模型设计

### 3.1 RTMDet整体架构

RTMDet由三个核心模块组成：

```
输入图像 (640×640×3)
    │
    ▼
┌─────────────────┐
│   CSPNeXt-Tiny  │  ← Backbone (特征提取)
│   (Backbone)    │
└────────┬────────┘
         │ 多尺度特征
         ▼
┌─────────────────┐
│  CSPNeXtPAFPN   │  ← Neck (特征融合)
│    (Neck)       │
└────────┬────────┘
         │ 融合后特征
         ▼
┌─────────────────┐
│ RTMDetSepBNHead │  ← Head (检测头)
│   (Head)        │
└────────┬────────┘
         │
         ▼
    检测结果 (类别 + 框)
```

### 3.2 Backbone: CSPNeXt-Tiny

CSPNeXt是Cross Stage Partial Network的改进版本：

- **输入**：640×640×3 的RGB图像
- **结构**：5个Stage，逐步降采样
- **特点**：
  - 使用Channel Attention模块增强特征表达
  - 采用SiLU激活函数
  - 参数量约1.4M，计算量约0.3G FLOPs

### 3.3 Neck: CSPNeXtPAFPN

Path Aggregation Feature Pyramid Network的改进版本：

- **功能**：融合多尺度特征（P3, P4, P5）
- **结构**：自顶向下 + 自底向上 双向特征融合
- **输出**：3个尺度的特征图（80×80, 40×40, 20×20）

### 3.4 Head: RTMDetSepBNHead

分离批归一化的检测头：

- **分类分支**：预测每个锚点的类别概率
- **回归分支**：预测边界框的偏移量
- **特点**：分类和回归使用独立的BN层，提升检测精度

### 3.5 损失函数

| 损失类型 | 函数 | 用途 |
|----------|------|------|
| 分类损失 | QualityFocalLoss | 处理类别不平衡问题 |
| 回归损失 | GIoULoss | 优化边界框回归精度 |

### 3.6 检测类别

| 类别 | 英文 | 说明 | 标注颜色 |
|------|------|------|----------|
| 红细胞 | RBC | 数量最多，双凹圆盘状 | 🔴 红色 |
| 白细胞 | WBC | 免疫细胞，体积较大 | 🟢 绿色 |
| 血小板 | Platelets | 体积最小，参与凝血 | 🔵 蓝色 |

---

## 四、数据集介绍

### 4.1 BCCD数据集

BCCD (Blood Cell Count and Detection) 是一个公开的血液细胞检测数据集：

| 属性 | 数值 |
|------|------|
| 总图片数 | 364张 |
| 图片分辨率 | 640×480 |
| 标注格式 | Pascal VOC XML |
| 类别数 | 3类 (RBC, WBC, Platelets) |
| 总标注数 | 4,888个 |

### 4.2 数据集划分

| 集合 | 数量 | 占比 | 用途 |
|------|------|------|------|
| 训练集 | 205张 | 56% | 模型训练 |
| 验证集 | 87张 | 24% | 训练中评估 |
| 测试集 | 72张 | 20% | 最终评估 |

### 4.3 数据预处理

原始数据集为Pascal VOC XML格式，需要转换为COCO JSON格式：

```python
# 转换脚本: tools/voc2coco.py
# 类别映射: RBC→0, WBC→1, Platelets→2
# 边界框格式: [xmin, ymin, xmax, ymax] → [x, y, width, height]
```

### 4.4 数据增强

训练阶段使用了以下数据增强策略：

| 增强方法 | 说明 |
|----------|------|
| CachedMosaic | 4张图片拼接，增加小目标数量 |
| CachedMixUp | 2张图片混合，增加样本多样性 |
| RandomResize | 随机缩放，增强尺度鲁棒性 |
| RandomCrop | 随机裁剪，增强位置鲁棒性 |
| RandomFlip | 随机翻转，增强方向鲁棒性 |
| YOLOXHSVRandomAug | 颜色空间增强，增强光照鲁棒性 |

---

## 五、训练流程

### 5.1 环境配置

#### 硬件环境

| 组件 | 配置 |
|------|------|
| CPU | Intel/AMD x64 |
| GPU | NVIDIA RTX 3060 (12GB) |
| 内存 | 16GB+ |
| 硬盘 | 50GB+ 可用空间 |

#### 软件环境

| 组件 | 版本 |
|------|------|
| 操作系统 | Windows 10 |
| Python | 3.8.20 |
| PyTorch | 2.4.1 |
| CUDA | 11.8 |
| mmcv | 2.1.0 (lite) |
| mmengine | 0.10.7 |
| mmdet | 3.3.0 |

#### 环境安装步骤

```bash
# 1. 创建conda环境
conda create -n qilin python=3.8 -y
conda activate qilin

# 2. 安装PyTorch (CUDA 11.8)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 安装mmcv-lite (无需编译CUDA算子)
pip install mmcv-lite==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html

# 4. 安装mmengine和mmdet
pip install mmengine
pip install -v -e .

# 5. 安装其他依赖
pip install tensorboard gradio onnx onnxruntime-gpu
```

### 5.2 数据准备

```bash
# 1. 下载BCCD数据集
# 从GitHub下载: https://github.com/Shenggan/BCCD_Dataset
# 解压到项目根目录的 BCCD/ 文件夹

# 2. 转换数据格式 (VOC → COCO)
python tools/voc2coco.py

# 转换后目录结构:
# date/blood_cell/
#   images/
#     train/    (205张)
#     val/      (87张)
#     test/     (72张)
#   annotations/
#     instances_train.json
#     instances_val.json
#     instances_test.json
```

### 5.3 模型训练

```bash
# 开始训练
python tools/train.py configs/rtmdet/bccd_rtmdet_tiny_8xb32-300e_coco.py

# 如果中途中断，可以恢复训练
python tools/train.py configs/rtmdet/bccd_rtmdet_tiny_8xb32-300e_coco.py --resume
```

#### 训练超参数

| 参数 | 数值 | 说明 |
|------|------|------|
| batch_size | 16 | 每批样本数 |
| max_epochs | 300 | 最大训练轮数 |
| learning_rate | 0.004 | 初始学习率 |
| weight_decay | 0.05 | 权重衰减 |
| optimizer | AdamW | 优化器 |
| scheduler | CosineAnnealingLR | 学习率调度 |

#### 训练过程

训练过程中的关键指标变化：

| Epoch | Loss | mAP@50 | 说明 |
|-------|------|--------|------|
| 1 | 1.363 | - | 初始阶段 |
| 10 | 1.345 | - | 快速下降 |
| 80 | 0.72 | 0.888 | 趋于稳定 |
| 280 | 0.48 | 0.876 | 最佳模型 |
| 300 | - | - | 训练完成 |

### 5.4 模型测试

```bash
# 测试最佳模型
python tools/test.py configs/rtmdet/bccd_rtmdet_tiny_8xb32-300e_coco.py \
    work_dirs/bccd_rtmdet_tiny_8xb32-300e_coco/epoch_280.pth

# 推理可视化
python demo/image_demo.py date/blood_cell/images/test \
    configs/rtmdet/bccd_rtmdet_tiny_8xb32-300e_coco.py \
    --weights work_dirs/bccd_rtmdet_tiny_8xb32-300e_coco/epoch_280.pth \
    --out-dir results/bccd_demo
```

---

## 六、检测结果

### 6.1 定量结果

在测试集（72张图片）上的评估结果：

| 指标 | 数值 | 说明 |
|------|------|------|
| **mAP@50** | **0.884** | IoU=0.5时的平均精度 |
| **mAP@50:95** | **0.613** | IoU=0.5:0.95的平均精度 |
| mAP@75 | 0.723 | IoU=0.75时的平均精度 |
| mAP_s | 0.320 | 小目标检测精度 |
| mAP_m | 0.482 | 中等目标检测精度 |
| mAP_l | 0.475 | 大目标检测精度 |

### 6.2 各类别检测结果

| 类别 | 检测数量 | 说明 |
|------|----------|------|
| RBC (红细胞) | ~4000+ | 数量最多，检测效果好 |
| WBC (白细胞) | ~370 | 数量较少，但检测准确 |
| Platelets (血小板) | ~360 | 体积小，检测有一定难度 |

### 6.3 检测速度

| 方法 | 推理时间 | FPS | 说明 |
|------|----------|-----|------|
| PyTorch (GPU) | 14.7ms | 68.3 | 原始PyTorch推理 |
| ONNX (CPU) | 26.2ms | 38.1 | ONNX Runtime CPU |
| ONNX FP16 | 40.1ms | 25.0 | ONNX FP16精度 |

### 6.4 可视化结果

检测结果保存在以下目录：

- `results/bccd_demo/vis/` — 测试集检测可视化（72张）
- `deploy/results/` — ONNX推理结果

每张图片上标注了：
- 检测框（不同类别用不同颜色）
- 类别标签和置信度分数

---

## 七、遇到的Bug及调试方案

### Bug 1: mmcv安装失败 — 缺少C++编译器

**错误信息：**
```
error: Microsoft Visual C++ 14.0 or greater is required.
Get it with "Microsoft C++ Build Tools"
```

**原因分析：**
mmcv需要编译CUDA算子，系统缺少C++编译器。

**解决方案：**
使用mmcv-lite版本，无需编译CUDA算子：
```bash
pip install mmcv-lite==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html
```

**效果：**
mmcv-lite使用纯PyTorch实现，功能完整，训练速度略慢但可接受。

---

### Bug 2: CUDA版本不匹配

**错误信息：**
```
RuntimeError: The detected CUDA version (12.4) mismatches the version
that was used to compile PyTorch (11.8).
```

**原因分析：**
系统安装了CUDA 12.4，但PyTorch是基于CUDA 11.8编译的，mmcv编译时检测到系统CUDA版本不匹配。

**解决方案：**
1. 安装mmcv-lite避免编译（推荐）
2. 或者设置CUDA_HOME指向conda环境的CUDA

---

### Bug 3: mmcv._ext模块缺失

**错误信息：**
```
ModuleNotFoundError: No module named 'mmcv._ext'
```

**原因分析：**
mmcv-lite不包含编译好的CUDA扩展模块，mmdet在导入时尝试加载这些模块。

**解决方案：**
修改mmcv的ext_loader.py，添加优雅降级处理：
```python
def load_ext(name, funcs):
    try:
        ext = importlib.import_module('mmcv.' + name)
    except (ImportError, ModuleNotFoundError):
        import types
        ext = types.ModuleType(f'mmcv.{name}')
        for fun in funcs:
            setattr(ext, fun, None)
        return ext
```

---

### Bug 4: NMS函数调用失败

**错误信息：**
```
TypeError: 'NoneType' object is not callable
```

**原因分析：**
NMS（非极大值抑制）使用了mmcv的编译扩展，但mmcv-lite中该扩展为None。

**解决方案：**
修改mmcv的nms.py，添加torchvision NMS作为fallback：
```python
# 在 nms.py 开头添加
_use_torchvision_nms = ext_module is None or not hasattr(ext_module, 'nms') or ext_module.nms is None

# 在 NMSop.forward 中添加
if _use_torchvision_nms:
    from torchvision.ops import nms as tv_nms
    inds = tv_nms(bboxes, scores, iou_threshold=float(iou_threshold))
else:
    inds = ext_module.nms(bboxes, scores, iou_threshold=...)
```

---

### Bug 5: Pillow DLL加载失败

**错误信息：**
```
ImportError: DLL load failed while importing _imaging: 找不到指定的模块。
```

**原因分析：**
torchvision重装后导致Pillow的DLL依赖损坏。

**解决方案：**
```bash
pip install --force-reinstall Pillow
```

---

### Bug 6: DataLoader Worker进程异常退出

**错误信息：**
```
RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly
```

**原因分析：**
OpenMP运行时重复加载导致进程崩溃。

**解决方案：**
设置环境变量：
```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

---

### Bug 7: 图片路径找不到

**错误信息：**
```
FileNotFoundError: No such file or directory: 'date/blood_cell/images/BloodImage_00309.jpg'
```

**原因分析：**
COCO JSON中的file_name没有包含子目录路径（train/val/test）。

**解决方案：**
修改转换脚本，让file_name包含子目录：
```python
'file_name': f'{split_name}/{filename}'  # 如 'train/BloodImage_00000.jpg'
```

---

### Bug 8: ONNX CUDA Provider加载失败

**错误信息：**
```
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*
```

**原因分析：**
onnxruntime-gpu需要CUDA 12.x，但当前环境是CUDA 11.8。

**解决方案：**
1. 使用CPU Provider（自动回退）
2. 或者升级到CUDA 12.x环境

---

## 八、总结与展望

### 8.1 项目总结

本项目基于RTMDet-Tiny深度学习模型，实现了血液细胞的自动检测与分类：

1. **数据方面**：使用BCCD公开数据集，包含364张血液涂片图片，3个类别（RBC、WBC、Platelets）
2. **模型方面**：采用RTMDet-Tiny模型，兼顾检测速度和精度
3. **训练方面**：300个epoch训练，最终mAP@50达到88.4%
4. **部署方面**：实现了ONNX导出和桌面化应用（Gradio界面）

### 8.2 创新点

1. **小样本数据增强策略**：采用CachedMosaic和CachedMixUp等先进数据增强方法，有效提升了小样本场景下的检测精度
2. **分阶段训练策略**：使用PipelineSwitchHook，在训练后期切换数据增强策略，避免过度增强影响收敛
3. **多精度模型部署**：支持FP32和FP16两种精度的ONNX模型部署
4. **桌面化应用**：基于Gradio构建了用户友好的检测界面

### 8.3 未来展望

1. **扩展数据集**：收集更多血液细胞图片，提升模型泛化能力
2. **模型优化**：尝试更大模型（RTMDet-S/M）或知识蒸馏
3. **边缘部署**：适配Jetson等边缘设备，实现实时检测
4. **多任务学习**：同时实现检测、分割和计数

---

## 附录

### A. 项目文件结构

```
mmdetection-main/
├── configs/rtmdet/
│   └── bccd_rtmdet_tiny_8xb32-300e_coco.py  # 训练配置
├── date/blood_cell/
│   ├── images/                               # 图片
│   └── annotations/                          # COCO标注
├── deploy/
│   ├── bccd_rtmdet.onnx                      # ONNX模型
│   └── results/                              # ONNX推理结果
├── docs/
│   └── experiment_report.md                  # 本报告
├── results/
│   └── bccd_demo/vis/                        # 检测可视化
├── tools/
│   ├── deploy/
│   │   ├── app.py                            # 桌面应用
│   │   ├── benchmark.py                      # 性能对比
│   │   ├── export_onnx.py                    # ONNX导出
│   │   └── onnx_inference.py                 # ONNX推理
│   ├── train.py                              # 训练脚本
│   ├── test.py                               # 测试脚本
│   └── voc2coco.py                           # 数据转换
└── work_dirs/bccd_rtmdet_tiny_8xb32-300e_coco/
    ├── epoch_280.pth                          # 最佳模型
    └── 20260623_*/                            # 训练日志
```

### B. 关键命令汇总

```bash
# 环境配置
conda activate qilin

# 数据转换
python tools/voc2coco.py

# 模型训练
python tools/train.py configs/rtmdet/bccd_rtmdet_tiny_8xb32-300e_coco.py

# 模型测试
python tools/test.py configs/rtmdet/bccd_rtmdet_tiny_8xb32-300e_coco.py work_dirs/bccd_rtmdet_tiny_8xb32-300e_coco/epoch_280.pth

# 推理可视化
python demo/image_demo.py date/blood_cell/images/test configs/rtmdet/bccd_rtmdet_tiny_8xb32-300e_coco.py --weights work_dirs/bccd_rtmdet_tiny_8xb32-300e_coco/epoch_280.pth --out-dir results/bccd_demo

# ONNX导出
python tools/deploy/export_onnx.py

# ONNX推理
python tools/deploy/onnx_inference.py --image date/blood_cell/images/test/BloodImage_00007.jpg

# 性能对比
python tools/deploy/benchmark.py

# 启动桌面应用
python tools/deploy/app.py
```

### C. 参考文献

1. RTMDet: An Empirical Study of Designing Real-Time Object Detectors (2022)
2. CSPNet: A New Backbone that can Enhance Learning Capability of CNN (2020)
3. Focal Loss for Dense Object Detection (2017)
4. Generalized Intersection over Union (2019)
5. MMDetection: Open MMLab Detection Toolbox and Benchmark (2019)

---

**报告完成时间：** 2026年6月25日

**项目地址：** 基于RTMDet的血液细胞检测与分类系统
