# 📸 RetinaFace: Single-stage Dense Face Localisation in the Wild

[![Downloads](https://img.shields.io/github/downloads/yakhyo/retinaface-pytorch/total)](https://github.com/yakhyo/retinaface-pytorch/releases)
[![GitHub Repo stars](https://img.shields.io/github/stars/yakhyo/retinaface-pytorch)](https://github.com/yakhyo/retinaface-pytorch/stargazers)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/retinaface-pytorch)
[![GitHub License](https://img.shields.io/github/license/yakhyo/retinaface-pytorch)](https://github.com/yakhyo/retinaface-pytorch/blob/main/LICENSE)

<video controls autoplay loop src="https://github.com/user-attachments/assets/ad279fea-33fb-43f1-884f-282e6d54c809" muted="false" width="100%"></video>

This is a face detection model for high-precision facial localization based on [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641). This model accurately detects facial landmarks and bounding boxes for faces in images. This repository provides custom training & inference code, and several new backbone models have been integrated for improved performance and flexibility.

> [!NOTE]  
> We've updated the codebase with new trained models and a refactored structure, enhancing functionality and maintainability. These improvements include support for MobileNetV1 (including v1_025 and v1_050), MobileNetV2, and various ResNet versions (18, 34, 50), offering a cleaner and more reproducible experience.

<div align="center">
<img src="assets/mv2_test.jpg">
</div>

In this implementation, we use several lightweight and powerful backbone architectures to provide flexibility between performance and accuracy.

## 📈 Results on WiderFace Eval

### Multi-scale Image Size

| Models                                  | Pretrained on ImageNet | Easy   | Medium | Hard   |
| --------------------------------------- | ---------------------- | ------ | ------ | ------ |
| RetinaFace-MobileNetV1(width mult=0.25) | True                   | 88.48% | 87.02% | 80.61% |
| Slim                                    | False                  | 88.04% | 85.47% | 55.40% |
| RFB                                     | False                  | 90.59% | 89.14% | 84.13% |

### Original Image Size

| Models                                  | Pretrained on ImageNet | Easy   | Medium | Hard   |
| --------------------------------------- | ---------------------- | ------ | ------ | ------ |
| RetinaFace-MobileNetV1(width mult=0.25) | True                   | %      | %      | %      |
| Slim                                    | False                  | 88.04% | 85.47% | 55.40% |
| RFB                                     | False                  | %      | %      | %      |

## ✨ Features

- ✅ **Cleaner & Reproducible Code**: Refactored for simplicity and consistency, making it easier to use and maintain.
- 📱 **MobileNetV1_0.25 & MobileNetV1_0.50**: Lightweight versions for faster inference with reduced computational cost.
- 📲 **MobileNetV1**: [Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) - Optimized for mobile and low-power applications.
- 📈 **MobileNetV2**: [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) - Improved efficiency for mobile use-cases with advanced architecture.
- 🔍 **ResNet Models (18, 34, 50)**: [Deep Residual Networks](https://arxiv.org/abs/1512.03385) - Enhanced accuracy with deeper residual connections, supporting a range of model complexities.

### 📅 Last Updated: November 5, 2024

## ⚙️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yakhyo/retinaface-pytorch.git
   cd retinaface-pytorch
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔄 Backbones

This RetinaFace implementation supports the following feature extractor backbones:

- 🟢 **MobileNetV1**: Lightweight and fast, suitable for mobile and embedded devices.
- 🟠 **MobileNetV1_0.25 & 0.50**: Variants of MobileNetV1 with reduced width multipliers for faster inference.
- 🔵 **MobileNetV2**: Improved version of MobileNetV1 with better accuracy.
- 🟣 **ResNet18/34/50**: A range of ResNet models providing a balance between complexity and performance.

## 📂 Dataset

### 📥 Download the WIDERFACE Dataset

1. **Download the Dataset**:

   - Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.
   - Download annotations (face bounding boxes & five facial landmarks) from [Baidu Cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) (password: `fstq`) or [Dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0).

2. **Organize the Dataset Directory**:

   Structure your dataset directory as follows:

   ```
   data/
   └── widerface/
      ├── train/
      │   ├── images/
      │   └── label.txt
      └── val/
         ├── images/
         └── wider_val.txt
   ```

> [!NOTE]  
> `wider_val.txt` only includes val file names but not label information.

There is also an organized dataset (as shown above): Link from [Google Drive](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [Baidu Cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) _(password: ruck)_. Thanks to [biubug6](https://github.com/biubug6) for the organized dataset.

## 🏋️‍♂️ Training

To train the RetinaFace model with a specific backbone, use the following command:

```bash
python train.py --network mobilenetv1  # Replace 'mobilenetv1' with your choice of backbone
```

Download [mobilenetv1_0.25.pretrained](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/mobilenetv1_025.pretrained) (pre-trained weights on ImageNet, weights ported from @biubug6) to reproduce the results.

### 🎛️ Available Backbone Options:

- `mobilenetv1_0.25`
- `mobilenetv1_0.50`
- `mobilenetv1`
- `mobilenetv2`
- `resnet18`
- `resnet34`
- `resnet50`

### ⬇️ Get Pretrained Models

#### MobileNet-based models:

| Model Name      | PyTorch Weights                                                                                                          | ONNX Weights Filename                                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| MobileNetV1_025 | [retinaface_mv1_0.25.pth](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.25.pth) | [retinaface_mv1_0.25.onnx](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.25.onnx) |
| MobileNetV1_050 | [retinaface_mv1_0.50.pth](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.50.pth) | [retinaface_mv1_0.50.onnx](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.50.onnx) |
| MobileNetV1     | [retinaface_mv1.pth](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1.pth)           | [retinaface_mv1.onnx](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1.onnx)           |
| MobileNetV2     | [retinaface_mv2.pth](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv2.pth)           | [retinaface_mv2.onnx](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv2.onnx)           |

#### ResNet-based models:

| Model Name | PyTorch Weights                                                                                                | ONNX Weights Filename                                                                                            |
| ---------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ResNet18   | [retinaface_r18.pth](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r18.pth) | [retinaface_r18.onnx](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r18.onnx) |
| ResNet34   | [retinaface_r34.pth](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.pth) | [retinaface_r34.onnx](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.onnx) |
| ResNet50   | [retinaface_r50.pth](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r50.pth) | [retinaface_r50.onnx](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r50.onnx) |
|            |

## 📊 Inference

Inference the model using:

```bash
python detect.py --network mobilenetv1 --weights retinaface_mv1.pth
```

<div align="center">
<p>Using MobileNet v2 as a backbone, 632 faces found on large selfi image, see the `assets` folder.</p>
<img src="assets/mv2_large_selfi_632people.jpg">
</div>

## 🧪 Evaluating RetinaFace on WiderFace Dataset

### 1. Get and Install WiderFace Evaluation Tool

1. Clone the WiderFace evaluation repository inside the `retinaface-pytorch` folder:
   ```bash
   git clone https://github.com/yakhyo/widerface_evaluation
   ```
2. Navigate to the `widerface_evaluation` folder and build the required extension:
   ```bash
   cd widerface_evaluation
   python3 setup.py build_ext --inplace
   ```
3. Return to the `retinaface-pytorch` folder after installation is complete:
   ```bash
   cd ..
   ```

### 2. Generate Predictions

Run the following command to evaluate your RetinaFace model with WiderFace, specifying the model architecture (`mobilenetv1` in this example) and the path to the trained weights. Predictions will be stored in `widerface_txt` inside the `widerface_evaluation` folder.

```bash
python evaluate_widerface.py --network mobilenetv1 --weights weights/mobilenetv1.pth
```

### 3. Run the Final Evaluation

After generating predictions, navigate to the widerface_evaluation folder and run the following command to compare predictions with the ground truth annotations:

```bash
cd widerface_evaluation
python evaluation.py -p widerface_txt -g ground_truth
```

> [!NOTE]  
> Ensure `ground_truth` is the path to the WiderFace ground truth directory.

This will begin the evaluation process of your model on the WiderFace dataset.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🔗 References

- https://github.com/biubug6/Pytorch_Retinaface
- https://github.com/yakhyo/faceboxes-pytorch
