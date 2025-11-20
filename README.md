# U-Net Medical Image Segmentation

![Python](https://badgen.net/badge/Python/3.x/blue)
![PyTorch](https://badgen.net/badge/PyTorch/1.x/red)
![License](https://badgen.net/badge/license/MIT/green)
![Status](https://badgen.net/badge/status/stable/green)

[English](README.md) | [中文](README_zh.md)

This repository contains a PyTorch implementation of U-Net for medical image segmentation tasks. The model is designed to accurately segment regions of interest in medical images, which is crucial for diagnosis and treatment planning.

## Table of Contents
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Testing](#testing)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualization](#visualization)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)

## Project Structure

```
.
├── model.py              # U-Net model implementation
├── dataset.py            # Custom dataset class for loading images and masks
├── transforms.py         # Data augmentation and preprocessing transformations
├── train.py              # Training script
├── test.py               # Testing and inference script
├── utils/
│   ├── metrics.py        # Evaluation metrics (e.g., Dice score)
│   └── visualize.py      # Visualization utilities
├── README.md             # Project documentation (English)
├── README_zh.md          # Project documentation (Chinese)
└── train_log.txt         # Training logs
```

## Model Architecture

The implementation follows the classic U-Net architecture with an encoder-decoder structure:

- **Encoder Path**: Series of convolutional blocks with downsampling operations
- **Bottleneck**: Central feature extraction layer
- **Decoder Path**: Upsampling layers with skip connections from the encoder
- **Output Layer**: Final segmentation map

Key components:
- Convolutional blocks with batch normalization and dropout (0.3)
- LeakyReLU activation functions
- Downsampling and upsampling modules
- Skip connections to preserve spatial information

## Dataset

The `UNetDataset` class handles loading of paired images and segmentation masks:

- Images are loaded in RGB format
- Masks are loaded in grayscale format
- Supports custom transformations for data augmentation

## Training

The training pipeline includes:

- Data augmentation with horizontal/vertical flips, rotations, and brightness adjustments
- Binary cross-entropy with logits loss function
- Adam optimizer with learning rate of 1e-4
- Batch size of 4
- Configurable number of epochs (default 20)

To train the model:
```bash
python train.py
```

Model checkpoints are saved in the `model_save/` directory.

## Testing

The testing script performs inference on test data:

- Loads the trained model weights
- Processes images without augmentation
- Generates binary segmentation masks
- Saves predictions as PNG images in the `results/` directory

To run inference:
```bash
python test.py
```

## Evaluation Metrics

We use the Dice coefficient to evaluate segmentation performance:

- Range: 0 (no overlap) to 1 (perfect overlap)
- Robust to class imbalance in medical images
- Computed as: 2 * |X ∩ Y| / (|X| + |Y|)

Implementation can be found in [utils/metrics.py](utils/metrics.py).

## Visualization

The visualization utility creates side-by-side comparisons of:
- Original input images
- Ground truth segmentation masks
- Predicted segmentation masks

This helps qualitatively assess model performance. Implementation is in [utils/visualize.py](utils/visualize.py).

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Albumentations
- OpenCV
- NumPy
- Matplotlib

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset with images and corresponding masks in separate folders
2. Update the data paths in `train.py` and `test.py` as needed
3. Run training: `python train.py`
4. Run inference: `python test.py`
5. Check results in the `results/` directory

## Results

The model achieves competitive performance on medical image segmentation tasks. Sample results can be found in the `results/` directory after running the test script.

## License

This project is released under the MIT License.