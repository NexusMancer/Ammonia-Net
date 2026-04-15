# Ammonia-Net

<p align="center">
  <img src="./Ammonia-Net%20Logo.png" alt="Ammonia-Net Logo" width="500" />
</p>

This project implements a **multi-task learning pipeline for tooth-marked tongue analysis**. The model combines:

- **Semantic segmentation** of tongue / toothmark regions
- **Grading classification** of tooth-mark severity
- **Binary detection** of whether toothmarks are present

The repository also emphasizes a **two-stage training schedule**:

1. **Pretrain stage** (`Pretrain_Epoch`)
2. **Post-train stage** (`PostTrain_Epoch`)

That pretrain -> post-train workflow, together with the segmentation + classification multi-task design, is the main idea of this repository.

Preprint: [Ammonia-Net: A Multi-task Joint Learning Model for Multi-class Segmentation and Classification in Tooth-marked Tongue Diagnosis](https://arxiv.org/abs/2310.03472)

## Highlights

- **Segmentation + classification multi-task learning** in a single model
- **Staged training** with an explicit pretrain phase and a post-train phase
- **Three outputs per image**: segmentation map, 4-class grade, and 2-class detection result
- **VOC-style dataset layout** with split files under `ImageSets/Segmentation`
- **Inference script included** for single-image and batch prediction

## Model Overview

`AmmoniaNet` is defined in `model/architectures/ammonia_net.py`.

The architecture contains three connected parts:

- **Segmentation branch**: a U-Net in `model/segmentation/unet.py`
- **Grading branch**: a ShuffleNetV2 head that takes **RGB image + segmentation output**
- **Detection branch**: a ShuffleNetV2 head that takes the **RGB image only**

In simplified form:

```text
Input image
   |
   +--> UNet ---------------------------> segmentation logits
   |
   +--> RGB + segmentation logits ------> grading head (4 classes)
   |
   +--> RGB ----------------------------> detection head (2 classes)
```

### Output Tasks

- **Segmentation**
  - Default number of classes: `3`
  - Default class names: `_background_`, `Tongue`, `toothmarks`
- **Grading classification**
  - `None`
  - `LightlyTooth`
  - `ModerateTooth`
  - `SevereTooth`
- **Detection**
  - `0`: no toothmark
  - `1`: toothmark present

### Why Multi-task Learning Matters Here

This codebase is built around **segmentation + classification multi-task learning**, not a segmentation-only workflow.

- The **segmentation branch** provides spatial structure
- The **grading branch** predicts severity
- The **detection branch** acts as a coarse presence/absence classifier
- During inference, detection / grading outputs are used to gate the final segmentation result

## Pretrain -> Post-train Workflow

One of the most important ideas in this repository is the staged training schedule configured in `config/config.py`.

### Stage 1: Pretrain

Controlled by:

- `Pretrain_Epoch`
- `Pretrain_batch_size`

This is the **first training stage** of the project workflow.

### Stage 2: Post-train

Controlled by:

- `PostTrain_Epoch`
- `PostTrain_batch_size`

When the training loop enters this stage, the code calls:

```python
model.freeze_segmentation_branch()
```

At that point the segmentation branch is frozen and training continues with the remaining trainable parts.

### Important Note About Terminology

In this repository, **"pretrain" and "post-train" refer to the two internal stages of the training schedule**. They do **not** primarily mean automatic downloading of external pretrained weights.

If you want a single-stage run, set:

```python
PostTrain_Epoch = 0
```

## Repository Structure

```text
Ammonia_Net_refactoring/
├── config/
│   └── config.py
├── dataset/
│   ├── dataset.py
│   └── prepare_dataset.py
├── engine/
│   ├── evaluation.py
│   ├── inference.py
│   ├── inference_utils.py
│   ├── trainer.py
│   └── training_utils.py
├── model/
│   ├── architectures/
│   ├── classifiers/
│   ├── encoders/
│   ├── layers/
│   └── segmentation/
├── script/
│   ├── predict.py
│   └── train.py
├── utils/
│   ├── checkpoint.py
│   ├── losses.py
│   ├── metrics.py
│   ├── preprocessing.py
│   └── wandb_logger.py
├── requirements.txt
└── README.md
```

## Installation

Create an environment first, then install the required packages.

```bash
pip install --upgrade pip
conda create -n AMN python=3.12
```

Install a matching PyTorch build for your machine, then install the rest:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Note:
Please adjust the PyTorch install command based on whether your machine uses CUDA and which CUDA version is available. For the exact version-specific install commands, see the official PyTorch guide:
https://pytorch.org/get-started/previous-versions/

Notes:

- `torch`, `torchvision`, and `cv2` are used by the codebase
- `requirements.txt` mainly lists the project-specific Python dependencies used by this repository

## Dataset Format

The training code expects a **VOC-style dataset layout**.

```text
dataset_root/
├── JPEGImages/
│   ├── None-0001.jpg
│   ├── LightlyTooth-0002.jpg
│   └── ModerateTooth-0003.jpg
├── SegmentationClass/
│   ├── None-0001.png
│   ├── LightlyTooth-0002.png
│   └── ModerateTooth-0003.png
└── ImageSets/
    └── Segmentation/
        ├── train.txt
        ├── val.txt
        ├── test.txt
        └── trainval.txt
```

### Naming Convention

Classification labels are parsed from the **filename prefix before the first `-`**.

Valid prefixes are:

- `None`
- `LightlyTooth`
- `ModerateTooth`
- `SevereTooth`

For example:

- `None-001.jpg` -> class `0`
- `LightlyTooth-002.jpg` -> class `1`
- `ModerateTooth-003.jpg` -> class `2`
- `SevereTooth-004.jpg` -> class `3`

### Split File Generation

If your images and masks already exist under `JPEGImages/` and `SegmentationClass/`, you can generate split files with:

```bash
python -m dataset.prepare_dataset /path/to/dataset_root \
  --trainval-percent 1.0 \
  --train-percent 0.9 \
  --seed 0
```

This script validates masks and writes:

- `train.txt`
- `val.txt`
- `test.txt`
- `trainval.txt`

under `ImageSets/Segmentation/`.

## Configuration

Most training behavior is controlled in `config/config.py`.

Before training, you will usually want to review at least:

- `dataset_root_path`
- `save_dir_template`
- `Pretrain_Epoch`
- `PostTrain_Epoch`
- `Pretrain_batch_size`
- `PostTrain_batch_size`
- `input_shape`
- `loss_weight`
- `loss_grade_weight`
- `wandb_mode`

### Useful Defaults to Know

- The default dataset root can also be set with `AMN_DATASET_ROOT`
- Weights & Biases logging defaults to `offline`
- Checkpoints are saved outside the repo by default through `save_dir_template`

## Training

### Quick Start

Option 1: pass the dataset path directly:

```bash
python script/train.py \
  --dataset-root-path /path/to/dataset_root \
  --fold 0
```

Option 2: use an environment variable:

```bash
export AMN_DATASET_ROOT=/path/to/dataset_root
python script/train.py --fold 0
```

### What Happens During Training

- The script reads `train.txt` and `val.txt`
- It builds `AmmoniaNet`
- It trains with the configured **pretrain stage**
- If `PostTrain_Epoch > 0`, it then switches into the **post-train stage**
- Validation metrics and checkpoints are written every epoch according to the config

### Checkpoints

The training loop writes:

- `best_epoch_weights.pth`
- `last_epoch_weights.pth`
- periodic epoch checkpoints with metrics in the filename

If checkpoint files already exist in the configured save directory, the script resumes from the latest one automatically.

## Inference

Run prediction on a single image:

```bash
python script/predict.py /path/to/image.jpg \
  --model-path /path/to/best_epoch_weights.pth \
  --output ./predictions/result.png \
  --draw-grade-label
```

Run prediction on a directory:

```bash
python script/predict.py /path/to/images \
  --model-path /path/to/best_epoch_weights.pth \
  --output ./predictions \
  --draw-grade-label
```

### Useful Inference Options

- `--cpu`: force CPU inference
- `--count`: print segmentation pixel statistics
- `--mix-type 0`: blend original image and segmentation result
- `--mix-type 1`: save segmentation mask only
- `--mix-type 2`: save foreground only

## Metrics

The project tracks both segmentation and classification performance.

### Segmentation Metrics

- mIoU
- recall
- precision
- pixel accuracy
- F-score

### Classification Metrics

- grade accuracy / precision / recall / F1
- detection accuracy / precision / recall / F1

Validation segmentation metrics are computed through `engine/evaluation.py`, and training statistics are logged through `utils/wandb_logger.py`.

## Practical Notes

- `num_classes` is expected to be `3` by the current `AmmoniaNet` implementation
- images are expected as `.jpg`
- masks are expected as `.png`
- split files are expected under `ImageSets/Segmentation`
- the grading label comes from the filename prefix, not from a separate annotation file

## Preprint

This repository is based on:

**Ammonia-Net: A Multi-task Joint Learning Model for Multi-class Segmentation and Classification in Tooth-marked Tongue Diagnosis**

Preprint: https://arxiv.org/abs/2310.03472

## Acknowledgement

The initial code draft, developed between October 2022 and April 30, 2023, referenced:
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing.git

## Affiliation

- Tsinghua University
- Peking University
- Shanghai Jiao Tong University
- Zhejiang University
- Xi'an Jiaotong University
- Shandong University
- ShanghaiTech University
