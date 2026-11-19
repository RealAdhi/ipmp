# IPMP – Plant Leaf Disease Classification

Image Processing Mini Project: classify plant leaf images as **Healthy** or **Diseased** using classical feature extraction (SIFT + Harris + SVM) and deep learning (ResNet-18, MViT-v2).

## Team

| Name | SRN |
|---|---|
| Sushruth Sukesh | PES1UG24CS486 |
| Prathamesh Kumar Singh | PES1UG24CS341 |
| Adhithya Sriram | PES1UG24CS393 |
| Ramith M R | PES1UG24CS367 |



## Problem

Checking crops for disease by eye is slow and error-prone. Given a photo of a leaf, the goal is to predict whether it is **Healthy** or **Diseased**. This is a binary image classification task, and the project compares features designed by hand with features a network learns.

## Repository layout

```
├── raw_images/              # 100 original leaf images from PlantVillage (256×256 RGB)
├── processed_dataset/       # 1.jpg … 100.jpg after preprocessing
├── dataset_summary.csv      # metadata + label for every processed image
├── process_100.py           # Mini Project 1: dataset preparation pipeline
├── PES1UG24CS486_PES1UG24CS393_PES1UG24CS367_PES1UG24CS341.py
│                            # Mini Project 2: feature extraction + classification
└── IP_report.pdf            # Mini Project 1 report
```

## Part 1: Dataset preparation (`process_100.py`)

Picks 100 images at random from `raw_images/` and runs each one through:

1. **Grayscale conversion:** 3 channels become 1, keeping structure and brightness.
2. **Gaussian blur (5×5):** removes sensor noise and fine grain.
3. **K-means clustering (K=5):** splits pixel brightness into 5 levels (background, healthy tissue, mid-tones, lesions).
4. **Save:** writes the result as `N.jpg` at 80% JPEG quality into `processed_dataset/`.
5. **Label + log:** labels the image `Diseased` if its original filename contains `blight`, `spot`, `rust`, `rot`, `scab` or `diseased`, and `Healthy` otherwise. Writes a row to `dataset_summary.csv`.

CSV columns: `Image Identification/Filename`, `Image Source`, `Original Dimensions (WxH)`, `Image Format`, `Category or Grouping`.

## Part 2: Feature extraction and classification (main script)

### Classical pipeline: SIFT + Harris → SVM

- **SIFT:** finds up to 256 keypoints, each with a 128-number descriptor (4×4 cells × 8 gradient-direction bins). The number of keypoints varies from image to image, so the descriptors are summarised as **mean (128) + standard deviation (128) = 256 features**.
- **Harris corners:** response `R = det(M) − k·trace(M)²` with `k = 0.04`. Pixels above 1% of the maximum count as corners. Gives **4 features**: corner count, corner density, average strength and maximum strength.
- **Feature vector:** 256 + 4 = **260 features per image**.
- **Classifier:** `StandardScaler` → `SVC(kernel="rbf", C=10, gamma="scale")`, with a stratified 80/20 train/test split.

### Deep learning

- **ResNet-18:** a lightweight CNN baseline. The last `fc` layer is replaced to output the project's classes, and ImageNet weights can be loaded with `--pretrained`.
- **MViT-v2-S:** a multiscale vision transformer, used for comparison with attention-based models. Note that torchvision's MViT-v2-S is a *video* model, and its pretrained weights come from Kinetics-400. Each image is treated as a single frame.
- **Training:** images resized to 224×224, random horizontal flip, ImageNet normalisation, AdamW optimiser and cross-entropy loss.

Each model prints its test accuracy and a per-class precision/recall/F1 report.

## Setup

```bash
pip install numpy pandas opencv-python scikit-learn torch torchvision pillow
```

## Usage

```bash
# 1. Build the processed dataset and CSV
python process_100.py

# 2. Run all classifiers
python PES1UG24CS486_PES1UG24CS393_PES1UG24CS367_PES1UG24CS341.py --model all --pretrained

# Or run one model
python PES1UG24CS486_PES1UG24CS393_PES1UG24CS367_PES1UG24CS341.py --model svm
python PES1UG24CS486_PES1UG24CS393_PES1UG24CS367_PES1UG24CS341.py --model resnet --pretrained
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset-dir` | `processed_dataset` | Folder with the processed images |
| `--labels` | `dataset_summary.csv` | Label CSV |
| `--model` | `all` | `svm`, `resnet`, `mvit` or `all` |
| `--epochs` | `5` | Training epochs (deep models) |
| `--batch-size` | `8` | Batch size (deep models) |
| `--learning-rate` | `1e-4` | AdamW learning rate |
| `--pretrained` | off | Start from pretrained weights |
| `--seed` | `42` | Random seed for the split |

## Known limitations

- **The current dataset has only one class.** All 100 raw images are PlantVillage *healthy* leaves (`RS_HL` in the filename), so every row in `dataset_summary.csv` is `Healthy`. The classification script stops with `"The labels contain only one class"` until diseased images are added. Labelling by source folder (e.g. `___Early_blight`) would be more reliable than matching keywords in filenames.
- **Grayscale + K-means removes colour**, and colour is a strong sign of disease. Feeding the classifiers RGB images is likely to improve accuracy.
- **Small dataset:** with 100 images the test split is 20 images, so each mistake costs 5% accuracy. Use `--pretrained` for the deep models.
- `process_100.py` samples without a fixed seed, so each run picks a different 100 images.

## Results

| Method | Test accuracy |
|---|---|
| SIFT + Harris + SVM | 60% |
| ResNet-18 | 90% |

ResNet-18 beats the classical pipeline by 30 percentage points. Features learned by a CNN capture leaf texture and lesion patterns much better than handcrafted SIFT and Harris descriptors.

## Dataset source

[PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset), an open dataset of plant leaf images.
