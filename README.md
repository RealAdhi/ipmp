# IPMP

Team Details:

Adhithya Sriram: PES1UG24CS393
Ramith M R : PES1UG24CS367
Prathamesh Kumar Singh: PES1UG24CS341
Sushruth Sukesh: PES1UG24CS486

---

## Mini Project 2: Feature Extraction and Classification

Code: `mp2_classification.py` (single file). Input: the 100 images in `processed_dataset/` from Mini Project 1.

### What was done

| Requirement | Implementation |
|---|---|
| SIFT features + SVM | SIFT descriptors, bag-of-visual-words histogram (100 words), SVM |
| Harris corner features + SVM | Harris response: corner stats, 4x4 corner density grid, strength histogram, SVM |
| Image classification with ResNet | ResNet-18 (ImageNet-pretrained), fine-tuned |
| Image classification with MViT | MViTv2-Tiny via `timm` (ImageNet-pretrained), fine-tuned |

The SVM uses scaled features, and C, kernel and gamma are picked by 5-fold cross-validation on the training set. A SIFT+Harris combined SVM is also reported.

### Labels

Our MP1 dataset has only healthy potato leaves (one class), so there is nothing to classify by disease. We classify **leaf shape: round vs elongated**:

1. Each processed image is matched back to its original colour photo in `raw_images/` (matching correlation >= 0.988).
2. The leaf is segmented (green mask) and its spine centre is found (centroid on the main axis).
3. Circle score = 1 - std/mean of the distance from the spine centre to the leaf edge in every direction.
   Rectangle score = leaf area / area of its tightest bounding rectangle. Aspect ratio = long side / short side.
4. Leaves are split at the median roundness into 50 round and 50 elongated.

The labels come from a geometric rule, so this is a shape task, not disease detection.

### How to run

```bash
pip install opencv-python scikit-learn scipy matplotlib pandas pillow torch torchvision timm
python mp2_classification.py
```

Run from the repo root (needs `processed_dataset/` and `raw_images/`). Internet is needed on the first run to download pretrained weights; otherwise it trains from scratch with a warning.

Options:

```bash
python mp2_classification.py --epochs 15      # more training (default 10)
python mp2_classification.py --skip_deep      # only SIFT/Harris + SVM, runs in seconds
```

About 3 minutes on CPU.

### Output

- Console: label statistics, then per model the best SVM params, CV accuracy, test accuracy, classification report and confusion matrix; per-epoch loss and accuracy for ResNet and MViT; and a final summary table.
- `shape_labels.csv`: per image circle %, rectangle %, aspect ratio and class.
- `shape_preview.png`: leaf outline, spine centre and axis drawn on the images, to check the segmentation.

Sample result (80/20 stratified split, 20 test images, seed 42):

| Method | Test acc |
|---|---|
| SIFT + SVM | 0.60 |
| Harris + SVM | 0.60 |
| SIFT + Harris + SVM | 0.65 |
| ResNet-18 | 0.55 |
| MViT | 0.80 |

The test set is tiny, so one image moves accuracy by 5% and results vary slightly between runs and machines. SIFT ignores rotation and scale, so it is weak at global shape.
