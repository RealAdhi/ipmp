"""
Mini Project 2: plant-image feature extraction and classification.

The script implements:
1. SIFT feature extraction
2. Harris corner feature extraction
3. SVM classification using the extracted features
4. ResNet and MViT image classification

Example:
    python PES1UG24CS486_PES1UG24CS393_PES1UG24CS367_PES1UG24CS341.py \
        --dataset-dir processed_dataset --labels dataset_summary.csv --model all

Install dependencies:
    pip install numpy pandas opencv-python scikit-learn torch torchvision
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_COLUMN = "Image Identification/Filename"
LABEL_COLUMN = "Category or Grouping"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_labels(labels_path: Path) -> Dict[str, str]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels CSV was not found: {labels_path}")

    labels = pd.read_csv(labels_path)
    missing = {IMAGE_COLUMN, LABEL_COLUMN} - set(labels.columns)
    if missing:
        raise ValueError(
            f"{labels_path} is missing required column(s): {', '.join(sorted(missing))}"
        )

    result = {}
    for _, row in labels.iterrows():
        filename = str(row[IMAGE_COLUMN]).strip()
        label = str(row[LABEL_COLUMN]).strip()
        if filename and label and label.lower() != "nan":
            result[filename] = label
    if not result:
        raise ValueError(f"No usable image labels were found in {labels_path}")
    return result


def find_labeled_images(dataset_dir: Path, labels: Dict[str, str]) -> Tuple[List[Path], List[str]]:
    image_paths = []
    image_labels = []
    for path in sorted(dataset_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            label = labels.get(path.name)
            if label is not None:
                image_paths.append(path)
                image_labels.append(label)

    if len(image_paths) < 4:
        raise ValueError(
            f"Only {len(image_paths)} labeled image(s) were found in {dataset_dir}; at least 4 are required."
        )
    if len(set(image_labels)) < 2:
        raise ValueError(
            "The labels contain only one class. Correct the CSV labels before running "
            "SVM, ResNet, or MViT classification."
        )
    return image_paths, image_labels


def extract_sift_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=256)
    _, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(256, dtype=np.float32)
    return np.concatenate(
        [descriptors.mean(axis=0), descriptors.std(axis=0)]
    ).astype(np.float32)


def extract_harris_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)
    response = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    threshold = 0.01 * response.max() if response.size else 0.0
    corners = response > threshold
    height, width = gray.shape
    count = float(corners.sum())
    density = count / float(max(height * width, 1))
    strength = float(response[corners].mean()) if count else 0.0
    max_strength = float(response.max()) if response.size else 0.0
    return np.array([count, density, strength, max_strength], dtype=np.float32)


def extract_classical_features(image_paths: Sequence[Path]) -> np.ndarray:
    features = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"OpenCV could not read image: {path}")
        features.append(
            np.concatenate(
                [extract_sift_features(image), extract_harris_features(image)]
            )
        )
    return np.asarray(features, dtype=np.float32)


def split_indices(labels: Sequence[str], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    try:
        return train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=labels
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to make a stratified train/test split. Each class needs enough "
            "labeled images for both splits."
        ) from exc


def run_svm(
    image_paths: Sequence[Path], labels: Sequence[str], seed: int
) -> None:
    features = extract_classical_features(image_paths)
    train_idx, test_idx = split_indices(labels, seed)
    classifier = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, gamma="scale"),
    )
    classifier.fit(features[train_idx], np.asarray(labels)[train_idx])
    predictions = classifier.predict(features[test_idx])
    print("\nSVM using SIFT + Harris features")
    print(f"Accuracy: {accuracy_score(np.asarray(labels)[test_idx], predictions):.4f}")
    print(
        classification_report(
            np.asarray(labels)[test_idx], predictions, zero_division=0
        )
    )


def build_torch_dataset(
    image_paths: Sequence[Path],
    labels: Sequence[str],
    transform,
):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset

    class CsvImageDataset(Dataset):
        def __init__(self):
            self.paths = list(image_paths)
            self.targets = [class_to_index[label] for label in labels]

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with Image.open(self.paths[index]) as image:
                image = image.convert("RGB")
            return transform(image), self.targets[index]

    class_to_index = {label: index for index, label in enumerate(sorted(set(labels)))}
    return CsvImageDataset(), class_to_index


def create_model(model_name: str, class_count: int, pretrained: bool):
    import torch.nn as nn
    from torchvision.models import (
        MViT_V2_S_Weights,
        ResNet18_Weights,
        mvit_v2_s,
        resnet18,
    )

    if model_name == "resnet":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, class_count)
    elif model_name == "mvit":
        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        model = mvit_v2_s(weights=weights)
        model.head[1] = nn.Linear(model.head[1].in_features, class_count)
    else:
        raise ValueError(f"Unsupported deep-learning model: {model_name}")
    return model


def run_deep_classifier(
    model_name: str,
    image_paths: Sequence[Path],
    labels: Sequence[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pretrained: bool,
    seed: int,
) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    train_idx, test_idx = split_indices(labels, seed)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    dataset, class_to_index = build_torch_dataset(image_paths, labels, transform)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, len(class_to_index), pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        mean_loss = running_loss / max(len(train_loader.dataset), 1)
        print(f"{model_name.upper()} epoch {epoch}/{epochs} - loss: {mean_loss:.4f}")

    model.eval()
    predictions, expected = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            output = model(images.to(device))
            predictions.extend(output.argmax(dim=1).cpu().tolist())
            expected.extend(targets.tolist())
    print(f"\n{model_name.upper()} test accuracy: {accuracy_score(expected, predictions):.4f}")
    print(
        classification_report(
            expected,
            predictions,
            labels=list(range(len(class_to_index))),
            target_names=[
                label for label, _ in sorted(class_to_index.items(), key=lambda item: item[1])
            ],
            zero_division=0,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("processed_dataset"))
    parser.add_argument("--labels", type=Path, default=Path("dataset_summary.csv"))
    parser.add_argument(
        "--model", choices=["svm", "resnet", "mvit", "all"], default="all"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs and batch-size must be positive; learning-rate must be greater than zero.")

    set_seed(args.seed)
    labels = load_labels(args.labels)
    image_paths, image_labels = find_labeled_images(args.dataset_dir, labels)
    print(f"Loaded {len(image_paths)} labeled images from {args.dataset_dir}")
    print(f"Classes: {', '.join(sorted(set(image_labels)))}")

    if args.model in {"svm", "all"}:
        run_svm(image_paths, image_labels, args.seed)
    if args.model in {"resnet", "mvit", "all"}:
        models = ["resnet", "mvit"] if args.model == "all" else [args.model]
        for model_name in models:
            run_deep_classifier(
                model_name,
                image_paths,
                image_labels,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.pretrained,
                args.seed,
            )


if __name__ == "__main__":
    main()
