"""
Mini Project 2 - Feature Extraction + Deep Learning Classification
===================================================================
Input : processed_dataset/  (100 leaf images produced in Mini Project 1)

Our dataset has only healthy potato leaves, so there is no disease label.
We therefore classify LEAF SHAPE: "round" vs "elongated/rectangular".

Pipeline
  Step 0  Label generation : segment the leaf (green mask on the original
                             colour photo, matched to each processed image), find the centre of its spine
                             (midrib), measure how circle-like / rectangle-like
                             it is, and split at the median into two classes.
  Step 1  SIFT  + SVM      : SIFT descriptors -> bag-of-visual-words -> SVM
  Step 2  Harris + SVM     : Harris corner responses -> histogram -> SVM
          (also SIFT+Harris combined)
  Step 3  ResNet-18        : ImageNet-pretrained, fine-tuned
  Step 4  MViT (MViTv2-T)  : ImageNet-pretrained multiscale vision transformer

Usage
  python mp2_classification.py                    # full run
  python mp2_classification.py --epochs 15        # more training
  python mp2_classification.py --skip_deep        # only SIFT/Harris + SVM
"""

import argparse
import os
import random
import warnings

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

warnings.filterwarnings("ignore")

CLASS_NAMES = ["round", "elongated"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# STEP 0 : shape-based label generation
# --------------------------------------------------------------------------
def segment_leaf(gray):
    """Binary leaf mask. Otsu threshold; the side touching the image border is
    background. Keep the largest blob and fill holes."""
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    if border.mean() > 127:          # border is white -> background is white
        m = 255 - m
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(m)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask, [big], -1, 255, -1)   # filled -> holes closed
    return mask


def segment_leaf_color(bgr):
    """Segment the leaf in the ORIGINAL colour image: leaf = green, background =
    pinkish grey. Otsu on the excess-green channel (2G - R - B)."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(cv2.GaussianBlur(bgr, (5, 5), 0))]
    exg = cv2.normalize(2 * g - r - b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(m)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return mask


def _signature(gray):
    g = cv2.GaussianBlur(gray, (0, 0), 3)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


def match_raw_images(files, data_dir, raw_dir):
    """MP1 renamed the images to 1.jpg..100.jpg, so the link to the original
    colour photo is lost. Recover it: the processed image is a blurred, quantised
    version of the raw one, so their low-res signatures correlate ~0.99. A
    one-to-one (Hungarian) assignment gives the pairing."""
    from scipy.optimize import linear_sum_assignment
    raw_files = sorted(f for f in os.listdir(raw_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    R = np.array([_signature(cv2.imread(os.path.join(raw_dir, f), 0)) for f in raw_files])
    P = np.array([_signature(cv2.imread(os.path.join(data_dir, f), 0)) for f in files])
    corr = P @ R.T / R.shape[1]
    rows, cols = linear_sum_assignment(-corr)
    print(f"[Step 0] matched processed->raw images: min corr {corr[rows, cols].min():.3f}, "
          f"mean {corr[rows, cols].mean():.3f}")
    return [os.path.join(raw_dir, raw_files[c]) for c in cols]


def shape_scores(mask, n_rays=180):
    """Measure how much a leaf looks like a circle / square / rectangle.

    Spine centre: the midrib runs along the leaf's main axis (PCA), so the
    leaf centroid on that axis is our centre of spine.
    circle_score : 1 - std(r)/mean(r), where r(theta) is the distance from the
                   centre to the edge in every direction. Circle -> 1.
    rect_score   : leaf area / area of tightest rotated bounding rectangle.
                   Rectangle or square -> 1.
    aspect       : long side / short side of that rectangle. Square -> 1.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    centre = pts.mean(0)
    cov = np.cov((pts - centre).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]                 # spine direction

    # radial profile from the spine centre to the leaf edge
    contour = cnt[:, 0, :].astype(np.float64)
    d = contour - centre
    ang = np.arctan2(d[:, 1], d[:, 0])
    dist = np.hypot(d[:, 0], d[:, 1])
    bins = np.linspace(-np.pi, np.pi, n_rays + 1)
    idx = np.clip(np.digitize(ang, bins) - 1, 0, n_rays - 1)
    r = np.array([dist[idx == k].max() if np.any(idx == k) else np.nan
                  for k in range(n_rays)])
    r = np.where(np.isnan(r), np.nanmean(r), r)
    circle_score = float(np.clip(1.0 - r.std() / r.mean(), 0, 1))

    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
    rect_area = max(w * h, 1e-6)
    rect_score = float(np.clip(area / rect_area, 0, 1))
    aspect = float(max(w, h) / max(min(w, h), 1e-6))
    return dict(circle_score=circle_score, rect_score=rect_score, aspect=aspect,
                area=float(area), cx=float(centre[0]), cy=float(centre[1]),
                axis_x=float(axis[0]), axis_y=float(axis[1]))


def generate_labels(files, data_dir, raw_dir, out_csv, preview_png):
    raw_paths = None
    if raw_dir and os.path.isdir(raw_dir):
        raw_paths = match_raw_images(files, data_dir, raw_dir)
    else:
        print("[warn] raw_images/ not found -> segmenting the grey K-Means images "
              "(less reliable, lit half of a leaf can merge with the background)")
    rows, masks, imgs = [], [], []
    for i, f in enumerate(files):
        g = cv2.imread(os.path.join(data_dir, f), cv2.IMREAD_GRAYSCALE)
        if raw_paths:
            raw = cv2.imread(raw_paths[i])
            m = segment_leaf_color(raw)
            if raw.shape[:2] != g.shape:
                m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            m = segment_leaf(g)
        s = shape_scores(m)
        s["file"] = f
        rows.append(s)
        masks.append(m)
        imgs.append(g)
    df = pd.DataFrame(rows)

    # Combined roundness: high circle score and low aspect ratio => round.
    # (Leaves are ovals, so "rectangle/square" in practice = long and narrow.)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # noqa: E731
    df["roundness"] = z(df["circle_score"]) - z(df["aspect"])
    df["pct_circle"] = (100 * df["circle_score"]).round(1)
    df["pct_rectangle"] = (100 * df["rect_score"]).round(1)
    df["label"] = (df["roundness"] < df["roundness"].median()).astype(int)
    df["class"] = df["label"].map(dict(enumerate(CLASS_NAMES)))
    df.to_csv(out_csv, index=False)

    # preview grid so the segmentation / centre / axis can be checked by eye
    n = min(24, len(df))
    cols = 6
    fig, ax = plt.subplots((n + cols - 1) // cols, cols, figsize=(2.2 * cols, 2.4 * ((n + cols - 1) // cols)))
    for k, a in enumerate(np.ravel(ax)):
        a.axis("off")
        if k >= n:
            continue
        vis = cv2.cvtColor(imgs[k], cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(masks[k], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        r = df.iloc[k]
        c = (int(r.cx), int(r.cy))
        p1 = (int(r.cx - 90 * r.axis_x), int(r.cy - 90 * r.axis_y))
        p2 = (int(r.cx + 90 * r.axis_x), int(r.cy + 90 * r.axis_y))
        cv2.line(vis, p1, p2, (255, 0, 0), 2)
        cv2.circle(vis, c, 5, (0, 0, 255), -1)
        a.imshow(vis[:, :, ::-1])
        a.set_title(f"{r['class']}\ncirc {r.pct_circle:.0f}% rect {r.pct_rectangle:.0f}%", fontsize=7)
    plt.tight_layout()
    plt.savefig(preview_png, dpi=110)
    plt.close()
    return df


# --------------------------------------------------------------------------
# STEP 1 / 2 : handcrafted features (SIFT, Harris)
# --------------------------------------------------------------------------
def sift_descriptors(gray, sift):
    _, des = sift.detectAndCompute(gray, None)
    return des if des is not None else np.zeros((0, 128), np.float32)


def build_sift_bovw(train_imgs, test_imgs, k=100, seed=42):
    """Bag of visual words: cluster all train descriptors into k 'words', then
    describe each image by a histogram of its words (fixed length = k)."""
    sift = cv2.SIFT_create()
    tr_des = [sift_descriptors(g, sift) for g in train_imgs]
    te_des = [sift_descriptors(g, sift) for g in test_imgs]
    stack = np.vstack([d for d in tr_des if len(d)])
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048).fit(stack)

    def hist(des):
        h = np.zeros(k, np.float32)
        if len(des):
            h += np.bincount(km.predict(des), minlength=k)
            h /= h.sum()
        return h

    return (np.array([hist(d) for d in tr_des]), np.array([hist(d) for d in te_des]),
            float(np.mean([len(d) for d in tr_des])))


def harris_feature(gray, grid=4, n_bins=8):
    """Harris corner response -> (a) corner count and strength statistics,
    (b) corner density on a grid x grid layout, (c) histogram of log-response."""
    g = np.float32(gray)
    resp = cv2.cornerHarris(g, blockSize=2, ksize=3, k=0.04)
    resp = cv2.dilate(resp, None)
    thr = 0.01 * resp.max()
    corners = resp > thr
    h, w = gray.shape
    dens = np.zeros((grid, grid), np.float32)
    ys, xs = np.nonzero(corners)
    for y, x in zip(ys, xs):
        dens[min(y * grid // h, grid - 1), min(x * grid // w, grid - 1)] += 1
    dens = dens.ravel() / max(corners.sum(), 1)
    vals = resp[corners]
    if len(vals):
        hist, _ = np.histogram(np.log10(vals / thr), bins=n_bins, range=(0, 4))
        hist = hist / hist.sum()
        stats = [corners.sum() / (h * w) * 1e3, np.log10(vals.mean() + 1e-12), np.log10(vals.max() + 1e-12)]
    else:
        hist, stats = np.zeros(n_bins), [0, 0, 0]
    return np.concatenate([stats, dens, hist]).astype(np.float32)


def run_svm(name, Xtr, ytr, Xte, yte, seed, results):
    """SVM with scaled features; C / kernel / gamma chosen by 5-fold CV on train."""
    pipe = make_pipeline(StandardScaler(), SVC(random_state=seed))
    grid = {"svc__kernel": ["linear", "rbf"], "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.01, 0.001]}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, grid, cv=cv, n_jobs=-1).fit(Xtr, ytr)
    pred = gs.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"\n=== {name} ===  features: {Xtr.shape[1]}-d")
    print(f"best params: {gs.best_params_}   CV acc: {gs.best_score_:.3f}   TEST acc: {acc:.3f}")
    print(classification_report(yte, pred, target_names=CLASS_NAMES, zero_division=0))
    print("confusion matrix [rows=true, cols=pred]:\n", confusion_matrix(yte, pred))
    results.append((name, gs.best_score_, acc))


# --------------------------------------------------------------------------
# STEP 3 / 4 : deep learning (ResNet-18, MViTv2)
# --------------------------------------------------------------------------
def run_deep(model_name, files, labels, tr_idx, te_idx, data_dir, args, results):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = 224
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf_train = transforms.Compose([
        transforms.Resize((size, size)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf_test = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean, std)])

    from PIL import Image

    class DS(Dataset):
        def __init__(self, idx, tf):
            self.idx, self.tf = idx, tf

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            img = Image.open(os.path.join(data_dir, files[j])).convert("RGB")  # gray -> 3 channels
            return self.tf(img), int(labels[j])

    tr_dl = DataLoader(DS(tr_idx, tf_train), batch_size=args.batch, shuffle=True)
    te_dl = DataLoader(DS(te_idx, tf_test), batch_size=args.batch)

    pretrained = not args.no_pretrained
    try:
        if model_name == "ResNet-18":
            net = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            net.fc = nn.Linear(net.fc.in_features, 2)
            lr = 1e-3
        else:
            import timm
            net = timm.create_model("mvitv2_tiny", pretrained=pretrained, num_classes=2)
            lr = 1e-4
    except Exception as e:  # e.g. no internet for pretrained weights
        print(f"[warn] pretrained weights unavailable ({e}); training from scratch")
        if model_name == "ResNet-18":
            net = models.resnet18(weights=None)
            net.fc = nn.Linear(net.fc.in_features, 2)
            lr = 1e-3
        else:
            import timm
            net = timm.create_model("mvitv2_tiny", pretrained=False, num_classes=2)
            lr = 1e-4
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    def evaluate():
        net.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in te_dl:
                ps += net(x.to(device)).argmax(1).cpu().tolist()
                ys += y.tolist()
        return np.array(ys), np.array(ps)

    print(f"\n=== {model_name} ===  device={device}  epochs={args.epochs}  pretrained={pretrained}")
    for ep in range(1, args.epochs + 1):
        net.train()
        tot, correct, n = 0.0, 0, 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = net(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()
        yt, yp = evaluate()
        print(f"epoch {ep:2d}/{args.epochs}  loss {tot / n:.4f}  train acc {correct / n:.3f}  "
              f"test acc {accuracy_score(yt, yp):.3f}")
    yt, yp = evaluate()
    acc = accuracy_score(yt, yp)
    print(classification_report(yt, yp, target_names=CLASS_NAMES, zero_division=0))
    print("confusion matrix [rows=true, cols=pred]:\n", confusion_matrix(yt, yp))
    results.append((model_name, float("nan"), acc))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="processed_dataset")
    ap.add_argument("--raw_dir", default="raw_images", help="original colour images (used only to make labels)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k_words", type=int, default=100, help="SIFT visual vocabulary size")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_deep", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)

    files = sorted([f for f in os.listdir(args.data_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))],
                   key=lambda s: int(os.path.splitext(s)[0]) if os.path.splitext(s)[0].isdigit() else s)
    print(f"Found {len(files)} images in '{args.data_dir}'")

    # ---- Step 0
    df = generate_labels(files, args.data_dir, args.raw_dir, "shape_labels.csv", "shape_preview.png")
    labels = df["label"].values
    print("\n[Step 0] shape labels written to shape_labels.csv, preview in shape_preview.png")
    print(df.groupby("class")[["pct_circle", "pct_rectangle", "aspect"]].mean().round(2))
    print("class counts:", dict(df["class"].value_counts()))

    # ---- split (shared by every model so results are comparable)
    idx = np.arange(len(files))
    tr_idx, te_idx = train_test_split(idx, test_size=args.test_size, stratify=labels, random_state=args.seed)
    ytr, yte = labels[tr_idx], labels[te_idx]
    gray = [cv2.imread(os.path.join(args.data_dir, f), cv2.IMREAD_GRAYSCALE) for f in files]
    gtr, gte = [gray[i] for i in tr_idx], [gray[i] for i in te_idx]
    print(f"train={len(tr_idx)}  test={len(te_idx)}")

    results = []
    # ---- Step 1: SIFT + SVM
    sift_tr, sift_te, avg_kp = build_sift_bovw(gtr, gte, k=args.k_words, seed=args.seed)
    print(f"\n[Step 1] SIFT: avg keypoints/image = {avg_kp:.1f}, visual words = {args.k_words}")
    run_svm("SIFT (BoVW) + SVM", sift_tr, ytr, sift_te, yte, args.seed, results)

    # ---- Step 2: Harris + SVM
    har_tr = np.array([harris_feature(g) for g in gtr])
    har_te = np.array([harris_feature(g) for g in gte])
    run_svm("Harris + SVM", har_tr, ytr, har_te, yte, args.seed, results)
    run_svm("SIFT + Harris + SVM", np.hstack([sift_tr, har_tr]), ytr, np.hstack([sift_te, har_te]),
            yte, args.seed, results)

    # ---- Step 3 / 4: deep learning
    if not args.skip_deep:
        run_deep("ResNet-18", files, labels, tr_idx, te_idx, args.data_dir, args, results)
        run_deep("MViT (MViTv2-T)", files, labels, tr_idx, te_idx, args.data_dir, args, results)

    print("\n================ SUMMARY (test set) ================")
    print(pd.DataFrame(results, columns=["Method", "CV acc (train)", "Test acc"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
