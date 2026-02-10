#!/usr/bin/env python
# coding: utf-8
"""
Galaxy morphology classification setup: data download, preparation, classical ML and deep learning training.
All execution is driven from main(); no loose executable code at module level.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import moments_hu
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------

ZIP_NAME = "galaxy-zoo-the-galaxy-challenge.zip"
INNER_ZIPS = ["images_training_rev1.zip", "training_solutions_rev1.zip"]
LABELS_CSV = "training_solutions_rev1.csv"
IMAGES_SUBDIR = "images_training_rev1"
N_PER_CLASS = 2000
BATCH_SIZE = 32
LABEL_TO_IDX = {"elliptical": 0, "spiral": 1}
IDX_TO_LABEL = {0: "Elliptical", 1: "Spiral"}


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Kaggle setup and download
# ---------------------------------------------------------------------------


def setup_kaggle_credentials(
    username: str,
    key: str,
) -> None:
    """Write Kaggle API credentials to ~/.kaggle/kaggle.json and set permissions (Unix only)."""
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_path, "w") as f:
        json.dump({"username": username, "key": key}, f)
    if sys.platform != "win32":
        os.chmod(kaggle_path, 0o600)


def install_kaggle() -> None:
    """Install kaggle CLI via pip."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "kaggle"],
        check=True,
    )


def download_competition_data(competition: str = "galaxy-zoo-the-galaxy-challenge") -> None:
    """Download competition data using Kaggle CLI."""
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition],
        check=True,
    )


# ---------------------------------------------------------------------------
# Data extraction and paths
# ---------------------------------------------------------------------------


def find_zip_path(zip_name: str = ZIP_NAME) -> tuple[str, str, str]:
    """
    Locate the competition zip in cwd or /content (Colab). Return (ZIP_PATH, RAW_DIR, DATA_DIR).
    Raises FileNotFoundError if zip not found.
    """
    candidates = [os.getcwd()]
    if os.path.isdir("/content"):
        candidates.append("/content")
    zip_path = None
    base_dir = None
    for d in candidates:
        p = os.path.join(d, zip_name)
        if os.path.isfile(p):
            zip_path = p
            base_dir = d
            break
    if zip_path is None:
        raise FileNotFoundError(
            f"{zip_name} not found in any of: {candidates}. "
            "Run download_competition_data() first."
        )
    raw_dir = os.path.join(base_dir, "galaxy_zoo_raw")
    data_dir = os.path.join(base_dir, "galaxy_zoo")
    return zip_path, raw_dir, data_dir


def extract_main_zip(zip_path: str, raw_dir: str, data_dir: str) -> list[str]:
    """Extract main competition zip into raw_dir; create data_dir. Return list of top-level names."""
    import zipfile
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)
    return os.listdir(raw_dir)


def extract_inner_archives(raw_dir: str, data_dir: str, names: list[str] | None = None) -> list[str]:
    """Extract inner zip files (e.g. images_training_rev1.zip) from raw_dir into data_dir."""
    import zipfile
    to_extract = names or INNER_ZIPS
    for name in to_extract:
        path = os.path.join(raw_dir, name)
        if os.path.isfile(path):
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(data_dir)
    return os.listdir(data_dir)


# ---------------------------------------------------------------------------
# Labels and dataframe preparation
# ---------------------------------------------------------------------------


def assign_morphology_label(row: pd.Series) -> Optional[str]:
    """Map Galaxy Zoo confidence columns to 'elliptical' or 'spiral' (or None)."""
    if row["Class1.1"] > 0.7:
        return "elliptical"
    if row["Class1.2"] > 0.7:
        return "spiral"
    return None


def load_and_prepare_labels(data_dir: str) -> pd.DataFrame:
    """Load training solutions CSV and add morphology column (confidence > 0.7)."""
    path = os.path.join(data_dir, LABELS_CSV)
    labels = pd.read_csv(path)
    labels["morphology"] = labels.apply(assign_morphology_label, axis=1)
    labels = labels.dropna(subset=["morphology"])
    return labels


def sample_balanced_dataset(
    labels: pd.DataFrame,
    n_per_class: int = N_PER_CLASS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample n_per_class for each morphology and concatenate."""
    elliptical = labels[labels["morphology"] == "elliptical"].sample(
        n=n_per_class, random_state=random_state
    )
    spiral = labels[labels["morphology"] == "spiral"].sample(
        n=n_per_class, random_state=random_state
    )
    return pd.concat([elliptical, spiral]).reset_index(drop=True)


def add_image_paths_and_filter(
    df: pd.DataFrame,
    data_dir: str,
    images_subdir: str = IMAGES_SUBDIR,
) -> pd.DataFrame:
    """Add image_path column and drop rows whose image file does not exist."""
    img_dir = os.path.join(data_dir, images_subdir)
    df = df.copy()
    df["image_path"] = df["GalaxyID"].apply(
        lambda x: os.path.join(img_dir, f"{x}.jpg")
    )
    return df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)


def train_val_test_split_df(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train, validation, and test."""
    trainval, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["morphology"],
        random_state=random_state,
    )
    train, val = train_test_split(
        trainval,
        test_size=val_ratio,
        stratify=trainval["morphology"],
        random_state=random_state,
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Classical ML: feature extraction and SVM
# ---------------------------------------------------------------------------


def extract_features(img_path: str, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Extract intensity stats, Hu moments, and GLCM texture from a single image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    features = [img.mean(), img.std()]
    hu = moments_hu(img)
    features += hu.tolist()
    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    features += [
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
    ]
    return np.array(features, dtype=np.float64)


def build_classical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features and labels for train/val/test. Returns X_train, X_val, X_test, y_train, y_val, y_test."""
    X_train = np.vstack(train_df["image_path"].apply(extract_features))
    X_val = np.vstack(val_df["image_path"].apply(extract_features))
    X_test = np.vstack(test_df["image_path"].apply(extract_features))
    y_train = train_df["morphology"].map(LABEL_TO_IDX).values
    y_val = val_df["morphology"].map(LABEL_TO_IDX).values
    y_test = test_df["morphology"].map(LABEL_TO_IDX).values
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_classical_ml_pipeline(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    param_grid: dict | None = None,
    cv: int = 5,
    save_path: str = "svm_model.joblib",
) -> tuple[SVC, np.ndarray, np.ndarray]:
    """Scale data, run GridSearchCV for SVM, evaluate on val and test; optionally save model. Returns (best_svm, y_val_pred, y_test_pred)."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    param_grid = param_grid or {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.001]}
    grid = GridSearchCV(
        SVC(kernel="rbf"),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train_s, y_train)
    best_svm = grid.best_estimator_
    y_val_pred = best_svm.predict(X_val_s)
    y_test_pred = best_svm.predict(X_test_s)
    if save_path:
        joblib.dump(best_svm, save_path)
    return best_svm, y_val_pred, y_test_pred


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def show_examples(
    df: pd.DataFrame,
    label: str,
    n: int = 5,
    random_state: int = 42,
) -> None:
    """Plot n example images for a given morphology label."""
    examples = df[df["morphology"] == label].sample(n=n, random_state=random_state)
    plt.figure(figsize=(15, 3))
    for i, (_, row) in enumerate(examples.iterrows()):
        img = Image.open(row["image_path"])
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    plt.show()


def plot_svm_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot confusion matrix for classical SVM."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=IDX_TO_LABEL.values(),
        yticklabels=IDX_TO_LABEL.values(),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Classical ML (SVM) Confusion Matrix")
    plt.show()


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Train vs Validation Loss",
) -> None:
    """Plot train and validation loss curves."""
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# PyTorch dataset and transforms
# ---------------------------------------------------------------------------


class GalaxyDataset(Dataset):
    """Dataset of galaxy images with morphology labels."""

    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.loc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = LABEL_TO_IDX[row["morphology"]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_baseline_transform() -> transforms.Compose:
    """Resize and ToTensor only."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def get_robust_transform() -> transforms.Compose:
    """Resize, augmentation (rotation, jitter, blur), then ToTensor."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Deep learning model and training
# ---------------------------------------------------------------------------


def get_model(
    num_classes: int = 2,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """ResNet-18 with last layer replaced; layer4 unfrozen."""
    dev = device or _get_device()
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(dev)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
) -> tuple[list[float], list[float]]:
    """Train model and return (train_losses, val_losses)."""
    dev = device or _get_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        avg_train = running_train_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        running_val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(dev), labels.to(dev)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val = running_val_loss / len(val_loader)
        val_losses.append(avg_val)
        acc = correct / total
        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {acc:.3f}"
        )
    return train_losses, val_losses


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) arrays."""
    dev = device or _get_device()
    model.eval()
    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(dev)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true_list.extend(labels.numpy().tolist())
            y_pred_list.extend(preds.cpu().numpy().tolist())
    return np.array(y_true_list), np.array(y_pred_list)


def show_predictions(
    model: nn.Module,
    dataset: GalaxyDataset,
    n: int = 6,
    device: Optional[torch.device] = None,
) -> None:
    """Plot n samples with ground truth and prediction."""
    dev = device or _get_device()
    model.eval()
    plt.figure(figsize=(12, 6))
    for i in range(n):
        img, label = dataset[i]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(dev)).argmax(dim=1).item()
        plt.subplot(2, n // 2, i + 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"GT: {IDX_TO_LABEL[label]}\nPred: {IDX_TO_LABEL[pred]}")
        plt.axis("off")
    plt.show()


def plot_dl_confusion_matrices(
    y_true_base: np.ndarray,
    y_pred_base: np.ndarray,
    y_true_robust: np.ndarray,
    y_pred_robust: np.ndarray,
) -> None:
    """Plot two confusion matrices side by side (baseline vs robust)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels_list = list(IDX_TO_LABEL.values())
    ConfusionMatrixDisplay(
        confusion_matrix(y_true_base, y_pred_base),
        display_labels=labels_list,
    ).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Baseline Model")
    ConfusionMatrixDisplay(
        confusion_matrix(y_true_robust, y_pred_robust),
        display_labels=labels_list,
    ).plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Robust Model")
    plt.show()


def save_models(
    baseline_model: nn.Module,
    robust_model: nn.Module,
    baseline_path: str = "baseline_galaxy_model.pth",
    robust_path: str = "robust_galaxy_model.pth",
) -> None:
    """Save model state dicts to disk."""
    torch.save(baseline_model.state_dict(), baseline_path)
    torch.save(robust_model.state_dict(), robust_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(
    kaggle_username: str = "tiffanydegbotse",
    kaggle_key: str = "f867da44f8b08718e4c74e2eb26c56c9",
    skip_kaggle_setup: bool = False,
    skip_download: bool = False,
    n_per_class: int = N_PER_CLASS,
    epochs_baseline: int = 10,
    epochs_robust: int = 10,
    run_classical: bool = True,
    save_models_at_end: bool = True,
    show_plots: bool = True,
) -> None:
    """
    Full pipeline: Kaggle setup, download, extract, prepare labels, optional classical ML,
    then deep learning (baseline + robust) training and evaluation.
    """
    device = _get_device()
    print(f"Using device: {device}")

    if not skip_kaggle_setup:
        setup_kaggle_credentials(kaggle_username, kaggle_key)
        install_kaggle()

    if not skip_download:
        download_competition_data()

    zip_path, raw_dir, data_dir = find_zip_path()
    extract_main_zip(zip_path, raw_dir, data_dir)
    extract_inner_archives(raw_dir, data_dir)

    labels = load_and_prepare_labels(data_dir)
    subset = sample_balanced_dataset(labels, n_per_class=n_per_class)
    subset = add_image_paths_and_filter(subset, data_dir)
    print("Final dataset size:", len(subset))

    train_df, val_df, test_df = train_val_test_split_df(subset)
    print("Train:", train_df["morphology"].value_counts().to_dict())
    print("Val:  ", val_df["morphology"].value_counts().to_dict())
    print("Test: ", test_df["morphology"].value_counts().to_dict())

    if run_classical:
        X_train, X_val, X_test, y_train, y_val, y_test = build_classical_features(
            train_df, val_df, test_df
        )
        best_svm, y_val_pred, y_test_pred = run_classical_ml_pipeline(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        print("Tuned SVM validation accuracy:", accuracy_score(y_val, y_val_pred))
        print("Tuned SVM test accuracy:", accuracy_score(y_test, y_test_pred))
        if show_plots:
            plot_svm_confusion_matrix(y_test, y_test_pred)

    baseline_tf = get_baseline_transform()
    robust_tf = get_robust_transform()
    train_ds = GalaxyDataset(train_df, transform=baseline_tf)
    train_ds_robust = GalaxyDataset(train_df, transform=robust_tf)
    val_ds = GalaxyDataset(val_df, transform=baseline_tf)
    test_ds = GalaxyDataset(test_df, transform=baseline_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_robust = DataLoader(train_ds_robust, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Training BASELINE model")
    baseline_model = get_model(num_classes=2, device=device)
    baseline_train_loss, baseline_val_loss = train_model(
        baseline_model, train_loader, val_loader, epochs=epochs_baseline, device=device
    )
    if show_plots:
        plot_loss_curves(baseline_train_loss, baseline_val_loss, "Baseline: Train vs Validation Loss")

    print("Training ROBUST model")
    robust_model = get_model(num_classes=2, device=device)
    robust_train_loss, robust_val_loss = train_model(
        robust_model, train_loader_robust, val_loader, epochs=epochs_robust, device=device
    )
    if show_plots:
        plot_loss_curves(robust_train_loss, robust_val_loss, "Robust: Train vs Validation Loss")

    y_true_base, y_pred_base = evaluate_model(baseline_model, test_loader, device=device)
    y_true_robust, y_pred_robust = evaluate_model(robust_model, test_loader, device=device)
    print("Baseline test accuracy:", accuracy_score(y_true_base, y_pred_base))
    print("Robust test accuracy:  ", accuracy_score(y_true_robust, y_pred_robust))

    if show_plots:
        plot_dl_confusion_matrices(y_true_base, y_pred_base, y_true_robust, y_pred_robust)
        show_predictions(baseline_model, val_ds, n=6, device=device)
        show_predictions(robust_model, val_ds, n=6, device=device)

    if save_models_at_end:
        save_models(baseline_model, robust_model)
        print("Models saved to baseline_galaxy_model.pth and robust_galaxy_model.pth")


if __name__ == "__main__":
    main(
        skip_kaggle_setup=False,
        skip_download=False,
        n_per_class=N_PER_CLASS,
        epochs_baseline=10,
        epochs_robust=10,
        run_classical=True,
        save_models_at_end=True,
        show_plots=True,
    )
