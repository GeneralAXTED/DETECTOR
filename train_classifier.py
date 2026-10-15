r"""
🎓 SMART CLASSROOM DETECTOR - Maxsus AI Tasniflagichini O'rgatish (Fine-tuning)
Ushbu skript `dataset/crops` papkasidagi to'plangan suratlar asosida
MobileNetV3 neyron tarmog'ini o'rgatadi va `models/behavior_classifier.pt` fayliga saqlaydi.

Ishlatish:
    .\.venv\Scripts\python.exe train_classifier.py --epochs 15 --batch_size 16
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("TRAIN")

def train_model(epochs: int = 15, batch_size: int = 16, lr: float = 0.0005):
    dataset_dir = Path("dataset/crops")
    if not dataset_dir.exists():
        logger.error(f"Dataset papkasi topilmadi: {dataset_dir}")
        return

    # Klass papkalaridagi suratlar sonini tekshirish
    total_imgs = 0
    classes = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    for cls in classes:
        count = len(list((dataset_dir / cls).glob("*.jpg"))) + len(list((dataset_dir / cls).glob("*.png")))
        logger.info(f"Klass [{cls}]: {count} ta surat mavjud.")
        total_imgs += count

    if total_imgs < 10:
        logger.warning(
            "Datasetda suratlar juda kam (kamida 20-30 ta kerak). "
            "Tizimni biroz ishlatib, 'dataset/crops' ichidagi papkalarga haqiqiy auditoriya "
            "kadrlaridan namunalar to'plangandan so'ng qayta o'rgating."
        )
        # Agar yetarli bo'lmasa, dastlabki transfer learning vaznlarini saqlab qo'yamiz
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"O'rgatish qurilmasi: {device}")

    # Data Augmentation (Auditoriyadagi yorug'lik va burchaklarga chidamlilik uchun)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = ImageFolder(root=str(dataset_dir), transform=train_transform)
    num_classes = len(full_dataset.classes)
    logger.info(f"Jami klasslar soni: {num_classes} ({full_dataset.classes})")

    # Train / Val (80 / 20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Pre-trained MobileNetV3-Small (Transfer Learning)
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "behavior_classifier.pt"

    logger.info(f"O'rgatish boshlandi: {epochs} ta davr (epochs)...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        scheduler.step()
        train_acc = (correct / total) * 100 if total > 0 else 0.0
        train_loss = running_loss / total if total > 0 else 0.0

        # Validatsiya
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                v_correct += torch.sum(preds == labels.data).item()
                v_total += labels.size(0)

        val_acc = (v_correct / v_total) * 100 if v_total > 0 else train_acc
        logger.info(f"Davr {epoch:02d}/{epochs:02d} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Loss: {train_loss:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), str(out_path))
            logger.info(f"🔥 Eng yaxshi model saqlandi -> {out_path} ({best_acc:.1f}%)")

    logger.info(f"Muvaffaqiyatli yakunlandi! Model tayyor: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15, help="Trening davrlari soni")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch o'lchami")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
