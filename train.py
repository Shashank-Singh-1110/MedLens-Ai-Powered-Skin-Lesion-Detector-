import json
import torch
import torch.nn as nn
import timm
from tqdm import tqdm

from config import (
    NUM_CLASSES, MODEL_NAME, MODEL_DIR, PROCESSED_DATA_DIR, CLASS_LIST,
)
from dataset import get_dataloaders


def get_device():
    if torch.backends.mps.is_available():
        print("   Device: Apple MPS GPU")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("   Device: CUDA GPU")
        return torch.device("cuda")
    print("   Device: CPU")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="   Train", ncols=70, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="   Eval ", ncols=70, leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train():
    print("=" * 55)
    print("  MedLens — Training (Literature-based)")
    print("=" * 55)

    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(device)
    print(f"   Model: {MODEL_NAME}, all params trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    print(f"\n🚀 Training 20 epochs (LR=1e-3, no class weights, oversampled)\n")

    for epoch in range(1, 21):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"   Epoch {epoch:2d}/20 │ "
              f"Train: {t_loss:.4f} / {t_acc:.4f} │ "
              f"Val: {v_loss:.4f} / {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": v_loss,
                "val_acc": v_acc,
                "class_list": CLASS_LIST,
            }, MODEL_DIR / "best_model.pth")
            print(f"   ✅ Saved! (val_acc: {v_acc:.4f})")
    print("\n" + "=" * 55)
    print("  FINAL TEST")
    print("=" * 55)
    ckpt = torch.load(MODEL_DIR / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n   Test Acc:  {test_acc:.4f}")
    print(f"   Best val_acc: {ckpt['val_acc']:.4f} at epoch {ckpt['epoch']}")


if __name__ == "__main__":
    train()