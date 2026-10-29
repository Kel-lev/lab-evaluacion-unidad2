import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import get_dataloaders, set_seed, SEED, CIFAR10_CLASSES
from model import CNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if return_preds:
                all_targets.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    if return_preds:
        return epoch_loss, epoch_acc, np.array(all_targets), np.array(all_preds)

    return epoch_loss, epoch_acc


def main():
    # Semilla individual: últimos dígitos 68
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="./data",
        batch_size=64,
        val_size=0.2,
        seed=SEED,
        num_workers=2
    )

    model = CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 15

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Época {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Cargar el mejor modelo según validación
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device, return_preds=True
    )

    print("=" * 60)
    print(f"Mejor Val Acc: {best_val_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)

    # Curvas de aprendizaje: Loss y Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title("Curva de Loss")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Accuracy", marker="o")
    axes[1].plot(history["val_acc"], label="Val Accuracy", marker="s")
    axes[1].set_title("Curva de Accuracy")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/loss_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Matriz de confusión sobre test
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 9))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CIFAR10_CLASSES
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False
    )
    plt.title("Matriz de confusión - CIFAR-10 - Semilla 68")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Guardar métricas en texto para el análisis del README
    with open("outputs/metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Semilla: {SEED}\n")
        f.write(f"Mejor Val Acc: {best_val_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Acc: {test_acc:.4f}\n")
        f.write(f"Train Loss final: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Train Acc final: {history['train_acc'][-1]:.4f}\n")
        f.write(f"Val Loss final: {history['val_loss'][-1]:.4f}\n")
        f.write(f"Val Acc final: {history['val_acc'][-1]:.4f}\n")

    print("Artefactos guardados en outputs/:")
    print("- outputs/loss_accuracy.png")
    print("- outputs/confusion_matrix.png")
    print("- outputs/metrics.txt")


if __name__ == "__main__":
    main()