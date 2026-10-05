from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from dataset import (
    create_dataloaders,
    CLASS_NAMES
)


# ============================================================
# 1. DUONG DAN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "best_resnet50.pth"


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("CNN BASELINE - RESNET50")
print("=" * 60)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 3. DATALOADER
# ============================================================

train_loader, val_loader, test_loader = (
    create_dataloaders(
        batch_size=32
    )
)


# ============================================================
# 4. TAO RESNET50
# ============================================================

print("\nDang khoi tao ResNet50...")

weights = ResNet50_Weights.DEFAULT

model = resnet50(
    weights=weights
)

# So dac trung dau vao cua lop cuoi
num_features = model.fc.in_features

# Thay lop cuoi thanh 7 lop HAM10000
model.fc = nn.Linear(
    num_features,
    7
)

model = model.to(device)

print("Da tao ResNet50 voi 7 lop.")


# ============================================================
# 5. LOSS FUNCTION
# ============================================================

# Baseline truoc tien dung CrossEntropyLoss binh thuong
criterion = nn.CrossEntropyLoss()


# ============================================================
# 6. OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ============================================================
# 7. HAM TINH METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions
):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        labels,
        predictions
    )

    return (
        accuracy,
        precision,
        recall,
        macro_f1,
        balanced_acc
    )


# ============================================================
# 8. TRAIN 1 EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader
):

    model.train()

    total_loss = 0

    all_labels = []
    all_predictions = []

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        # Xoa gradient cu
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Cap nhat trong so
        optimizer.step()

        total_loss += loss.item()

        # Lay nhan du doan
        _, predictions = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Hien thi tien do
        if (batch_idx + 1) % 50 == 0:

            print(
                f"Batch "
                f"{batch_idx + 1}/"
                f"{len(loader)} "
                f"- Loss: "
                f"{loss.item():.4f}"
            )

    average_loss = (
        total_loss / len(loader)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return average_loss, metrics


# ============================================================
# 9. VALIDATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    total_loss = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            _, predictions = torch.max(
                outputs,
                1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = (
        total_loss / len(loader)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return (
        average_loss,
        metrics,
        all_labels,
        all_predictions
    )


# ============================================================
# 10. TRAIN MODEL
# ============================================================

# Luc dau nen test 5 epoch.
# Khi pipeline on dinh co the tang len 15-30.
NUM_EPOCHS = 5

train_losses = []
val_losses = []

train_f1_scores = []
val_f1_scores = []

best_val_f1 = 0.0


for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(
        f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
    )
    print("=" * 60)

    # -------------------------
    # TRAIN
    # -------------------------

    train_loss, train_metrics = (
        train_one_epoch(
            model,
            train_loader
        )
    )

    (
        train_acc,
        train_precision,
        train_recall,
        train_f1,
        train_bal_acc
    ) = train_metrics


    # -------------------------
    # VALIDATION
    # -------------------------

    (
        val_loss,
        val_metrics,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )

    (
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        val_bal_acc
    ) = val_metrics


    # -------------------------
    # LUU LICH SU
    # -------------------------

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_f1_scores.append(
        train_f1
    )

    val_f1_scores.append(
        val_f1
    )


    # -------------------------
    # HIEN THI KET QUA
    # -------------------------

    print("\nTRAIN:")

    print(
        f"Loss: {train_loss:.4f}"
    )

    print(
        f"Accuracy: "
        f"{train_acc:.4f}"
    )

    print(
        f"Macro F1: "
        f"{train_f1:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{train_bal_acc:.4f}"
    )


    print("\nVALIDATION:")

    print(
        f"Loss: {val_loss:.4f}"
    )

    print(
        f"Accuracy: "
        f"{val_acc:.4f}"
    )

    print(
        f"Precision Macro: "
        f"{val_precision:.4f}"
    )

    print(
        f"Recall Macro: "
        f"{val_recall:.4f}"
    )

    print(
        f"Macro F1: "
        f"{val_f1:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{val_bal_acc:.4f}"
    )


    # -------------------------
    # LUU MODEL TOT NHAT
    # -------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "\nDa luu model tot nhat!"
        )

        print(
            "Validation Macro-F1:",
            round(
                best_val_f1,
                4
            )
        )


# ============================================================
# 11. VE LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    train_losses,
    label="Train Loss"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ResNet50 Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cnn_loss.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. VE MACRO F1
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    train_f1_scores,
    label="Train Macro-F1"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    val_f1_scores,
    label="Validation Macro-F1"
)

plt.xlabel("Epoch")
plt.ylabel("Macro-F1")

plt.title(
    "ResNet50 Macro-F1"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cnn_macro_f1.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. LOAD MODEL TOT NHAT
# ============================================================

print("\n" + "=" * 60)
print("DANH GIA TREN TEST SET")
print("=" * 60)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


# ============================================================
# 14. TEST MODEL
# ============================================================

(
    test_loss,
    test_metrics,
    test_labels,
    test_predictions
) = evaluate(
    model,
    test_loader
)

(
    test_acc,
    test_precision,
    test_recall,
    test_f1,
    test_bal_acc
) = test_metrics


print("\nTEST RESULTS:")

print(
    f"Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Accuracy: "
    f"{test_acc:.4f}"
)

print(
    f"Precision Macro: "
    f"{test_precision:.4f}"
)

print(
    f"Recall Macro: "
    f"{test_recall:.4f}"
)

print(
    f"Macro-F1: "
    f"{test_f1:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{test_bal_acc:.4f}"
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT:")

print(
    classification_report(
        test_labels,
        test_predictions,
        labels=list(
            range(len(CLASS_NAMES))
        ),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=list(
        range(len(CLASS_NAMES))
    )
)

plt.figure(figsize=(9, 8))

plt.imshow(cm)

plt.title(
    "ResNet50 Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    np.arange(
        len(CLASS_NAMES)
    ),
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    np.arange(
        len(CLASS_NAMES)
    ),
    CLASS_NAMES
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "cnn_confusion_matrix.png",
    dpi=300
)

plt.close()


print("\nDa luu cac ket qua:")

print(
    OUTPUT_DIR /
    "cnn_loss.png"
)

print(
    OUTPUT_DIR /
    "cnn_macro_f1.png"
)

print(
    OUTPUT_DIR /
    "cnn_confusion_matrix.png"
)

print(
    MODEL_PATH
)

print("\nHOAN THANH CNN BASELINE!")