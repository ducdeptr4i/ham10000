from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights
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

MODEL_PATH = OUTPUT_DIR / "best_vit_b16.pth"


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("VISION TRANSFORMER BASELINE - VIT-B/16")
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

# ViT nang hon ResNet50.
# Ban dau dung batch_size = 8 cho an toan.
train_loader, val_loader, test_loader = (
    create_dataloaders(
        batch_size=8
    )
)


# ============================================================
# 4. TAO VIT-B/16 PRETRAINED
# ============================================================

print("\nDang khoi tao ViT-B/16...")

weights = ViT_B_16_Weights.DEFAULT

model = vit_b_16(
    weights=weights
)

# Lay so feature truoc classifier
num_features = model.heads.head.in_features

# Thay classifier ImageNet thanh 7 lop HAM10000
model.heads.head = nn.Linear(
    num_features,
    7
)

model = model.to(device)

print("Da tao ViT-B/16 voi 7 lop.")


# ============================================================
# 5. LOSS
# ============================================================

# Baseline dung CrossEntropyLoss binh thuong
criterion = nn.CrossEntropyLoss()


# ============================================================
# 6. OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.00001,
    weight_decay=0.01
)


# ============================================================
# 7. METRICS
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

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

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

        if (batch_idx + 1) % 100 == 0:

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
# 9. VALIDATION / TEST
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
# 10. TRAIN
# ============================================================

# Dau tien chi test 1 epoch.
NUM_EPOCHS = 5

train_losses = []
val_losses = []

train_f1_scores = []
val_f1_scores = []

best_val_f1 = -1.0


for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(
        f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
    )
    print("=" * 60)

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


    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_f1_scores.append(train_f1)
    val_f1_scores.append(val_f1)


    print("\nTRAIN:")

    print(
        f"Loss: {train_loss:.4f}"
    )

    print(
        f"Accuracy: {train_acc:.4f}"
    )

    print(
        f"Macro-F1: {train_f1:.4f}"
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
        f"Accuracy: {val_acc:.4f}"
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
        f"Macro-F1: {val_f1:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{val_bal_acc:.4f}"
    )


    # Luu model tot nhat theo Macro-F1
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "\nDa luu ViT tot nhat!"
        )


# ============================================================
# 11. LOAD MODEL TOT NHAT
# ============================================================

print("\n" + "=" * 60)
print("DANH GIA VIT TREN TEST SET")
print("=" * 60)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


# ============================================================
# 12. TEST
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
    f"Loss: {test_loss:.4f}"
)

print(
    f"Accuracy: {test_acc:.4f}"
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
    f"Macro-F1: {test_f1:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{test_bal_acc:.4f}"
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT:")

print(
    classification_report(
        test_labels,
        test_predictions,
        labels=list(range(7)),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=list(range(7))
)

plt.figure(
    figsize=(9, 8)
)

plt.imshow(cm)

plt.title(
    "ViT-B/16 Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    np.arange(7),
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    np.arange(7),
    CLASS_NAMES
)

for i in range(7):

    for j in range(7):

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
    "vit_confusion_matrix.png",
    dpi=300
)

plt.close()


print("\nDa luu model:")
print(MODEL_PATH)

print("\nHOAN THANH VIT BASELINE!")