from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    TensorDataset,
    DataLoader
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

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. DUONG DAN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"

FEATURE_DIR = OUTPUT_DIR / "hybrid_features"

MODEL_PATH = OUTPUT_DIR / "best_hybrid.pth"


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("CNN + VIT HYBRID")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 3. LOAD FEATURES
# ============================================================

def load_feature_file(name):

    path = FEATURE_DIR / f"{name}_features.pt"

    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {path}"
        )

    data = torch.load(
        path,
        map_location="cpu"
    )

    return (
        data["features"],
        data["labels"]
    )


train_x, train_y = load_feature_file("train")
val_x, val_y = load_feature_file("val")
test_x, test_y = load_feature_file("test")


print("\nTrain features:")
print(train_x.shape)

print("\nValidation features:")
print(val_x.shape)

print("\nTest features:")
print(test_x.shape)


# ============================================================
# 4. DATASET + DATALOADER
# ============================================================

train_dataset = TensorDataset(
    train_x,
    train_y
)

val_dataset = TensorDataset(
    val_x,
    val_y
)

test_dataset = TensorDataset(
    test_x,
    test_y
)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# ============================================================
# 5. HYBRID CLASSIFIER
# ============================================================

class HybridClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                2816,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                7
            )
        )


    def forward(self, x):

        return self.classifier(x)


model = HybridClassifier().to(device)


# ============================================================
# 6. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=0.0001
)


# ============================================================
# 7. METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions
):

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "precision": precision_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        ),

        "f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                labels,
                predictions
            )
    }


# ============================================================
# 8. TRAIN / EVALUATE
# ============================================================

def run_epoch(
    loader,
    training=True
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0

    all_labels = []
    all_predictions = []

    if training:
        context = torch.enable_grad()
    else:
        context = torch.no_grad()

    with context:

        for features, labels in loader:

            features = features.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(
                outputs,
                labels
            )

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    avg_loss = total_loss / len(loader)

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return (
        avg_loss,
        metrics,
        all_labels,
        all_predictions
    )


# ============================================================
# 9. TRAIN MODEL
# ============================================================

NUM_EPOCHS = 20

best_val_f1 = -1.0

train_losses = []
val_losses = []


for epoch in range(NUM_EPOCHS):

    (
        train_loss,
        train_metrics,
        _,
        _
    ) = run_epoch(
        train_loader,
        training=True
    )

    (
        val_loss,
        val_metrics,
        _,
        _
    ) = run_epoch(
        val_loader,
        training=False
    )

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    print("\n" + "=" * 60)

    print(
        f"EPOCH "
        f"{epoch + 1}/{NUM_EPOCHS}"
    )

    print("=" * 60)

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"Train Macro-F1: "
        f"{train_metrics['f1']:.4f}"
    )

    print("\nVALIDATION")

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"Val Precision Macro: "
        f"{val_metrics['precision']:.4f}"
    )

    print(
        f"Val Recall Macro: "
        f"{val_metrics['recall']:.4f}"
    )

    print(
        f"Val Macro-F1: "
        f"{val_metrics['f1']:.4f}"
    )

    print(
        f"Val Balanced Accuracy: "
        f"{val_metrics['balanced_accuracy']:.4f}"
    )


    # Luu model tot nhat theo Validation Macro-F1
    if val_metrics["f1"] > best_val_f1:

        best_val_f1 = val_metrics["f1"]

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "\nDa luu Hybrid tot nhat!"
        )


# ============================================================
# 10. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


# ============================================================
# 11. TEST
# ============================================================

(
    test_loss,
    test_metrics,
    test_labels,
    test_predictions
) = run_epoch(
    test_loader,
    training=False
)


print("\n" + "=" * 60)
print("HYBRID TEST RESULTS")
print("=" * 60)

print(
    f"Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Accuracy: "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Precision Macro: "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Recall Macro: "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"Macro-F1: "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{test_metrics['balanced_accuracy']:.4f}"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

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
# 13. CONFUSION MATRIX
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
    "CNN + ViT Confusion Matrix"
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
    "hybrid_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

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
    "Hybrid Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "hybrid_loss.png",
    dpi=300
)

plt.close()


print("\nDa luu:")

print(MODEL_PATH)

print(
    OUTPUT_DIR /
    "hybrid_confusion_matrix.png"
)

print(
    OUTPUT_DIR /
    "hybrid_loss.png"
)

print("\nHOAN THANH CNN + VIT HYBRID!")