from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torchvision.models import resnet50

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

MODEL_PATH = OUTPUT_DIR / "best_resnet50.pth"


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("DANH GIA CNN - RESNET50")
print("=" * 60)

print("Device:", device)


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

_, _, test_loader = create_dataloaders(
    batch_size=32
)


# ============================================================
# 4. TAO LAI KIEN TRUC RESNET50
# ============================================================

model = resnet50(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    7
)

model = model.to(device)


# ============================================================
# 5. LOAD MODEL DA TRAIN
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("\nDa load model:")
print(MODEL_PATH)


# ============================================================
# 6. CHAY TEST
# ============================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

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


# ============================================================
# 7. TINH METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

balanced_acc = balanced_accuracy_score(
    all_labels,
    all_predictions
)


print("\n" + "=" * 60)
print("KET QUA TEST")
print("=" * 60)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Precision Macro:   {precision:.4f}")
print(f"Recall Macro:      {recall:.4f}")
print(f"Macro-F1:          {macro_f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(7)),
    target_names=CLASS_NAMES,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

print("\nCLASSIFICATION REPORT:")
print(report_df)

report_df.to_csv(
    OUTPUT_DIR /
    "cnn_classification_report.csv"
)


# ============================================================
# 9. LUU TONG HOP METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Model": ["ResNet50"],
    "Accuracy": [accuracy],
    "Precision_Macro": [precision],
    "Recall_Macro": [recall],
    "Macro_F1": [macro_f1],
    "Balanced_Accuracy": [balanced_acc]
})

metrics_df.to_csv(
    OUTPUT_DIR /
    "cnn_test_metrics.csv",
    index=False
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(7))
)

plt.figure(
    figsize=(9, 8)
)

plt.imshow(cm)

plt.title(
    "ResNet50 Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

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
    "cnn_confusion_matrix.png",
    dpi=300
)

plt.close()


print("\nDa luu:")

print(
    OUTPUT_DIR /
    "cnn_test_metrics.csv"
)

print(
    OUTPUT_DIR /
    "cnn_classification_report.csv"
)

print(
    OUTPUT_DIR /
    "cnn_confusion_matrix.png"
)

print("\nDANH GIA CNN HOAN THANH!")