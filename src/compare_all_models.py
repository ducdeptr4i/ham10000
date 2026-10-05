from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# KET QUA 5 MO HINH
# ============================================================

data = {
    "Model": [
        "ResNet50",
        "ViT-B/16",
        "CNN + ViT",
        "CNN + ViT + Weighted CE",
        "CNN + ViT + Focal Loss"
    ],

    "Accuracy": [
        81.36,
        79.63,
        83.89,
        83.36,
        83.69
    ],

    "Precision Macro": [
        72.37,
        68.07,
        73.56,
        72.71,
        74.13
    ],

    "Recall Macro": [
        65.37,
        66.31,
        67.70,
        70.97,
        69.39
    ],

    "Macro-F1": [
        66.80,
        64.89,
        69.90,
        71.59,
        71.28
    ],

    "Balanced Accuracy": [
        65.37,
        66.31,
        67.70,
        70.97,
        69.39
    ]
}

df = pd.DataFrame(data)

# ============================================================
# IN BANG KET QUA
# ============================================================

print("=" * 90)
print("SO SANH 5 MO HINH")
print("=" * 90)

print(df.to_string(index=False))

# Luu CSV
csv_path = OUTPUT_DIR / "all_models_comparison.csv"

df.to_csv(
    csv_path,
    index=False
)

print("\nDa luu bang:")
print(csv_path)

# ============================================================
# BIEU DO 1 - SO SANH TOAN BO METRICS
# ============================================================

metrics = [
    "Accuracy",
    "Precision Macro",
    "Recall Macro",
    "Macro-F1",
    "Balanced Accuracy"
]

x = np.arange(len(metrics))

width = 0.15

plt.figure(figsize=(14, 7))

for i, model in enumerate(df["Model"]):

    values = df.loc[
        df["Model"] == model,
        metrics
    ].values.flatten()

    plt.bar(
        x + (i - 2) * width,
        values,
        width,
        label=model
    )

plt.xticks(
    x,
    metrics,
    rotation=10
)

plt.ylabel("Percentage (%)")

plt.title(
    "Comparison of CNN, ViT and Hybrid Models on HAM10000"
)

plt.ylim(0, 100)

plt.legend()

plt.tight_layout()

chart1 = OUTPUT_DIR / "all_models_metrics.png"

plt.savefig(
    chart1,
    dpi=300
)

plt.close()

# ============================================================
# BIEU DO 2 - MACRO F1
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["Macro-F1"]
)

plt.ylabel("Macro-F1 (%)")

plt.title(
    "Macro-F1 Comparison"
)

plt.ylim(0, 100)

plt.xticks(
    rotation=20
)

# Hien thi gia tri tren cot
for i, value in enumerate(df["Macro-F1"]):

    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

chart2 = OUTPUT_DIR / "all_models_macro_f1.png"

plt.savefig(
    chart2,
    dpi=300
)

plt.close()

# ============================================================
# BIEU DO 3 - BALANCED ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["Balanced Accuracy"]
)

plt.ylabel("Balanced Accuracy (%)")

plt.title(
    "Balanced Accuracy Comparison"
)

plt.ylim(0, 100)

plt.xticks(
    rotation=20
)

for i, value in enumerate(
    df["Balanced Accuracy"]
):

    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

chart3 = (
    OUTPUT_DIR /
    "all_models_balanced_accuracy.png"
)

plt.savefig(
    chart3,
    dpi=300
)

plt.close()

# ============================================================
# BIEU DO 4 - ACCURACY VS MACRO-F1
# ============================================================

x = np.arange(
    len(df["Model"])
)

width = 0.35

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    x - width / 2,
    df["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x + width / 2,
    df["Macro-F1"],
    width,
    label="Macro-F1"
)

plt.xticks(
    x,
    df["Model"],
    rotation=20
)

plt.ylabel(
    "Percentage (%)"
)

plt.title(
    "Accuracy vs Macro-F1"
)

plt.ylim(
    0,
    100
)

plt.legend()

plt.tight_layout()

chart4 = (
    OUTPUT_DIR /
    "accuracy_vs_macro_f1.png"
)

plt.savefig(
    chart4,
    dpi=300
)

plt.close()

# ============================================================
# KET LUAN TU DONG
# ============================================================

best_accuracy = df.loc[
    df["Accuracy"].idxmax()
]

best_f1 = df.loc[
    df["Macro-F1"].idxmax()
]

best_balanced = df.loc[
    df["Balanced Accuracy"].idxmax()
]

print("\n" + "=" * 90)
print("KET QUA NOI BAT")
print("=" * 90)

print(
    "\nAccuracy cao nhat:"
)

print(
    best_accuracy["Model"],
    "-",
    best_accuracy["Accuracy"],
    "%"
)

print(
    "\nMacro-F1 cao nhat:"
)

print(
    best_f1["Model"],
    "-",
    best_f1["Macro-F1"],
    "%"
)

print(
    "\nBalanced Accuracy cao nhat:"
)

print(
    best_balanced["Model"],
    "-",
    best_balanced["Balanced Accuracy"],
    "%"
)

print("\nDa luu cac bieu do:")

print(chart1)
print(chart2)
print(chart3)
print(chart4)