from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


metrics = [
    "Accuracy",
    "Precision Macro",
    "Recall Macro",
    "Macro-F1",
    "Balanced Accuracy"
]


cnn = [
    81.36,
    72.37,
    65.37,
    66.80,
    65.37
]


vit = [
    79.63,
    68.07,
    66.31,
    64.89,
    66.31
]


x = np.arange(len(metrics))

width = 0.35


plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    cnn,
    width,
    label="ResNet50"
)

plt.bar(
    x + width / 2,
    vit,
    width,
    label="ViT-B/16"
)


plt.ylabel("Percentage (%)")

plt.title(
    "Comparison of ResNet50 and ViT-B/16 on HAM10000"
)

plt.xticks(
    x,
    metrics,
    rotation=15
)

plt.ylim(0, 100)

plt.legend()

plt.tight_layout()


output = (
    OUTPUT_DIR /
    "cnn_vs_vit.png"
)

plt.savefig(
    output,
    dpi=300
)

plt.show()

print("Da luu:")
print(output)