from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


classes = [
    "AKIEC",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "VASC"
]


cnn_f1 = [
    62.71,
    74.67,
    58.36,
    68.29,
    42.74,
    90.53,
    70.27
]


vit_f1 = [
    52,
    66,
    47,
    63,
    58,
    91,
    76
]


x = np.arange(
    len(classes)
)

width = 0.35


plt.figure(
    figsize=(11, 6)
)


plt.bar(
    x - width / 2,
    cnn_f1,
    width,
    label="ResNet50"
)

plt.bar(
    x + width / 2,
    vit_f1,
    width,
    label="ViT-B/16"
)


plt.ylabel(
    "F1-score (%)"
)

plt.xlabel(
    "Skin lesion class"
)

plt.title(
    "F1-score by class: CNN vs ViT"
)

plt.xticks(
    x,
    classes
)

plt.ylim(
    0,
    100
)

plt.legend()

plt.tight_layout()


output = (
    OUTPUT_DIR /
    "cnn_vit_f1_by_class.png"
)

plt.savefig(
    output,
    dpi=300
)

plt.show()

print(output)