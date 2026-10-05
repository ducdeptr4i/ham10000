from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.models import (
    resnet50,
    vit_b_16
)

from dataset import (
    HAM10000Dataset,
    create_image_map,
    val_test_transform
)


# ============================================================
# 1. DUONG DAN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

CNN_PATH = OUTPUT_DIR / "best_resnet50.pth"
VIT_PATH = OUTPUT_DIR / "best_vit_b16.pth"

FEATURE_DIR = OUTPUT_DIR / "hybrid_features"

FEATURE_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("TRICH XUAT CNN + VIT FEATURES")
print("=" * 60)

print("Device:", device)


# ============================================================
# 3. LOAD RESNET50
# ============================================================

print("\nDang load ResNet50...")

cnn = resnet50(
    weights=None
)

cnn_features = cnn.fc.in_features

cnn.fc = nn.Linear(
    cnn_features,
    7
)

cnn.load_state_dict(
    torch.load(
        CNN_PATH,
        map_location=device
    )
)

# Bo classifier de lay feature 2048
cnn.fc = nn.Identity()

cnn = cnn.to(device)

cnn.eval()

for param in cnn.parameters():
    param.requires_grad = False


# ============================================================
# 4. LOAD VIT-B/16
# ============================================================

print("Dang load ViT-B/16...")

vit = vit_b_16(
    weights=None
)

vit_features = (
    vit.heads.head.in_features
)

vit.heads.head = nn.Linear(
    vit_features,
    7
)

vit.load_state_dict(
    torch.load(
        VIT_PATH,
        map_location=device
    )
)

# Bo classifier de lay vector 768
vit.heads = nn.Identity()

vit = vit.to(device)

vit.eval()

for param in vit.parameters():
    param.requires_grad = False


print(
    "CNN feature:",
    cnn_features
)

print(
    "ViT feature:",
    vit_features
)

print(
    "Fusion feature:",
    cnn_features + vit_features
)


# ============================================================
# 5. TAO DATALOADER
# ============================================================

image_map = create_image_map()


def create_loader(csv_path):

    df = pd.read_csv(
        csv_path
    )

    # Dung transform deterministic
    # de feature khong thay doi ngau nhien
    dataset = HAM10000Dataset(
        dataframe=df,
        image_map=image_map,
        transform=val_test_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    return loader


train_loader = create_loader(
    DATA_DIR / "train.csv"
)

val_loader = create_loader(
    DATA_DIR / "val.csv"
)

test_loader = create_loader(
    DATA_DIR / "test.csv"
)


# ============================================================
# 6. HAM TRICH XUAT FEATURE
# ============================================================

def extract_features(
    loader,
    split_name
):

    all_features = []
    all_labels = []

    total_batches = len(loader)

    print(
        f"\nDang xu ly {split_name}..."
    )

    with torch.no_grad():

        for batch_idx, (
            images,
            labels
        ) in enumerate(loader):

            images = images.to(device)

            # CNN feature
            cnn_feat = cnn(images)

            # ViT feature
            vit_feat = vit(images)

            # Feature Fusion
            fused_feat = torch.cat(
                (
                    cnn_feat,
                    vit_feat
                ),
                dim=1
            )

            all_features.append(
                fused_feat.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

            if (
                batch_idx + 1
            ) % 100 == 0:

                print(
                    f"{batch_idx + 1}"
                    f"/{total_batches}"
                )

    features = torch.cat(
        all_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    print(
        f"{split_name} feature shape:",
        features.shape
    )

    print(
        f"{split_name} label shape:",
        labels.shape
    )

    torch.save(
        {
            "features": features,
            "labels": labels
        },
        FEATURE_DIR /
        f"{split_name}_features.pt"
    )


# ============================================================
# 7. EXTRACT
# ============================================================

extract_features(
    train_loader,
    "train"
)

extract_features(
    val_loader,
    "val"
)

extract_features(
    test_loader,
    "test"
)

print("\nHOAN THANH TRICH XUAT FEATURE!")
print("Thu muc:")
print(FEATURE_DIR)