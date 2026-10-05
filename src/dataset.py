from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# 1. DUONG DAN PROJECT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"


# ============================================================
# 2. NHAN CUA 7 LOP HAM10000
# ============================================================

LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# ============================================================
# 3. TIEN XU LY TAP TRAIN
# ============================================================

train_transform = transforms.Compose([

    # Dua tat ca anh ve cung kich thuoc
    transforms.Resize((224, 224)),

    # Tang cuong du lieu
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=20
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),

    # Chuyen anh thanh Tensor
    transforms.ToTensor(),

    # Chuan hoa theo ImageNet
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. TIEN XU LY VALIDATION / TEST
# ============================================================

val_test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. HAM10000 DATASET
# ============================================================

class HAM10000Dataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_map,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.image_map = image_map

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_id = row["image_id"]

        label_name = row["dx"]

        # Tim duong dan anh
        image_path = self.image_map.get(
            image_id
        )

        if image_path is None:

            raise FileNotFoundError(
                f"Khong tim thay anh: {image_id}"
            )

        # Doc anh
        image = Image.open(
            image_path
        ).convert("RGB")

        # Tien xu ly
        if self.transform is not None:

            image = self.transform(image)

        # Chuyen label thanh so
        label = LABEL_MAP[label_name]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# 6. TAO BAN DO IMAGE_ID -> IMAGE PATH
# ============================================================

def create_image_map():

    image_files = list(
        IMAGE_DIR.rglob("*.jpg")
    )

    image_map = {
        image_path.stem: image_path
        for image_path in image_files
    }

    return image_map


# ============================================================
# 7. TAO DATALOADER
# ============================================================

def create_dataloaders(
    batch_size=32
):

    print("=" * 60)
    print("TAO HAM10000 DATALOADER")
    print("=" * 60)

    # Doc CSV
    train_df = pd.read_csv(
        TRAIN_CSV
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    test_df = pd.read_csv(
        TEST_CSV
    )

    # Tim tat ca anh
    image_map = create_image_map()

    print(
        "\nTong so anh tim thay:",
        len(image_map)
    )

    # Tao Dataset
    train_dataset = HAM10000Dataset(
        dataframe=train_df,
        image_map=image_map,
        transform=train_transform
    )

    val_dataset = HAM10000Dataset(
        dataframe=val_df,
        image_map=image_map,
        transform=val_test_transform
    )

    test_dataset = HAM10000Dataset(
        dataframe=test_df,
        image_map=image_map,
        transform=val_test_transform
    )

    # Tao DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,

        # Windows: ban dau de 0 cho de chay
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(
        "Train images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    print(
        "Test images:",
        len(test_dataset)
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# 8. CHAY THU
# ============================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=32
        )
    )

    # Lay thu 1 batch
    images, labels = next(
        iter(train_loader)
    )

    print("\nKich thuoc batch anh:")
    print(images.shape)

    print("\nKich thuoc batch label:")
    print(labels.shape)

    print("\nNhan cua batch dau:")
    print(labels)

    print("\nMin pixel sau normalize:")
    print(images.min().item())

    print("\nMax pixel sau normalize:")
    print(images.max().item())

    print("\nDATASET HOAT DONG BINH THUONG!")