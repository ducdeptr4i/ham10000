from pathlib import Path
import shutil
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
DEMO_DIR = BASE_DIR / "demo_samples"

TEST_CSV = DATA_DIR / "test.csv"

DEMO_DIR.mkdir(exist_ok=True)

CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

# Doc test set
df = pd.read_csv(TEST_CSV)

# Tim tat ca anh
image_files = list(
    IMAGE_DIR.rglob("*.jpg")
)

image_map = {
    p.stem: p
    for p in image_files
}

print("=" * 60)
print("CHUAN BI ANH DEMO")
print("=" * 60)

for class_name in CLASSES:

    class_df = df[
        df["dx"] == class_name
    ]

    if len(class_df) == 0:
        print(
            f"Khong co mau: {class_name}"
        )
        continue

    # Lay anh dau tien cua lop
    row = class_df.iloc[0]

    image_id = row["image_id"]

    image_path = image_map.get(
        image_id
    )

    if image_path is None:
        print(
            f"Khong tim thay anh: {image_id}"
        )
        continue

    output_name = (
        f"{class_name.upper()}_"
        f"{image_id}.jpg"
    )

    output_path = (
        DEMO_DIR /
        output_name
    )

    shutil.copy2(
        image_path,
        output_path
    )

    print(
        f"{class_name.upper():6s}"
        f" -> {output_name}"
    )

print("\nDa luu tai:")
print(DEMO_DIR)