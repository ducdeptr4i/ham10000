from pathlib import Path
from collections import Counter
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. ĐƯỜNG DẪN PROJECT
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
OUTPUT_DIR = BASE_DIR / "outputs"

METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. ĐỌC METADATA
# ==========================================

df = pd.read_csv(METADATA_PATH)

print("=" * 60)
print("KIEM TRA ANH HAM10000")
print("=" * 60)

print("\nTong so dong metadata:", len(df))

# ==========================================
# 3. TÌM TẤT CẢ FILE JPG
# ==========================================

image_files = list(IMAGE_DIR.rglob("*.jpg"))

print("Tong so file JPG tim thay:", len(image_files))

# Tạo dictionary:
# image_id -> đường dẫn ảnh
image_map = {
    img.stem: img
    for img in image_files
}

# ==========================================
# 4. KIỂM TRA ẢNH THIẾU
# ==========================================

missing_images = []

for image_id in df["image_id"]:
    if image_id not in image_map:
        missing_images.append(image_id)

print("\nSo anh khong tim thay:")
print(len(missing_images))

if len(missing_images) > 0:
    print("\nMot so anh bi thieu:")
    print(missing_images[:10])

# ==========================================
# 5. KIỂM TRA ẢNH LỖI
# ==========================================

invalid_images = []

sizes = []
modes = []

for image_id, image_path in image_map.items():

    try:
        with Image.open(image_path) as img:

            # Kiểm tra file ảnh có hợp lệ hay không
            img.verify()

        # Mở lại để đọc thông tin ảnh
        with Image.open(image_path) as img:

            sizes.append(img.size)
            modes.append(img.mode)

    except Exception as e:

        invalid_images.append(
            (image_id, str(e))
        )

print("\nSo anh loi:")
print(len(invalid_images))

if len(invalid_images) > 0:
    print("\nMot so anh loi:")
    for item in invalid_images[:10]:
        print(item)

# ==========================================
# 6. THỐNG KÊ KÍCH THƯỚC
# ==========================================

size_counter = Counter(sizes)

print("\nCac kich thuoc anh pho bien:")

for size, count in size_counter.most_common(10):
    print(size, ":", count)

print("\nSo loai kich thuoc khac nhau:")
print(len(size_counter))

# ==========================================
# 7. THỐNG KÊ COLOR MODE
# ==========================================

mode_counter = Counter(modes)

print("\nColor mode:")

for mode, count in mode_counter.items():
    print(mode, ":", count)

# ==========================================
# 8. HIỂN THỊ ẢNH MẪU CỦA 7 LỚP
# ==========================================

classes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

plt.figure(figsize=(14, 8))

for i, class_name in enumerate(classes):

    class_data = df[df["dx"] == class_name]

    if len(class_data) == 0:
        continue

    row = class_data.iloc[0]

    image_id = row["image_id"]

    if image_id not in image_map:
        continue

    image_path = image_map[image_id]

    image = Image.open(image_path).convert("RGB")

    plt.subplot(2, 4, i + 1)

    plt.imshow(image)

    plt.title(class_name.upper())

    plt.axis("off")

plt.tight_layout()

sample_output = OUTPUT_DIR / "sample_7_classes.png"

plt.savefig(
    sample_output,
    dpi=300
)

print("\nDa luu anh mau 7 lop tai:")
print(sample_output)

plt.show()