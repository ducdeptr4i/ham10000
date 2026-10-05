from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"

OUTPUT_DIR.mkdir(exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv(METADATA_PATH)

print("=" * 60)
print("PHAN TICH BO DU LIEU HAM10000")
print("=" * 60)

print("\n1. Kich thuoc du lieu:")
print(df.shape)

print("\n2. Cac cot:")
print(df.columns.tolist())

print("\n3. Du lieu thieu:")
print(df.isnull().sum())

# Số lượng từng lớp
class_counts = df["dx"].value_counts()

print("\n4. So luong anh tung lop:")
print(class_counts)

# Tỷ lệ %
class_percent = (
    df["dx"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n5. Ty le phan tram:")
print(class_percent)

# Số image và lesion
print("\n6. So image_id:")
print(df["image_id"].nunique())

print("\n7. So lesion_id:")
print(df["lesion_id"].nunique())

# Kiểm tra trùng
print("\n8. So image_id bi trung:")
print(df["image_id"].duplicated().sum())

# Mất cân bằng
largest_class = class_counts.idxmax()
smallest_class = class_counts.idxmin()

largest_count = class_counts.max()
smallest_count = class_counts.min()

ratio = largest_count / smallest_count

print("\n9. Lop nhieu anh nhat:")
print(largest_class, largest_count)

print("\n10. Lop it anh nhat:")
print(smallest_class, smallest_count)

print("\n11. Chenh lech:")
print(round(ratio, 2), "lan")

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))

class_counts.plot(kind="bar")

plt.title("Phan bo cac lop trong HAM10000")
plt.xlabel("Loai ton thuong")
plt.ylabel("So luong anh")
plt.xticks(rotation=0)

plt.tight_layout()

output_file = OUTPUT_DIR / "class_distribution.png"

plt.savefig(output_file, dpi=300)

print("\n12. Da luu bieu do tai:")
print(output_file)

plt.show()