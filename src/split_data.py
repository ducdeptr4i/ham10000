from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. ĐƯỜNG DẪN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"

# ==========================================
# 2. ĐỌC METADATA
# ==========================================

df = pd.read_csv(METADATA_PATH)

print("=" * 60)
print("CHIA DU LIEU HAM10000")
print("=" * 60)

print("\nTong so anh:", len(df))
print("Tong so lesion:", df["lesion_id"].nunique())

# ==========================================
# 3. KIỂM TRA MỖI LESION CHỈ CÓ 1 NHÃN
# ==========================================

label_per_lesion = (
    df.groupby("lesion_id")["dx"]
    .nunique()
)

invalid_lesions = label_per_lesion[
    label_per_lesion > 1
]

print("\nSo lesion co nhieu hon 1 nhan:")
print(len(invalid_lesions))

if len(invalid_lesions) > 0:
    print("Can kiem tra lai du lieu!")
    print(invalid_lesions)
    raise ValueError(
        "Phat hien lesion_id co nhieu nhan khac nhau."
    )

# ==========================================
# 4. TẠO BẢNG LESION
# ==========================================

lesion_df = (
    df.groupby("lesion_id")
    .agg(
        dx=("dx", "first")
    )
    .reset_index()
)

print("\nSo lesion dung de chia:")
print(len(lesion_df))

print("\nPhan bo lesion theo lop:")
print(lesion_df["dx"].value_counts())

# ==========================================
# 5. CHIA TRAIN 70% - TEMP 30%
# ==========================================

train_lesions, temp_lesions = train_test_split(
    lesion_df,
    test_size=0.30,
    random_state=42,
    stratify=lesion_df["dx"]
)

# ==========================================
# 6. CHIA TEMP THÀNH VAL 15% - TEST 15%
# ==========================================

val_lesions, test_lesions = train_test_split(
    temp_lesions,
    test_size=0.50,
    random_state=42,
    stratify=temp_lesions["dx"]
)

# ==========================================
# 7. LẤY TẤT CẢ ẢNH THEO LESION
# ==========================================

train_df = df[
    df["lesion_id"].isin(
        train_lesions["lesion_id"]
    )
].copy()

val_df = df[
    df["lesion_id"].isin(
        val_lesions["lesion_id"]
    )
].copy()

test_df = df[
    df["lesion_id"].isin(
        test_lesions["lesion_id"]
    )
].copy()

# ==========================================
# 8. HIỂN THỊ KẾT QUẢ
# ==========================================

print("\n" + "=" * 60)
print("KET QUA CHIA DU LIEU")
print("=" * 60)

print("\nTRAIN")
print("So anh:", len(train_df))
print("So lesion:", train_df["lesion_id"].nunique())

print("\nVALIDATION")
print("So anh:", len(val_df))
print("So lesion:", val_df["lesion_id"].nunique())

print("\nTEST")
print("So anh:", len(test_df))
print("So lesion:", test_df["lesion_id"].nunique())

# ==========================================
# 9. PHÂN BỐ LỚP
# ==========================================

print("\n--- TRAIN ---")
print(train_df["dx"].value_counts())
print(
    train_df["dx"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n--- VALIDATION ---")
print(val_df["dx"].value_counts())
print(
    val_df["dx"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n--- TEST ---")
print(test_df["dx"].value_counts())
print(
    test_df["dx"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ==========================================
# 10. KIỂM TRA DATA LEAKAGE
# ==========================================

train_ids = set(train_df["lesion_id"])
val_ids = set(val_df["lesion_id"])
test_ids = set(test_df["lesion_id"])

train_val_overlap = train_ids.intersection(val_ids)
train_test_overlap = train_ids.intersection(test_ids)
val_test_overlap = val_ids.intersection(test_ids)

print("\n" + "=" * 60)
print("KIEM TRA DATA LEAKAGE")
print("=" * 60)

print(
    "Train <-> Validation:",
    len(train_val_overlap)
)

print(
    "Train <-> Test:",
    len(train_test_overlap)
)

print(
    "Validation <-> Test:",
    len(val_test_overlap)
)

# ==========================================
# 11. LƯU CSV
# ==========================================

train_path = DATA_DIR / "train.csv"
val_path = DATA_DIR / "val.csv"
test_path = DATA_DIR / "test.csv"

train_df.to_csv(
    train_path,
    index=False
)

val_df.to_csv(
    val_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)

print("\nDa luu:")

print(train_path)
print(val_path)
print(test_path)
