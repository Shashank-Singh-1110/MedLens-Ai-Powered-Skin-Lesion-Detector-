import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASSES, CLASS_LIST,
    IMG_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
)


def load_data():
    df = pd.read_csv(RAW_DATA_DIR / "HAM10000_metadata.csv")
    print(f"✅ Metadata: {len(df)} rows")
    image_map = {p.stem: p for p in RAW_DATA_DIR.rglob("*.jpg")}
    print(f"✅ Images on disk: {len(image_map)}")
    df = df[df["image_id"].isin(image_map)].copy()
    return df, image_map


def run_eda(df):

    print("\n" + "=" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    total = len(df)
    print("\n📊 Class Distribution:\n")
    class_counts = df["dx"].value_counts()

    for cls, count in class_counts.items():
        pct = count / total * 100
        info = CLASSES[cls]
        bar = "█" * int(pct / 2)
        print(f"   {cls:6s} | {count:5d} ({pct:5.1f}%) | {bar} | {info['name']} [{info['severity']}]")

    imbalance = class_counts.max() / class_counts.min()
    print(f"\n   ⚖️  Imbalance ratio: {imbalance:.1f}x")
    print(f"   Largest:  {class_counts.idxmax()} ({class_counts.max()})")
    print(f"   Smallest: {class_counts.idxmin()} ({class_counts.min()})")
    print(f"\n📊 Unique lesions: {df['lesion_id'].nunique()}")
    print(f"   Total images:   {total}")
    print(f"   ⚠️  Some lesions have multiple photos — split by lesion_id!")
    print(f"\n👤 Age: {df['age'].mean():.1f} ± {df['age'].std():.1f} (missing: {df['age'].isna().sum()})")
    print(f"   Sex: {dict(df['sex'].value_counts())}")
    print(f"\n🔬 Diagnosis method:")
    for method, count in df["dx_type"].value_counts().items():
        print(f"   {method}: {count} ({count/total*100:.1f}%)")

def create_splits(df):

    print("\n" + "=" * 55)
    print("  CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 55)

    lesion_to_class = df.groupby("lesion_id")["dx"].first()
    lesion_ids = lesion_to_class.index.values
    lesion_classes = lesion_to_class.values

    print(f"\n   Splitting {len(lesion_ids)} lesions (70/15/15)...")
    train_lesions, temp_lesions, _, temp_classes = train_test_split(
        lesion_ids, lesion_classes,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=lesion_classes,
        random_state=RANDOM_SEED,
    )

    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_classes,
        random_state=RANDOM_SEED,
    )
    train_set = set(train_lesions)
    val_set = set(val_lesions)

    df["split"] = df["lesion_id"].apply(
        lambda lid: "train" if lid in train_set
        else ("val" if lid in val_set else "test")
    )
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        print(f"\n   📂 {split}: {len(split_df)} images")
        for cls in CLASS_LIST:
            count = (split_df["dx"] == cls).sum()
            print(f"      {cls}: {count}")

    return df

def organize_images(df, image_map):
    print("\n" + "=" * 55)
    print(f"  RESIZING TO {IMG_SIZE}x{IMG_SIZE} & ORGANIZING")
    print("=" * 55)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]

        for _, row in tqdm(split_df.iterrows(), total=len(split_df),
                           desc=f"   {split}", ncols=70):

            img_id = row["image_id"]
            cls = row["dx"]
            src = image_map.get(img_id)
            if src is None:
                continue

            dst_dir = PROCESSED_DATA_DIR / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{img_id}.jpg"

            try:
                with Image.open(src) as img:
                    img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS).save(
                        dst, "JPEG", quality=95
                    )
            except Exception as e:
                print(f"\n   ⚠️ Failed: {img_id} — {e}")

    print("\n✅ Done! Final structure:")
    for split in ["train", "val", "test"]:
        split_dir = PROCESSED_DATA_DIR / split
        n_images = sum(1 for _ in split_dir.rglob("*.jpg"))
        print(f"   {split}/ → {n_images} images across 7 classes")

def compute_class_weights(df):
    train_df = df[df["split"] == "train"]
    counts = train_df["dx"].value_counts()
    total = len(train_df)
    n_classes = len(counts)

    weights = {}
    for cls, count in counts.items():
        weights[cls] = total / (n_classes * count)
    min_w = min(weights.values())
    weights = {k: round(v / min_w, 4) for k, v in weights.items()}

    print("\n⚖️  Class Weights for Training:")
    for cls in CLASS_LIST:
        bar = "█" * int(weights[cls])
        print(f"   {cls:6s} → {weights[cls]:6.2f}  {bar}")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "class_weights.json"
    with open(out_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\n   Saved → {out_path}")

    return weights

if __name__ == "__main__":
    print("=" * 55)
    print("  MedLens — Preprocessing Pipeline")
    print("=" * 55)

    df, image_map = load_data()
    run_eda(df)
    df = create_splits(df)
    organize_images(df, image_map)
    compute_class_weights(df)
