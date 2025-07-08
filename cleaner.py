import pandas as pd
import os

# List of parquet file paths (only those with 'preds' in name)
pred_files = [
    "output/ChestXRayCNN_ChestXRay_224_e10_preds.parquet",
    "output/ChestXRayCNN_ChestXRay_512_e10_preds.parquet",
    "output/ChestXRayCNN_ChestXRay_224_e10_rmods_preds.parquet",
    "output/ChestXRayCNN_ChestXRay_224_e10_rmods_norm_preds.parquet",
    "output/ChestXRayResNet_ChestXRay_preds_e5.parquet",
    "output/ChestXRayResNet_ChestXRay_preds_b7.parquet",
    "output/ChestXRaySVM_ChestXRay_preds.parquet",
    "output/ChestXRaySvmRFF_ChestXRay_preds.parquet",
]

for path in pred_files:
    print(f"Processing: {path}")
    df = pd.read_parquet(path)

    # Drop all columns that start with 'embed_'
    embed_cols = [col for col in df.columns if col.startswith("embed_")]
    df.drop(columns=embed_cols, inplace=True)

    # Save back to the same file
    df.to_parquet(path, index=False)
    print(f"Cleaned and saved: {path}")