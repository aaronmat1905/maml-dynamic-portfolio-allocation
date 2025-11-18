# Setup Instructions for Google Colab

## Step 1: Clone Repository in Colab

```python
# In Colab cell
!git clone https://github.com/PreethamVJ/maml-dynamic-portfolio-allocation.git
%cd maml-dynamic-portfolio-allocation
!git checkout feature/data-cleaning-pipeline
```

## Step 2: Generate Augmented Data

**Important:** The augmented R2 tasks (`.pkl` files) are not tracked in git due to `.gitignore`. 

Run this in Colab to generate them:

```python
!python scripts/augment_regime2.py --target_count 20 --methods jitter,scale,mixup
```

This creates:
- `data/augmented/augmented_regime2_tasks.pkl` (15 synthetic tasks)
- `data/processed/task_split_indices_v2_no_leakage_augmented.json` (already in repo ✅)

## Step 3: Install Dependencies

```python
!pip install torch pandas matplotlib tqdm scikit-learn scipy
```

## Step 4: Train MAML

```python
!python scripts/train_maml.py \
    --use_augmented \
    --augmented_file data/augmented/augmented_regime2_tasks.pkl \
    --epochs 50 \
    --output_dir experiments/maml_augmented
```

## Step 5: Train Baseline

```python
!python scripts/train_baseline.py \
    --epochs 50 \
    --output_dir experiments/baseline_standard
```

## Step 6: Compare Results

Use the comparison code in `notebooks/MAML_Training_Colab.ipynb`

---

## Alternative: Use Google Drive

If you prefer using Drive instead of cloning:

1. Upload your local `data/processed/` and `data/augmented/` folders to Drive
2. Mount Drive in Colab
3. Navigate to your project folder
4. Run training commands

See `notebooks/MAML_Training_Colab.ipynb` for the full Drive-based workflow.

---

## Expected Runtime

| Task | Time (Colab CPU) |
|------|------------------|
| Augmentation | 1-2 min |
| MAML training (50 epochs) | 15-20 min |
| Baseline training (50 epochs) | 5-10 min |
| **Total** | **~25 min** |

---

## Files in Repository

✅ **Included in Git:**
- All Python scripts (`scripts/*.py`)
- Model definitions (`src/models/*.py`)
- Dataset class (`src/data/task_dataset.py`)
- Augmented splits JSON (`task_split_indices_v2_no_leakage_augmented.json`)
- Colab notebook (`notebooks/MAML_Training_Colab.ipynb`)
- Documentation (`docs/Regime2_Augmentation_Guide.md`)

❌ **Not Included (generated locally or in Colab):**
- Augmented tasks pickle (`data/augmented/*.pkl`)
- Raw data CSVs (`data/processed/*.csv`, `data/raw/*.csv`)
- Segment pickles (`data/processed/*.pkl`)

**Why?** Large binary/CSV files are gitignored to keep repo lightweight. Generate them fresh or upload to Drive.
