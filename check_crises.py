import pickle
import pandas as pd
import json

# Load segments
with open('data/processed/regime_segments_ma_vix_3regime.pkl', 'rb') as f:
    segs = pickle.load(f)

# Load splits
with open('data/processed/task_split_indices_v2_no_leakage_augmented.json', 'r') as f:
    splits = json.load(f)

# Filter R2 segments
r2_segs = [s for s in segs if s['regime_label'] == 2]

print("=" * 60)
print(f"REGIME 2 (CRISIS) SEGMENTS: {len(r2_segs)} total")
print("=" * 60)

for i, seg in enumerate(r2_segs):
    print(f"\nSegment {i}:")
    print(f"  Dates: {seg['start_date']} to {seg['end_date']}")
    print(f"  Length: {seg['length']} days")
    print(f"  Start idx: {seg['start_idx']}")
    
    # Determine which split it's in
    seg_tasks = []
    for task_list, split_name in [(splits['task_indices']['train_task_ids'], 'TRAIN'),
                                     (splits['task_indices']['val_task_ids'], 'VAL'),
                                     (splits['task_indices']['test_task_ids'], 'TEST')]:
        # Would need to check which tasks come from this segment
        pass
    
print("\n" + "=" * 60)
print("TASK DISTRIBUTION")
print("=" * 60)
print(f"Train R2 tasks: {splits['regime_distribution']['train']['regime_2']}")
print(f"Val R2 tasks: {splits['regime_distribution']['val']['regime_2']}")  
print(f"Test R2 tasks: {splits['regime_distribution']['test']['regime_2']}")

# Check date ranges
print("\n" + "=" * 60)
print("DATA COVERAGE")
print("=" * 60)
df = pd.read_csv('data/processed/features_imputed_with_targets.csv', index_col=0, parse_dates=True)
print(f"Dataset spans: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Total days: {len(df)}")
