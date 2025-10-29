import sys, os
root = r'c:\Users\Hp\Desktop\maml-dynamic-portfolio-allocation'
if root not in sys.path:
    sys.path.insert(0, root)
from src.data import feature_engineering as fe

# Try to load canonical normalized features, otherwise compute and save them
try:
    df = fe.load_normalized_features()
    print('Loaded normalized features with shape', df.shape)
except Exception as e:
    print('Normalized features not found or failed to load, running pipeline:', e)
    df = fe.run_feature_pipeline(save_filename='features_with_scaler.csv', normalize=True)
    print('Pipeline produced features with shape', df.shape)

# Choose up to 6 columns that have some non-null data
cols = [c for c in df.columns if df[c].notna().sum() > 0][:6]
print('Columns chosen for plotting:', cols)

out_dir = os.path.join(root, 'docs', 'figures')
os.makedirs(out_dir, exist_ok=True)

saved = fe.plot_features(df, cols, out_dir=out_dir)
print('Saved files count:', len(saved))
for p in saved[:10]:
    print('SAVED:', p)
