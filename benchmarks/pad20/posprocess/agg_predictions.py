from pathlib import Path
import pandas as pd
from itertools import product
from metrics.metrics import get_metrics
import re
import numpy as np
base_path = Path('./benchmarks/pad20/results/opt_adam_early_stop_loss/')
comb_paths = [p for p in base_path.iterdir() if p.is_dir()]

def get_subdirs(path):
    return (p for p in path.iterdir() if p.is_dir())
models = ['resnet-50', 'mobilenet', 'efficientnet-b4', 'caformer_s18']
missing = [0, 5, 10, 20, 30, 50, 70]
comb_methods = ['metablock', 'cross-attention', 'bayesiannetwork', 'metablock-se']
metric_names = ['balanced_accuracy', 'auc', 'f1_score']

index = pd.MultiIndex.from_tuples(list(product(comb_methods, models, missing, metric_names)))
df = pd.DataFrame(columns=['AVG', 'STD'] + [f'FOLDER-{i}' for i in range(1,6)], index=index)
labels = None
folder_pattern = r'folder_(\d)'
for comb_path in comb_paths:
    if comb_path.stem not in comb_methods:
        continue
    number_id_path = next(get_subdirs(comb_path))
    for model_path in get_subdirs(number_id_path):
        if model_path.stem not in models:
            continue
        for missing_path in get_subdirs(model_path):
            missing_percentage = int(missing_path.stem.replace('missing_', ''))
            if missing_percentage not in missing:
                continue
            metrics = {}
            for preds_path in missing_path.rglob('**/predictions_best_test.csv'):
                folder_number = re.search(folder_pattern, str(preds_path)).group(0).replace('folder_', '')
                df_preds = pd.read_csv(preds_path)
                labels = labels if labels is not None else df_preds['REAL'].unique()
                metrics[folder_number] = get_metrics(df_preds['REAL'], df_preds[labels])
            
            for metric in metric_names:
                row = {}
                for folder_number, values in metrics.items():
                    row[f'FOLDER-{folder_number}'] = values[metric]
                values = [v[metric] for v in metrics.values()]
                row['AVG'] = np.mean(values)
                row['STD'] = np.std(values)

                df.loc[pd.IndexSlice[comb_path.stem, model_path.stem, missing_percentage, metric]] = row
df.to_csv('./benchmarks/pad20/agg.csv')