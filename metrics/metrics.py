import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, recall_score, f1_score, precision_score
from sklearn.metrics import roc_auc_score

def get_metrics(labels, probs):
    if isinstance(probs, pd.DataFrame):
        predictions = probs.idxmax(axis=1)
    elif isinstance(probs, np.ndarray):
        predictions = probs.argmax(axis=1)
    else:
        raise ValueError("probs must be a DataFrame or a numpy array")

    sorted_labels = sorted([l for l in labels.unique()])
    return {
        'accuracy': accuracy_score(labels, predictions),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall': recall_score(labels, predictions, average='macro', zero_division=0),
        'f1_score': f1_score(labels, predictions, average='macro', zero_division=0),
        'auc': roc_auc_score(labels, probs[sorted_labels], multi_class='ovr', average='weighted', labels=sorted_labels),
    }

def aggregate_metrics(results : list, labels:list=[]):
    metrics = sorted(results[0].keys())
    mean_series = pd.Series([np.mean([result[metric] for result in results]) for metric in metrics], index=metrics)
    std_series = pd.Series([np.std([result[metric] for result in results]) for metric in metrics], index=[f'{m}_std' for m in metrics])
    return pd.concat([mean_series, std_series])