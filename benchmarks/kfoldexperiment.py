from benchmarks.pad20.folder_experiment import ex as experiment_pad_20
from sacred.observers import FileStorageObserver
import time
import argparse

CONFIG_METABLOCK_BY_MODEL = {
    'caformer_s18': 16, 
    'resnet-50': 64,
    'mobilenet': 40,
    'efficientnet-b4': 56,
}

def get_comb_config(comb_method, n_metadata, model_name):
    if comb_method:
        return [CONFIG_METABLOCK_BY_MODEL[model_name], n_metadata]
    else
        return None

if __name__=="__main__":
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run kfold validation')
    parser.add_argument('--feature-fusion', type=str, required=False, help='Method to combine metadata features [metablock, cross-attention, all]')
    parser.add_argument('--preprocess', type=str, default='onehot', help='Method to used to preprocess the metadata [onehot, sentence-embedding]')
    parser.add_argument('--missing', action='store_true', help='')
    
    args = parser.parse_args()
    benchmark = 'pad20'

    experiment = experiment_pad_20
    config_metablock = 81 if args.preprocess == 'onehot' else 768

    start_time = str(time.time()).replace('.', '')
    if args.feature_fusion == 'all':
        feature_fusion_methods = ['metablock-se', 'metablock', 'cross-attention']
    else:
        feature_fusion_methods = [args.feature_fusion]

    models = ['efficientnet-b4', 'caformer_s18', 'resnet-50', 'mobilenet', ] 
    
    optimizer = 'adam'
    best_metric = 'loss'
    missing_percentages = [0, 5, 10, 20, 30, 50, 70] if args.missing else [None]
    training_info_folder = f'opt_{optimizer}_early_stop_{best_metric}'
    _preprocessing = args.preprocess
    for _comb_method in feature_fusion_methods:

        if _comb_method:
            if _comb_method.endswith('-se'):
                _comb_method = _comb_method[:-3]
                _preprocessing = 'sentence-embedding'
            else:
                _preprocessing = 'onehot'

        metadata_comb_method = f'{_comb_method}{"-se" if _preprocessing == "sentence-embedding" else ""}' if _comb_method else 'no_metadata'
        for model_name in models:
            for missing_percentage in missing_percentages:
                folder_name = f'{training_info_folder}/{metadata_comb_method}/{start_time}/{model_name}'
                if missing_percentage is not None:
                    folder_name = f'{folder_name}/missing_{missing_percentage}'

                for folder in range(1, 6):
                    save_folder = f"benchmarks/{benchmark}/results/{folder_name}/folder_{str(folder)}"

                    experiment.observers = []
                    experiment.observers.append(FileStorageObserver.create(save_folder))
                    
                    config = {
                        "_missing_percentage": missing_percentage,
                        "_use_meta_data": _comb_method is not None,
                        "_comb_method": _comb_method,
                        "_comb_config": [CONFIG_METABLOCK_BY_MODEL[model_name], n_metadata] if _comb_method else None,
                        "_save_folder": save_folder,
                        "_folder": folder,
                        "_model_name": model_name,
                        "_sched_patience": 10,
                        "_early_stop": 15,
                        "_batch_size": 64,
                        "_optimizer": optimizer,
                        "_epochs": 100,
                        '_lr_init': 0.0001,
                        '_append_observer': False,
                        '_preprocessing': _preprocessing,
                        '_llm_type': "small",
                        '_best_metric': best_metric,
                        '_img_type': 'clinical',
                        '_initial_weights_path': None
                    }
                    experiment.run(config_updates=config)