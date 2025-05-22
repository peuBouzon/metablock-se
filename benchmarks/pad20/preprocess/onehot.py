
import config
import numpy as np
import pandas as pd
from raug.raug.utils.loader import split_k_folder_csv, label_categorical_to_number
from benchmarks.pad20.dataset import PAD20
import argparse
from utils.simulatemissingdata import simulate_missing_data

def one_hot_encode(df):
    for col in df.select_dtypes(include='category').columns:
        if 'EMPTY' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories('EMPTY')
    df = df.replace(" ", np.nan).replace("  ", np.nan)
    df = df.fillna('EMPTY')
    df[PAD20.NUMERICAL_FEATURES] = df[PAD20.NUMERICAL_FEATURES].replace("EMPTY", 0).astype(float)
    df[PAD20.TARGET_COLUMN] = df[PAD20.TARGET_COLUMN].cat.remove_categories('EMPTY')

    df.loc[:, ['background_father', 'background_mother']].replace('BRASIL', 'BRAZIL', inplace=True)
    df = pd.get_dummies(df, columns=PAD20.RAW_CATEGORICAL_FEATURES, dtype=np.int8)
    return df.drop(columns=[c for c in df.columns if c.endswith('EMPTY')])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--missing', action='store_true', default=True,
                        help='Generate one-hot-encoded files with simulated missing' \
                        ' metadata with different missing percentages')
    args = parser.parse_args()

    print("- Loading the dataset")
    df = pd.read_csv(config.PAD_20_RAW_METADATA)

    print("- Splitting the dataset")
    df = split_k_folder_csv(df, PAD20.TARGET_COLUMN, save_path=None, k_folder=5, seed_number=42)

    print("- Converting the labels to numbers")
    df = label_categorical_to_number (df, PAD20.TARGET_COLUMN, col_target_number=PAD20.TARGET_NUMBER_COLUMN)

    if args.missing:
        simulate_missing_data(df, save_folder=config.PAD_20_ONE_HOT_ENCODED.parent,
                              save_raw_metadata_with_missing_values = True,
                              encoder_function=one_hot_encode,
                              encoder_name= 'one_hot',
                              features=PAD20.RAW_CATEGORICAL_FEATURES + PAD20.NUMERICAL_FEATURES,
                              dataset='pad-20')
    else:
        df = one_hot_encode(df)
        
        df.to_csv(config.PAD_20_ONE_HOT_ENCODED)

        print("- Checking the target distribution")
        print(df[PAD20.TARGET_COLUMN].value_counts())
        print(f"Total number of samples: {df[PAD20.TARGET_COLUMN].value_counts().sum()}")
