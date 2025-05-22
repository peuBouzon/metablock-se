
import config
import argparse
import pandas as pd
from itertools import chain
from benchmarks.pad20.dataset import PAD20
from raug.raug.utils.loader import split_k_folder_csv, label_categorical_to_number
from utils.simulatemissingdata import simulate_missing_data

def generate_sentence(df:pd.DataFrame):
    df = df.reset_index(drop=True)
    for col in df.select_dtypes(include='category').columns:
        if 'EMPTY' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories('EMPTY')
    df.fillna("EMPTY", inplace=True)

    sentences = []
    for _, row in df.iterrows():
        anamnese = ""
        for col in chain(PAD20.RAW_CATEGORICAL_FEATURES, PAD20.NUMERICAL_FEATURES):
            anamnese += f"{col}: {row[col]}, "  
        sentences.append(anamnese)
    df['sentence'] = pd.Series(sentences)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--missing', action='store_true', default=True,
                        help='Generate sentences with simulated missing' \
                        ' metadata with different missing percentages')
    args = parser.parse_args()

    print("- Loading the dataset")
    df = pd.read_csv(config.PAD_20_RAW_METADATA)

    print("- Splitting the dataset")
    df = split_k_folder_csv(df, PAD20.TARGET_COLUMN, save_path=None, k_folder=5, seed_number=34)

    print("- Converting the labels to numbers")
    df = label_categorical_to_number (df, PAD20.TARGET_COLUMN, col_target_number=PAD20.TARGET_NUMBER_COLUMN)
    
    if args.missing:
        simulate_missing_data(df, save_folder=config.PAD_20_SENTENCE_ENCODED.parent,
                                encoder_function=generate_sentence,
                                encoder_name= 'sentence',
                                features=PAD20.RAW_CATEGORICAL_FEATURES + PAD20.NUMERICAL_FEATURES,
                                dataset='pad-20')
    else:
        print("- Generating setences")
        df = generate_sentence(df)

        print("- Checking the target distribution")
        print(df[PAD20.TARGET_COLUMN].value_counts())
        print(f"Total number of samples: {df[PAD20.TARGET_COLUMN].value_counts().sum()}")

        df.to_csv(config.PAD_20_SENTENCE_ENCODED)