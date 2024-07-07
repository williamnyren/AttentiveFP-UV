import os
import shutil
import yaml
import pandas as pd
import logging
import argparse
# src/processing/preprocess.py
from src.utils.data_utils import GenSplit, DatasetAttentiveFP, createDataframe

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')

def load_config(config_path):
    with open(config_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def check_raw_data_exists(raw_data_path):
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"The raw data directory {raw_data_path} does not exist.")
    logging.info(f"Raw data directory exists at {raw_data_path}")

def create_dataframe_if_needed(data_path, raw_data, config_path):
    root_df = f'./{data_path}/data/raw/data/dataframe.pkl'
    if not os.path.exists(root_df):
        logging.info("Creating dataframe from raw data...")
        num_molecules = createDataframe(root=root_df, raw_data=raw_data, config=config_path)
        if num_molecules is None:
            df = pd.read_pickle(root_df)
            num_molecules = len(df)
            del df
        logging.info(f"Dataframe created successfully with {num_molecules} molecules.")
    else:
        logging.info("Dataframe already exists. Loading existing dataframe...")
        df = pd.read_pickle(root_df)
        num_molecules = len(df)
        logging.info(f"Loaded existing dataframe with {num_molecules} molecules.")
    return num_molecules, root_df

def copy_dataframe_to_split_dirs(data_path, root_df, split_sets):
    for split_set in split_sets:
        dest_path = f'./{data_path}/{split_set}/data/raw/data/dataframe.pkl'
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if not os.path.exists(dest_path):
            shutil.copy(root_df, dest_path)
            logging.info(f"Copied dataframe to {split_set} directory at {dest_path}.")
        else:
            logging.info(f"Dataframe already exists in {split_set} directory at {dest_path}.")

def process_split_set(data_path, num_molecules, config, force_recreate=False):
    split_dict_path_train = f'{data_path}/train/data/raw/data/split_dict.pt'
    split_dict_path_val = f'{data_path}/val/data/raw/data/split_dict.pt'
    split_dict_path_test = f'{data_path}/test/data/raw/data/split_dict.pt'
    
    if not (os.path.exists(split_dict_path_train) and os.path.exists(split_dict_path_val) and os.path.exists(split_dict_path_test)) or force_recreate:
        split = [config['split_train'], config['split_val'], config['split_test']]
        split_dict_path = f'{data_path}'
        logging.info(f"Generating splits with {num_molecules} molecules in total...")
        GenSplit(root=split_dict_path, num_molecules=num_molecules, split=split, force_recreate=force_recreate)
        logging.info(f"Splits generated successfully and saved to {split_dict_path}.")
    else:
        logging.info(f"Splits already exists at {split_dict_path_train}, {split_dict_path_val}, {split_dict_path_test}.")

def create_datasets(split_sets, data_path, config_spectra):
    for split_set in split_sets:
        root = f'{data_path}/{split_set}/data/'
        os.makedirs(root, exist_ok=True)
        logging.info(f"Executing DatasetAttentiveFP() for {split_set}...")
        DatasetAttentiveFP(root=root, split=split_set, one_hot=True, config=config_spectra, add_fake_edges=True)
        logging.info(f"DatasetAttentiveFP executed for {split_set}. SQLite database stored at {root}.")

def main():
    parser = argparse.ArgumentParser(description='Process some paths for data preparation.')
    parser.add_argument('--data_path', type=str, default='data',
                        help='Path to the data directory (default: data)')
    parser.add_argument('--config_path', type=str, default='src/config/config_spectra.yml',
                        help='Path to the configuration file (default: src/config/config_spectra.yml)')
    parser.add_argument('--raw_data_path', type=str, default='ORNL_data',
                        help='Path to the raw data directory (default: ORNL_data)')
    parser.add_argument('--force_recreate', action='store_true', default=False, 
                        help='Force recreate split_dict.pt')

    args = parser.parse_args()

    split_sets = ['train', 'val', 'test']
    data_path = args.data_path
    config_path = args.config_path
    raw_data_path = args.raw_data_path
    force_recreate = args.force_recreate

    config_spectra = load_config(config_path)
    logging.info(f"Configuration loaded from {config_path}.")
    
    check_raw_data_exists(raw_data_path)

    num_molecules, root_df = create_dataframe_if_needed(data_path, raw_data_path, config_path)
    copy_dataframe_to_split_dirs(data_path, root_df, split_sets)

    process_split_set(data_path, num_molecules, config_spectra, force_recreate=force_recreate)

    create_datasets(split_sets, data_path, config_spectra)

    logging.info("All datasets created successfully.")

if __name__ == "__main__":
    main()
