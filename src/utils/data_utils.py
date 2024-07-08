import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import Data, OnDiskDataset, download_url, extract_zip
from torch_geometric.data.data import BaseData, Data
# src/utils/data_utils.py
from src.modifications.torch_geometric_modified import from_smiles

import yaml

import shutil
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
import logging

# Configure logging to include line number
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')

class DatasetAttentiveFP(OnDiskDataset):
    r"""
        Creating a dataset from the dataframe. The dataframe should have the following columns:
        smiles: SMILES string of the molecule
        smooth_spectra: Smoothed spectra of the molecule as a np.array
        
        The dataframe is created using the createDataframe function in data_utils.py
    """

    split_mapping = {
        'train': 'train',
        'val': 'valid',
        'test': 'test',
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        backend: str = 'sqlite',
        one_hot = False,
        config = None,
        add_fake_edges = False
    ) -> None:
        assert split in ['train', 'test', 'val']
        self.split = split
        self.one_hot = one_hot
        self.config = config
        self.add_fake_edges = add_fake_edges
        if config is None:
            logging.info('Please provide the configuration file path. Exiting...')
            return
                
        if self.one_hot:
            schema = {
                'x': dict(dtype=torch.float, size=(-1, 26)),
                'edge_index': dict(dtype=torch.long, size=(2, -1)),
                'edge_attr': dict(dtype=torch.float, size=(-1, 14)),
                'smiles': str,
                'y': dict(dtype=torch.float, size=(-1, self.config['out_dim'])),
            }
        else:
            schema = {
                'x': dict(dtype=torch.float, size=(-1, 9)),
                'edge_index': dict(dtype=torch.long, size=(2, -1)),
                'edge_attr': dict(dtype=torch.float, size=(-1, 3)),
                'smiles': str,
                'y': dict(dtype=torch.float, size=(-1, self.config['out_dim'])),
            }

        super().__init__(root, transform, backend=backend, schema=schema)

        split_idx = torch.load(self.raw_paths[1])
        self._indices = split_idx[self.split_mapping[split]].tolist()

    
    
    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join('data', 'dataframe.pkl'),
            osp.join('data', 'split_dict.pt'),
        ]
    def get_indecies(self):
        return self._indices

    def download(self) -> None:
        pass        
    
    def process(self) -> None:
        # Process all at once. Then use self._indices to select data.
        import pandas as pd
        import time
        split_idx = torch.load(self.raw_paths[1])
        indices = split_idx[self.split_mapping[self.split]].tolist()
        df = pd.read_pickle(self.raw_paths[0])
        df = df.iloc[indices]  
        data_list: List[Data] = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            data = from_smiles(row['smiles'], with_hydrogen=True, one_hot=self.one_hot, add_fake_edges=self.add_fake_edges)
            y = torch.tensor(row['smooth_spectra'], dtype=torch.float).view(-1, self.config['out_dim'])
            data.y = y
            data_list.append(data)
            if i + 1 == len(df) or (i + 1) % 1000 == 0:  # Write batch-wise:
                self.extend(data_list)
                data_list = []

    def serialize(self, data: BaseData) -> Dict[str, Any]:
        assert isinstance(data, Data)
        return dict(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            y=data.y,
            smiles=data.smiles,
        )

    def deserialize(self, data: Dict[str, Any]) -> Data:
        return Data.from_dict(data)

class GenSplit():
    r""" Args:
            root (str): Root directory where the dataset should be saved.
            num_molecules (int): Number of molecules in the dataset.
            split (list): List of floats representing the fraction of the dataset
                that should be used for training, validation, and test. The sum
                of the list elements should be 1. (default: [0.8, 0.1, 0.1])
        
    """
    def __init__(self, root = 'split_dict.pt', num_molecules = 10582578 ,split = [0.8, 0.1, 0.1], force_recreate = False):
        root_dirs = [osp.join(root, split_name, 'data/raw/data/split_dict.pt') for split_name in ['train', 'val', 'test']]
        if all([osp.exists(root_dir) for root_dir in root_dirs]) and not force_recreate:
            logging.info('Split already exists. Skipping...')
            return
        else:
            logging.info('Creating splits...')
        self.num_molecules = num_molecules-1
        self.split = split
        assert sum(split) == 1
        self.indices = torch.arange(num_molecules)
        if force_recreate:
            #Shuffle the indices
            self.indices = self.indices[torch.randperm(num_molecules)]
        split_dict = {'train': self.indices[0:int(num_molecules*split[0])],
                      'valid': self.indices[int(num_molecules*split[0]):int(num_molecules*(split[0]+split[1]))],
                      'test': self.indices[int(num_molecules*(split[0]+split[1])):]}
        for root in root_dirs:
            torch.save(split_dict, root)

def search_for_dots(smis):
    df_lst_reject = []
    df_lst_pass = []
    for smi in smis:
        if '.' in smi:
            df_lst_reject.append(smi)
        else:
            df_lst_pass.append(smi)
    return df_lst_reject, df_lst_pass

def smooth_spectra(config, df_split):
    def convert_ev_in_nm(ev_value):
        planck_constant = 4.1357 * 1e-15  # eV s
        light_speed = 299792458  # m / s
        meter_to_nanometer_conversion = 1e+9
        return 1 / ev_value * planck_constant * light_speed * meter_to_nanometer_conversion

    def energy_to_wavelength(ev_list, prob_list):
        nm_list = [convert_ev_in_nm(value) for value in ev_list]
        combined = list(zip(nm_list, prob_list))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        nm_list, prob_list = zip(*sorted_combined)
        return nm_list, prob_list

    def gauss(a, m, x, w):
        return a * np.exp(-(np.log(2) * ((m - x) / w) ** 2))

    def raw_to_smooth(nm_list, prob_list, config, w=10.0):
        spectrum_discretization_step = config['resolution']
        xmin_spectrum = 0
        xmax_spectrum = config['max_wavelength']
        xmax_spectrum_tmp = xmax_spectrum * 2

        gauss_sum = []
        x = np.arange(xmin_spectrum, xmax_spectrum_tmp, spectrum_discretization_step)

        for index, wn in enumerate(nm_list):
            gauss_sum.append(gauss(prob_list[index], x, wn, w))

        gauss_sum_y = np.sum(gauss_sum, axis=0)
        index_min = int(config['min_wavelength'] / spectrum_discretization_step)

        x = x[index_min:int(len(x) / 2)]
        gauss_sum_y = gauss_sum_y[index_min:int(len(gauss_sum_y) / 2)]
        y = np.where(gauss_sum_y < 1e-3, 0, gauss_sum_y)
        return y

    ex_indices = [i for i in range(len(df_split.columns)) if df_split.columns[i][:2] == 'ex']
    prob_indices = [i for i in range(len(df_split.columns)) if df_split.columns[i][:4] == 'prob']

    df_smooth = {'smiles': [], 'smooth_spectra': []}
    for _, row in df_split.iterrows():
        ev_list = row.iloc[ex_indices].values
        prob_list = row.iloc[prob_indices].values
        nm_list, prob_list = energy_to_wavelength(ev_list, prob_list)
        y = raw_to_smooth(nm_list, prob_list, config, config['peak_width'])
        df_smooth['smiles'].append(row['smiles'])
        df_smooth['smooth_spectra'].append(y)

    return pd.DataFrame(df_smooth)

def createDataframe(root: str = '', raw_data: str = '', config: str = '', num_cpus: int = 0, force_recreate: bool = False):
    if os.path.exists(root) and not force_recreate:
        logging.info('Dataframe already exists. Skipping...')
        return

    config = validate_and_load_config(root, raw_data, config)
    num_cpus = determine_num_cpus(num_cpus)

    df_all = filter_and_load_smiles(raw_data, num_cpus, force_recreate)
    
    assert df_all.duplicated(subset='smiles').sum() == 0, "There are still duplicate SMILES in df_all"

    df_all_chunks = split_dataframe(df_all, num_cpus)
    results = process_dataframe_chunks(df_all_chunks, config, num_cpus)

    save_dataframe(results, root)
    return len(results)

def validate_and_load_config(root, raw_data, config_path):
    validate_paths(root, raw_data, config_path)
    return load_config(config_path)

def validate_paths(root, raw_data, config_path):
    if not os.path.exists(raw_data):
        raise FileNotFoundError('Raw data file not found. Exiting...')
    if not os.path.exists(config_path):
        raise FileNotFoundError('Configuration file not found. Exiting...')
    if not root:
        raise ValueError('Please provide the root directory to store the dataframe. Exiting...')
    os.makedirs(os.path.dirname(root), exist_ok=True)

def determine_num_cpus(num_cpus):
    if num_cpus == 0:
        return max(os.cpu_count() - 2, 2)
    return num_cpus

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['resolution'] = (config['max_wavelength'] - config['min_wavelength']) / config['out_dim']
    return config

def filter_and_load_smiles(raw_data, num_cpus, force_recreate):
    filtered_path = os.path.join(raw_data, 'df_filtered.pkl')
    if os.path.exists(filtered_path) and not force_recreate:
        logging.info('SMILES already filtered. Reading dataframe...')
        return pd.read_pickle(filtered_path)

    logging.info('Filtering SMILES...')
    df_all = filter_smiles(raw_data, num_cpus)
    df_all.to_pickle(filtered_path)
    return df_all

def filter_smiles(raw_data, num_cpus):
    raw_data_tmp = os.path.join(raw_data, 'extracted')
    df_small = pd.read_csv(os.path.join(raw_data_tmp, 'gdb9_ex.csv'))
    files = [f for f in os.listdir(raw_data_tmp) if 'ornl_aisd_ex_' in f]
    dfs_large = [pd.read_csv(os.path.join(raw_data_tmp, f)) for f in files]
    df_large = pd.concat(dfs_large)
    df_all = pd.concat([df_small, df_large])

    smiles_all = np.concatenate((df_large['smiles'].values, df_small['smiles'].values))
    smiles_all = np.array_split(smiles_all, num_cpus)

    pool = mp.Pool(num_cpus)
    func = partial(search_for_dots)
    results = pool.map(func, smiles_all)
    pool.close()
    pool.join()

    df_reject = [smi for res in results for smi in res[0]]
    logging.info(f'Number of rejected SMILES: {len(df_reject)}')

    df_reject = pd.DataFrame(df_reject, columns=['smiles'])
    df_all = df_all[~df_all['smiles'].isin(df_reject['smiles'])]
    # Identify and handle duplicates
    df_duplicates = df_all[df_all.duplicated('smiles', keep=False)]
    df_duplicates.to_pickle(os.path.join(raw_data, 'df_duplicates.pkl'))
    df_reject.to_pickle(os.path.join(raw_data, 'df_reject.pkl'))
    
    df_all = df_all.drop_duplicates(subset='smiles', keep='first')
    df_all.to_pickle(os.path.join(raw_data, 'df_filtered.pkl'))
    return df_all

def split_dataframe(df_all, num_cpus):
    length = len(df_all)
    chunks = 200 if length > 1000000 else 20 if length > 100000 else num_cpus
    size = (length // chunks) + (1 if length % chunks != 0 else 0)
    return [df_all[i * size:(i + 1) * size] for i in range(chunks)]

def process_dataframe_chunks(df_all_chunks, config, num_cpus):
    pool = mp.Pool(num_cpus)
    func = partial(smooth_spectra, config)
    results = pool.map(func, df_all_chunks)
    pool.close()
    pool.join()
    return pd.concat(results)

def save_dataframe(df, root):
    df.to_pickle(root)
    logging.info(f'Dataframe created and saved to {root}')
