import os
import sys

# Add the src directory to the sys path to allow absolute imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# PyTorch imports
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler

# Pytorch Geometric imports
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.data import Data, OnDiskDataset, download_url, extract_zip
from torch_geometric.data.data import BaseData, Data
from src.modifications.torch_geometric_modified import from_smiles

# Project specific imports
from src.models.AttentiveFP_v2 import AttentiveFP
from src.utils.data_utils import DatasetAttentiveFP, GenSplit
from src.utils.plot_utils import make_subplot, make_density_plot, make_subplot_v2
from src.loss.spectral_loss import get_spectral_fn, normalize_spectra
from src.filters.savitzky_golay_torch import initialize_savgol_filter
import src.config.params as params
from src.utils.sweep_utils import find_batch_size
from src.utils.wandb_utils import log_to_wandb
from src.utils.scheduler_utils import build_lr_scheduler
# Other imports
import yaml
import wandb
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import random
import logging
import pickle


# Check for config_spectra.yml
config_path = os.path.join(os.path.dirname(__file__), 'src', 'config', 'config_spectra.yml')
if not os.path.exists(config_path):
    raise FileNotFoundError("config_spectra.yml not found in the config directory")
with open(config_path) as file:
    config_spectra = yaml.load(file, Loader=yaml.FullLoader)

# Configure logging to include line number
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')

## Uncomment to enable anomaly detection
#torch.autograd.set_detect_anomaly(True)


WORKSPACE_PATH = params.WORKSPACE_PATH
PERSISTENT_STORAGE_PATH = params.PERSISTENT_STORAGE_PATH


default_config = {
    'lr': 5e-4,
    'init_lr': 5e-8,
    'final_lr': 5e-6,
    'hidden_channels': 250,
    'num_layers': 4,
    'num_timesteps': 2,
    'dropout': 0.025,
    'seed': None,
    'num_workers': 4,
    'total_epochs': 10,
    'warmup_epochs': 1,
    'run_id': None,
    'batch_size': 0,
    'Attention_mode': 'MoGATv2',
    'heads': 2,
    'loss_function': 'mse_loss',
    'metric': 'srmse',
    'savitzkey_golay': [],
    'window_length': 5,
    'polyorder': 3,
    'padding': True,
    'lr_ddp_scaling': 0,
    'batch_ddp_scaling': 1,
    'with_fake_edges': 0,
    'LOSS_FUNCTION': '',
    'METRIC_FUNCTION': 'srmse',
    'NUM_PROCESSES': 1,
    'DATA_DIRECTORY': 'data'
    
}

"""
    Function to parse the arguments passed to the script.
    The default values are stored in the default_config dictionary.
    The default values are overriden if the argument is passed.
    The default values are then used to initialize the model.
"""
def parse_args():
    "Overriding default parameters"
    argparser = argparse.ArgumentParser(description='Process hyperparameters')
    argparser.add_argument('--lr', type=float, default=default_config['lr'], help='Learning rate (Max learning rate)')
    argparser.add_argument('--init_lr', type=float, default=default_config['init_lr'], help='Initial learning rate')
    argparser.add_argument('--final_lr', type=float, default=default_config['final_lr'], help='Final learning rate')
    argparser.add_argument('--hidden_channels', type=int, default=default_config['hidden_channels'], help='Hidden channels')
    argparser.add_argument('--num_layers', type=int, default=default_config['num_layers'], help='Number of layers')
    argparser.add_argument('--num_timesteps', type=int, default=default_config['num_timesteps'], help='Number of timesteps')
    argparser.add_argument('--dropout', type=float, default=default_config['dropout'], help='Dropout')
    argparser.add_argument('--seed', type=int, default=default_config['seed'], help='Seed')
    argparser.add_argument('--num_workers', type=int, default=default_config['num_workers'], help='Number of workers')
    argparser.add_argument('--run_id', type=str, default=default_config['run_id'], help='Run ID for resmuming training')
    argparser.add_argument('--total_epochs', type=int, default=default_config['total_epochs'], help='Number of epochs to train')
    argparser.add_argument('--batch_size', type=int, default=default_config['batch_size'], help='Batch size')
    argparser.add_argument('--Attention_mode', type=str, default=default_config['Attention_mode'], help='Attention mode')
    argparser.add_argument('--heads', type=int, default=default_config['heads'], help='Number of heads')
    argparser.add_argument('--loss_function', type=str, default=default_config['loss_function'], help='Loss function')
    argparser.add_argument('--metric', type=str, default=default_config['metric'], help='Metric')
    argparser.add_argument('--savitzkey_golay', type=list, default=default_config['savitzkey_golay'], help='Order of derivatives for Savitzkey Golay filter')
    argparser.add_argument('--with_fake_edges', type=int, default=default_config['with_fake_edges'], help='Data with fake edges')
    argparser.add_argument('--DATA_DIRECTORY', type=str, default=default_config['DATA_DIRECTORY'], help='Data directory')
    argparser.add_argument('--lr_ddp_scaling', type=int, default=default_config['lr_ddp_scaling'], help='Scale learning rate with number of GPUs')
    argparser.add_argument('--batch_ddp_scaling', type=int, default=default_config['batch_ddp_scaling'], help='Scale batch size with number of GPUs')
    argparser.add_argument('--warmup_epochs', type=int, default=default_config['warmup_epochs'], help='Warmup epochs')
    args = argparser.parse_args()
    for arg in vars(args):
        default_config[arg] = args.__dict__[arg]
    default_config['LOSS_FUNCTION'] = default_config['loss_function']
    default_config['METRIC_FUNCTION'] = default_config['metric']
    if default_config['with_fake_edges'] == 1:
        default_config['DATA_DIRECTORY'] = 'data_add_fake_edges'    
    default_config['out_dim'] = config_spectra['out_dim']
    default_config['split'] = [config_spectra['split_train'], config_spectra['split_val'], config_spectra['split_test']]
    default_config['one_hot'] = config_spectra['one_hot']
    return



def init_random_seed(seed, rank):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    return


def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    #torch.cuda.set_device(rank)

def setup_wandb(config, rank):
    assert rank == 0
    wandb.watch_called = False  # Ensure that wandb.watch isn't called automatically
    if default_config['run_id'] is not None:
        return wandb.init(project=params.WANDB_PROJECT,
                    entity=params.ENTITY,
                    config=config,
                    job_type='traning',
                    id=config['run_id'],
                    resume=True)

    else:
        return wandb.init(project=params.WANDB_PROJECT,
                                entity=params.ENTITY,
                                config=config,
                                job_type='traning')

def load_model(model, total_epochs, map_location=None, checkpoint_file=None, rank=0):
    assert map_location is not None
    checkpoint_path = os.path.join(PERSISTENT_STORAGE_PATH, 'models', checkpoint_file)

    with open(os.path.join(checkpoint_path, f'config_{checkpoint_file}.pkl'), 'rb') as f:
        config = pickle.load(f)
    checkpoint_file = os.path.join(checkpoint_path, f'checkpoint_{checkpoint_file}.pt')
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(map_location)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    total_epochs = total_epochs + epoch
    logging.info(f'Loaded model from checkpoint: {file}')
    if rank == 0:
        #best_val_metric = checkpoint['loss']
        return model, optimizer, epoch, total_epochs, 999999, config#best_val_metric
    else:
        return model, optimizer, epoch, total_epochs, config
    
def maximize_batch_size(rank, result_queue):
    ddp_setup(rank, 1)
    model = AttentiveFP(
        in_channels=26,
        hidden_channels=default_config["hidden_channels"]+10,
        out_channels=config_spectra["out_dim"],
        edge_dim=14,
        num_layers=default_config["num_layers"]+1,
        num_timesteps=default_config["num_timesteps"]+1,
        dropout=0,
        attention_mode = default_config['Attention_mode'],
        heads=default_config['heads']
        )
    gpu_id = rank
    device = torch.device(f'cuda:{gpu_id}')
    path_tmp = os.path.join(PERSISTENT_STORAGE_PATH, default_config['DATA_DIRECTORY'], 'test', 'data')
    data_tmp = DatasetAttentiveFP(root=path_tmp, split='test', one_hot=config_spectra['one_hot'], config=config_spectra)
    batch_size = find_batch_size(model, device, gpu_id, data_tmp)
    result_queue.put(batch_size)
    logging.info(f"Maximized batch size: {batch_size}")
    del model
    del data_tmp
    del device
    
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    
def correct_wandb_list_args():
    # Recives a list of strings like: ['0'] or ['1', ' ', '2']
    # Want it to be a list of integers like: [0] or [1, 2]
    input = default_config['savitzkey_golay']
    # Make it in to a list of integers
    default_config['savitzkey_golay'] = [int(i) for i in input if i != ' ']
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
class SpectralTrainer:
    def __init__(self, model, config, rank, dataset_paths, process_dir):
        self.rank = rank
        self.device = torch.device(rank)
        self.config = config
        self.dataset_paths = dataset_paths
        self.process_dir = process_dir
        self.total_epochs = config['total_epochs']
        self.epoch = 0
        self.warmup_epochs = config['warmup_epochs']
        if rank == 0:
            self.best_val_acc = 999999
            self.run_wandb = setup_wandb(config, rank)
            self.epoch = self.run_wandb.step
            logging.info(f"Training from epoch {self.epoch}")
        dist.barrier()
        if config['run_id'] is not None:
            if rank == 0:
                model, self.optimizer, self.epoch, self.total_epochs, self.best_val_acc, self.config = load_model(model, 
                                                                                                    self.total_epochs,
                                                                                                    self.device,
                                                                                                    checkpoint_file=self.config['run_id'],
                                                                                                    rank=self.rank
                                                                                                    )
                config = self.config
            else:
                model, self.optimizer, self.epoch, self.total_epochs, self.config = load_model(model, 
                                                                                  self.total_epochs,
                                                                                  self.device,
                                                                                  checkpoint_file=self.config['run_id'],
                                                                                  rank=self.rank
                                                                                  )
                config = self.config
            self.model = model.to(rank)
        else:               
            self.model = model.to(rank)
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=4.e-4)
        self.model = DDP(model, device_ids=[rank])
        self.setup_data_loaders()
        self.scheduler = build_lr_scheduler(self.optimizer, self.config, self.num_train_samples)
        self.loss_function = get_spectral_fn(self.config['loss_function'], epoch = self.epoch, warmup_epochs = self.warmup_epochs)
        self.metric = get_spectral_fn(self.config['metric'])
        self.filters = []
        for deriv in self.config['savitzkey_golay']:
            logging.info(f"Initializing Savitzkey Golay filter with deriv={deriv}")
            if deriv == 0:
                continue
            self.filters.append(initialize_savgol_filter(self.device, window_length=self.config['window_length'],
                                                         polyorder=self.config['polyorder'], deriv=deriv,
                                                         padding=self.config['padding']))

        

    def setup_data_loaders(self):
        self.train_data = DatasetAttentiveFP(root=self.dataset_paths['train'], split='train', one_hot=self.config['one_hot'], config=self.config)
        logging.info(f'Number of training samples: {len(self.train_data)}')
        self.num_train_samples = len(self.train_data)
        self.val_data = DatasetAttentiveFP(root=self.dataset_paths['val'], split='val', one_hot=self.config['one_hot'], config=self.config)
        logging.info(f'Number of validation samples: {len(self.val_data)}')
        self.test_data = DatasetAttentiveFP(root=self.dataset_paths['test'], split='test', one_hot=self.config['one_hot'], config=self.config)
        logging.info(f'Number of test samples: {len(self.test_data)}')
        self.train_loader = DataLoader(self.train_data, batch_size=self.config['batch_size'], pin_memory=True, drop_last=True, shuffle = False, sampler=DistributedSampler(self.train_data))
        self.val_loader = DataLoader(self.val_data, batch_size=self.config['batch_size'], pin_memory=True, drop_last=True, shuffle = False, sampler=DistributedSampler(self.val_data))
        self.test_loader = DataLoader(self.test_data, batch_size=self.config['batch_size'], pin_memory=True, drop_last=True, shuffle = False, sampler=DistributedSampler(self.test_data))

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(self, epoch):
        self.model.train()
        total_metric = 0
        total_examples = 0
        self.train_loader.sampler.set_epoch(epoch)
        with tqdm(total=len(self.train_loader), bar_format='{desc}{percentage:3.0f}%|{bar:10}{r_bar}', desc=f'Traning GPU:{self.rank}', leave=False) as bar:
            for data in self.train_loader:
                data = data.to(self.rank)
                self.optimizer.zero_grad()
                output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = self.loss_function(output, data.y, torch_device=self.device)
                for filter in self.filters:
                    filtered_output = filter(output)
                    filtered_y = filter(data.y)
                    loss += 0.5*F.mse_loss(filtered_y, filtered_output)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.config['LOSS_FUNCTION'] == 'mse_loss' or self.config['LOSS_FUNCTION'] == 'kl_div':
                    output = torch.pow(output, 2)

                metric_score = self.metric(output, data.y, torch_device=self.device)
                metric_score = torch.mean(metric_score).item()
                total_metric += metric_score * data.num_graphs
                total_examples += data.num_graphs
                bar.update(1)
                bar.set_postfix(metric=metric_score, loss=loss.item(), lr=self.get_lr(), epoch=epoch)

        average_metric = total_metric / total_examples
        return average_metric

    def validate(self, epoch):
        assert self.rank == 0
        self.model.eval()
        total_metric = 0
        total_examples = 0
        self.val_loader.sampler.set_epoch(epoch)
        with torch.no_grad(), tqdm(total=len(self.val_loader), bar_format='{desc}{percentage:3.0f}%|{bar:10}{r_bar}', desc='Validation', leave=False) as bar:
            for data in self.val_loader:
                data = data.to(self.rank)
                output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                if self.config['LOSS_FUNCTION'] == 'mse_loss' or self.config['LOSS_FUNCTION'] == 'kl_div':
                    output = torch.pow(output, 2)
                    
                metric_score = self.metric(output, data.y, torch_device=self.device)
                metric_score = torch.mean(metric_score).item()
                total_metric += metric_score * data.num_graphs
                total_examples += data.num_graphs
                
                bar.update(1)
                bar.set_postfix(metric=metric_score, lr=self.get_lr(), epoch=epoch)
        average_metric = total_metric / total_examples
        return average_metric
    
    def test(self, epoch):
        assert self.rank == 0
        self.model.eval()
        predictions = {'smiles': [], 'y': [], 'y_pred': [], 'loss': []}
        self.test_loader.sampler.set_epoch(epoch)
        with torch.no_grad(), tqdm(total=len(self.test_loader), bar_format='{desc}{percentage:3.0f}%|{bar:10}{r_bar}', desc='Test', leave=False) as bar:
            for data in self.test_loader:
                data = data.to(self.device)
                output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                target_spectra = data.y
                if self.config['LOSS_FUNCTION'] == 'mse_loss' or self.config['LOSS_FUNCTION'] == 'kl_div':
                    output = torch.pow(output, 2)
                    target_spectra = target_spectra/torch.max(target_spectra, dim=1, keepdim=True)[0]
                metric_score = self.metric(output, data.y, torch_device=self.device)
                if len(list(metric_score.shape)) != 1:
                    metric_score = torch.mean(metric_score, axis=1)
                if self.config['LOSS_FUNCTION'] != 'mse_loss' or self.config['LOSS_FUNCTION'] != 'kl_div':
                    output, target_spectra, _ = normalize_spectra(output, target_spectra, torch_device=self.device)
                metric_score = metric_score.cpu().detach().numpy()
                target_spectra = target_spectra.cpu().detach().numpy()
                output = output.cpu().detach().numpy()
                smiles = data.smiles
                # Group loss, target and output to iterate over associated rows
                for i in range(target_spectra.shape[0]):
                    predictions['smiles'].append(smiles[i])
                    predictions['y'].append(target_spectra[i])
                    predictions['y_pred'].append(output[i])
                    predictions['loss'].append(metric_score[i])
                bar.update()
        predictions = pd.DataFrame(predictions)
        # Example of generating plots or further analysis
        fig = make_subplot_v2(predictions, config_spectra)
        density_fig_name = make_density_plot(predictions, config_spectra, self.process_dir)
        density_fig_name = os.path.join(self.process_dir, 'plots', density_fig_name)
        return fig, density_fig_name

    def run_training(self, total_epochs):
        logging.info(f"Ready for training on GPU:{self.rank}")
        dist.barrier()
        # Flag tensor for early stopping of all processes
        break_flag = torch.zeros(1).to(self.device)
        lr_reduce_flag = torch.zeros(1).to(self.device)
        if self.rank == 0:
            patience = 6
        for epoch in range(self.epoch, self.total_epochs):
            if epoch > 0:
                self.loss_function = get_spectral_fn(self.config['loss_function'], epoch = epoch)
            _train_average_metric = torch.zeros(1).to(self.device)                    
            _train_average_metric += self.train(epoch) 
            dist.all_reduce(_train_average_metric, op=dist.ReduceOp.SUM)
            if self.rank == 0:
                image_log = False
                train_average_metric = _train_average_metric / self.config['NUM_PROCESSES']

                validate_average_metric = self.validate(epoch)
                lr = get_lr(self.optimizer)
                if validate_average_metric < self.best_val_acc:
                    patience = 6
                    self.best_val_acc = validate_average_metric
                    self.save_checkpoint(epoch, self.config)
                    
                    fig, density_fig_name = self.test(epoch)
                    data = {'val': validate_average_metric,
                            'train': train_average_metric,
                            'epoch': epoch,
                            'lr': lr,
                            'batch_size': self.config['batch_size'],
                            'spectra': fig,
                            'density_plot': density_fig_name}
                    image_log = True
                else:
                    data = {'val': validate_average_metric,
                            'train': train_average_metric,
                            'epoch': epoch,
                            'lr': lr,
                            'batch_size': self.config['batch_size']}
                    image_log = False
                    patience -= 1
                    if patience <= 6/2:
                        lr_reduce_flag += 1
                        
                    elif patience == 0:
                        break_flag += 1
                log_to_wandb(self.run_wandb, data, self.config['METRIC_FUNCTION'], self.config['LOSS_FUNCTION'], image_log)           
            dist.all_reduce(break_flag, op=dist.ReduceOp.SUM)
            if break_flag == 1:
                break
            """
            dist.all_reduce(lr_reduce_flag, op=dist.ReduceOp.SUM)
            if lr_reduce_flag == 1:
                lr_reduce_flag *= 0
                if self.scheduler is not None:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / 2
            """
            
            dist.barrier()
        return
    # Change the path to save the checkpoint
    def save_checkpoint(self, epoch, config):
        assert self.rank == 0
        
        checkpoint_file = self.run_wandb.id
        checkpoint_path = os.path.join(PERSISTENT_STORAGE_PATH, 'models', checkpoint_file)
        
        # Make directory if it does not exist
        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.join(checkpoint_path), exist_ok=True)

        with open(os.path.join(checkpoint_path, f'config_{checkpoint_file}.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        checkpoint_file = os.path.join(checkpoint_path, f'checkpoint_{checkpoint_file}.pt')
        ckp_m = self.model.module.state_dict()
        ckp_op = self.optimizer.state_dict()
        torch.save({'epoch': epoch,
                    'model_state_dict': ckp_m,
                    'optimizer_state_dict': ckp_op},
                   checkpoint_file)    
        logging.info(f"Epoch {epoch} | Training checkpoint saved at {checkpoint_file}")
    

def load_config_and_initialize(rank: int, config: dict):
    if config['seed'] is not None:
        init_random_seed(config['seed'], rank)
    if not isinstance(WORKSPACE_PATH, list):
        process_dir = ''
    else:
        workspace_for_rank = WORKSPACE_PATH[rank]
        process_dir = os.path.join(workspace_for_rank, f'process_{rank}')
        os.makedirs(process_dir, exist_ok=True)
    os.makedirs(os.path.join(process_dir, config['DATA_DIRECTORY']), exist_ok=True)
    os.makedirs(os.path.join(process_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(process_dir, 'models'), exist_ok=True)
    
    logging.info(os.path.join(process_dir, config['DATA_DIRECTORY'], 'train', 'data', 'processed', 'sqlite.db'))
    if not os.path.exists(os.path.join(process_dir, config['DATA_DIRECTORY'], 'train', 'data', 'processed', 'sqlite.db')):
        persistent_data = os.path.join(PERSISTENT_STORAGE_PATH, config['DATA_DIRECTORY'])
        logging.info(f"Copying data from {persistent_data} to {process_dir}")
        os.system(f'cp -r {persistent_data} {process_dir}')
        logging.info(f"Transfer complete for process {rank}")

    dataset_paths = {
        'train': os.path.join(process_dir, config['DATA_DIRECTORY'], 'train', 'data'),
        'val': os.path.join(process_dir, config['DATA_DIRECTORY'], 'val', 'data'),
        'test': os.path.join(process_dir, config['DATA_DIRECTORY'], 'test', 'data'),
    }
    dataset_paths['train_split'] = os.path.join(dataset_paths['train'], 'raw', 'data', 'split_dict.pt')
    dataset_paths['val_split'] = os.path.join(dataset_paths['val'], 'raw', 'data', 'split_dict.pt')
    dataset_paths['test_split'] = os.path.join(dataset_paths['test'], 'raw', 'data', 'split_dict.pt')
    logging.info(f'Dataset paths: {dataset_paths} are ready for process {rank}')
    
    
    model = AttentiveFP(
        in_channels=26,
        hidden_channels=config["hidden_channels"],
        out_channels=config["out_dim"], 
        edge_dim=14,
        num_layers=config["num_layers"],
        num_timesteps=config["num_timesteps"],
        dropout=config["dropout"],
        attention_mode=config['Attention_mode'],
        heads=config['heads']
    )
    dist.barrier()
    return config, model, dataset_paths, process_dir

def main(rank, world_size, config):
    try:
        ddp_setup(rank, world_size)
        config, model, dataset_paths, process_dir = load_config_and_initialize(rank, config)
        trainer = SpectralTrainer(model, config, rank, dataset_paths, process_dir)
        trainer.run_training(total_epochs = config['total_epochs'])
        dist.barrier()
        if rank == 0:
            trainer.run_wandb.finish()
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.info(f"Error in process {rank} | {e}")
    finally:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    parse_args()
    logging.info("Maximizing batch size before starting multi-processing for DDP.")
    result_queue = mp.Queue()
    mp.Process(target=maximize_batch_size, args=(0, result_queue)).start()
    max_batch_size = result_queue.get()
    if max_batch_size < default_config['batch_size'] or default_config['batch_size'] == 0:
        default_config['batch_size'] = max_batch_size
    logging.info(f'Batch size set for training: {default_config["batch_size"]}')

    world_size = torch.cuda.device_count()
    default_config['NUM_PROCESSES'] = world_size
    logging.info(f'Number of GPUs and processes to train the model: {world_size}')
    if world_size > 1:
        logging.info(f'Scaling ether batch size or learning rate when using multiple GPUs.')
        assert default_config['lr_ddp_scaling'] != default_config['batch_ddp_scaling']
        if default_config['lr_ddp_scaling']:
            default_config['lr'] = default_config['lr'] * np.sqrt(world_size)
        elif default_config['batch_ddp_scaling'] == 1:
            default_config['batch_size'] = int(default_config['batch_size'] / world_size)
        
    correct_wandb_list_args()
    mp.spawn(main, args=(world_size, default_config), nprocs=world_size)
