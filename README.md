# AttentiveFP-UV

## Table of Contents
- [Project Overview](#project-overview)
  - [Background](#background)
  - [Goals and Objectives](#goals-and-objectives)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Retrieving the Dataset](#retrieving-the-dataset)
- [Training the Model](#training-the-model)
  - [Using Command Line Arguments](#using-command-line-arguments)
  - [Using a WandB Config File](#using-a-wandb-config-file)


## Project Overview

Built around the model AttentiveFP provided by PyG ([PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)), this project aims to train versions of the model to predict the UV-absorption spectra in organic molecules. The dataset is provided by Oak Ridge National Laboratory (ORNL), which includes a wide range of organic molecules and their corresponding UV-Vis absorption spectra.

This project seeks to leverage graph neural networks to understand and predict how organic molecules absorb UV light, which is crucial for various applications in chemistry and materials science. By accurately predicting these spectra, we can facilitate the design of new materials and molecules with desired optical properties.

### Background

The prediction of absorption spectra using deep learning models has become a significant topic in spectroscopy. Traditional methods for predicting UV-Vis spectra often involve complex quantum mechanical calculations, which are computationally expensive and time-consuming. Deep learning models, especially graph neural networks like AttentiveFP, offer a promising alternative by learning directly from data.

AttentiveFP, a model based on attention mechanisms within graph neural networks, has shown promising results in various molecular property prediction tasks. In this project, we adapt AttentiveFP and explore different graph attention methods such as GAT, GATv2, MoGAT(v2), and DenseGAT to enhance the prediction accuracy of UV-Vis spectra.

### Goals and Objectives

The main objectives of this project are:
1. **Data Preparation**: Prepare a dataset of molecules with corresponding UV-Vis spectra.
2. **Model Training**: Train AttentiveFP with different Graph Attention methods, such as GAT, GATv2, MoGAT(v2), and DenseGAT.
3. **Prediction and Evaluation**: Predict UV-Vis spectra for unseen molecules and evaluate the performance of the models.

## Setup

### Prerequisites

- Python 3.11+
- [Anaconda](https://www.anaconda.com/products/individual)

### Installation

Follow these steps to set up your environment and install all necessary dependencies:

1. **Clone the Repository**

   ```bash
      git clone https://github.com/williamnyren/AttentiveFP-UV.git
      cd AttentiveFP-UV
   ```

2. **Create and Activate the Conda Environment**

   This will create the environment and install all conda and pip dependencies specified in the environment.yml file.

   ```bash
      conda env create -f environment.yml
      conda activate attentive_fp_test
   ```


3. **Make the postBuild Script Executable**

   The postBuild script is used to install PyTorch, torchvision, and torchaudio with the specified CUDA version.

   ```bash
      chmod +x postBuild
   ```
4. **Run the postBuild Script**

   Execute the script to complete the installation of the required packages.

   ```bash
      ./postBuild
   ```

   By following these steps, you will set up your development environment with all the necessary dependencies to run the AttentiveFP-UV project.

## Retrieving the Dataset

   To retrieve the dataset required for this project, follow these detailed steps:

### Based on the Work Here
   Refer to the article [Two excited-state datasets for quantum chemical UV-vis spectra of organic molecules](https://www.nature.com/articles/s41597-023-02408-4#Sec10).

### Steps to Download the Dataset

   1. **Go to the ORNL Data Transfer Guide**
      - Follow the instructions on how to install and use Globus from the ORNL Data Transfer Guide: 
         - [Data Transferring Guide](https://docs.olcf.ornl.gov/data/index.html#data-transferring-data)

   2. **Install Globus**
      - Download and install the Globus app from the official Globus site:
         - [Globus Download Site](https://app.globus.org/collections/gcp)

   3. **Connect to Globus**
      - Open the Globus app and sign in using your credentials. Follow the on-screen instructions to connect to the Globus file transfer service.

   4. **Locate the Files on Globus**
      - The files to transfer on Globus can be found via the references in the article:
         - Yoo, P., Lupo Pasini, M., Mehta, K. & Irle, S. Supplementary material for GDB-9-Ex. OSTI.gov [DOI](https://doi.org/10.13139/OLCF/1985521) (2023).
         - Yoo, P., Lupo Pasini, M., Mehta, K. & Irle, S. Supplementary material for ORNL_AISD-Ex. OSTI.gov [DOI](https://doi.org/10.13139/OLCF/1985737) (2023).

   5. **Transfer the Files**
      - Ensure that you are connected to Globus in the installed app.
      - Navigate to the file you want to download from Globus via the references mentioned above.
      - Select a directory on your local system where you want to transfer the files.
      - Follow the instructions in the Globus app to initiate and complete the file transfer.

   By following these steps, you will be able to download the dataset required for this project.


## Training the Model
   Train the model using command line arguments or using a WandB configuration file. You can 
### Using Command Line Arguments
   Override the default parameters directly through command line arguments.
   python like display in README.md github file
   ```
      import argparse

   default_config = {
      'lr': 5e-4,
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
      'Attention_mode': 'GAT',
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

   def parse_args():
      "Overriding default parameters"
      argparser = argparse.ArgumentParser(description='Process hyperparameters')
      argparser.add_argument('--lr', type=float, default=default_config['lr'], help='Learning rate')
      argparser.add_argument('--hidden_channels', type=int, default=default_config['hidden_channels'], help='Hidden channels')
      argparser.add_argument('--num_layers', type=int, default=default_config['num_layers'], help='Number of layers')
      argparser.add_argument('--num_timesteps', type=int, default=default_config['num_timesteps'], help='Number of timesteps')
      argparser.add_argument('--dropout', type=float, default=default_config['dropout'], help='Dropout')
      argparser.add_argument('--seed', type=int, default=default_config['seed'], help='Seed')
      argparser.add_argument('--num_workers', type=int, default=default_config['num_workers'], help='Number of workers')
      argparser.add_argument('--run_id', type=str, default=default_config['run_id'], help='Run ID for resuming training')
      argparser.add_argument('--total_epochs', type=int, default=default_config['total_epochs'], help='Number of epochs to train')
      argparser.add_argument('--batch_size', type=int, default=default_config['batch_size'], help='Batch size')
      argparser.add_argument('--Attention_mode', type=str, default=default_config['Attention_mode'], help='Attention mode')
      argparser.add_argument('--heads', type=int, default=default_config['heads'], help='Number of heads')
      argparser.add_argument('--loss_function', type=str, default=default_config['loss_function'], help='Loss function')
      argparser.add_argument('--metric', type=str, default=default_config['metric'], help='Metric')
      argparser.add_argument('--savitzkey_golay', type=list, default=default_config['savitzkey_golay'], help='Savitzkey Golay filter')
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
   ```

   Run the script with your desired parameters:
   ```
      python train.py --lr 0.001 --hidden_channels 256 --num_layers 6
   ```

### Using a WandB Config File
   You can also use a WandB configuration file to set the parameters. Create a `config.yaml` file with the following content:
   
   ```
      #Program to run
      program: 'train_ddp.py'
      #Sweep search method: random, grid or bayes
      method: 'random'

      # Project this sweep is part of
      project: 'sweep_large'
      entity: 'nyrenw'

      # Metric to optimize
      metric:
      name: 'val_srmse'
      goal: 'minimize'

      # Parameters search space
      parameters:
      lr: 
         values: [0.0001]
      hidden_channels:
         values: [600]
      num_layers:
         values: [8]
      num_timesteps:
         values: [2]
      dropout:
         values: [0.05]
      num_workers:
         value: 3
      total_epochs:
         value: 50
      warmup_epochs:
         value: 2
      batch_size:
         values: [0]
      Attention_mode:
         values: ['GATv2']
      heads: 
         values: [3]
      loss_function:
         values: ['mse_loss']
      with_fake_edges: 
         value: 0
      lr_ddp_scaling:
         value: 0
      batch_ddp_scaling:
         value: 1
      savitzkey_golay:
         values: [0]
      #  seed:
      #    values: [42, 13, 7]
   ```
   