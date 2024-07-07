# AttentiveFP-UV

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
   Run the postBuild Script

Execute the script to complete the installation of the required packages.

   ```bash
      ./postBuild
   ```

By following these steps, you will set up your development environment with all the necessary dependencies to run the AttentiveFP-UV project.