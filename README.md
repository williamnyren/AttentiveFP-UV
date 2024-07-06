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

- Python 3.8+
- [Anaconda](https://www.anaconda.com/products/individual)
- PyTorch
- PyTorch Geometric

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/AttentiveFP-UV.git
   cd AttentiveFP-UV
