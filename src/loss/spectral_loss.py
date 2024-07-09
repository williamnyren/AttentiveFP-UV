from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F


def get_spectral_fn(loss_fn: str, epoch = None, warmup_epochs = 0):
    if epoch is not None:
        if epoch <= warmup_epochs:
            return mse_loss
    if loss_fn == 'sid':
        return sid
    elif loss_fn == 'srmse':
        return srmse
    elif loss_fn == 'mse_loss':
        return mse_loss
    elif loss_fn == 'mse_metric':
        return mse_metric
    elif loss_fn == 'kl_div':
        return kl_div_scipy
    else:
        raise ValueError(f'Invalid spectral loss function: {loss_fn}')

def sid(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device = torch.device('cpu')) -> torch.tensor:
    model_spectra, target_spectra, nan_mask = normalize_spectra(model_spectra, target_spectra, threshold, eps, torch_device)
    
    # calculate loss value
    if not isinstance(target_spectra,torch.Tensor):
        target_spectra = torch.tensor(target_spectra)
    target_spectra = target_spectra.to(torch_device)
    loss = torch.ones_like(target_spectra)
    loss = loss.to(torch_device)
    target_spectra[nan_mask]=1
    model_spectra[nan_mask]=1
    loss = torch.mul(torch.log(torch.div(model_spectra,target_spectra)),model_spectra) \
        + torch.mul(torch.log(torch.div(target_spectra,model_spectra)),target_spectra)
    loss[nan_mask]=0
    loss = torch.mean(loss)
    return loss


def srmse(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device = torch.device('cpu')) -> torch.tensor:
    model_spectra, target_spectra, nan_mask = normalize_spectra(model_spectra, target_spectra, threshold, eps, torch_device)

    # calculate loss value
    if not isinstance(target_spectra,torch.Tensor):
        target_spectra = torch.tensor(target_spectra)
    target_spectra = target_spectra.to(torch_device)
    loss = torch.ones_like(target_spectra)
    loss = loss.to(torch_device)
    target_spectra[nan_mask]=1
    model_spectra[nan_mask]=1
    loss = torch.mean((model_spectra-target_spectra)**2,dim=1)
    loss = torch.sqrt(loss + eps)
    return loss




def mse_loss(model_spectra, target_spectra, threshold: float = 1e-8, eps: float = 1e-8, torch_device = torch.device('cpu')):
    target_spectra = target_spectra/(torch.max(target_spectra, dim=1, keepdim=True)[0]+eps)

    target_spectra = torch.sqrt(target_spectra)
    loss = F.mse_loss(target_spectra, model_spectra)
    return loss

def mse_metric(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device = torch.device('cpu'), mode: str = 'train'):
    target_spectra = target_spectra/(torch.max(target_spectra, dim=1, keepdim=True)[0]+eps)
    score = F.mse_loss(target_spectra, model_spectra, reduction='none')    
    return score


def normalize_spectra(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device = torch.device('cpu')) -> torch.tensor:
    # normalize the model spectra before comparison
    nan_mask=torch.isnan(model_spectra)+torch.isnan(target_spectra)
    nan_mask=nan_mask.to(device=torch_device)
    zero_sub=torch.zeros_like(target_spectra,device=torch_device)
    target_spectra = target_spectra.to(torch_device)
    target_spectra[target_spectra < threshold] = threshold
    sum_target_spectra= torch.sum(torch.where(nan_mask,zero_sub,target_spectra),axis=1)

    sum_target_spectra = torch.unsqueeze(sum_target_spectra,axis=1)
    target_spectra = torch.div(target_spectra,sum_target_spectra)
    
    model_spectra = model_spectra.to(torch_device)
    model_spectra[model_spectra < threshold] = threshold
    sum_model_spectra = torch.sum(torch.where(nan_mask,zero_sub,model_spectra),axis=1)

    sum_model_spectra = torch.unsqueeze(sum_model_spectra,axis=1)
    model_spectra = torch.div(model_spectra,sum_model_spectra)

    return model_spectra, target_spectra, nan_mask

def kl_div_scipy(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device = torch.device('cpu')) -> torch.tensor:
    # Normalize the spectra
    target_spectra = target_spectra/(torch.max(target_spectra, dim=1, keepdim=True)[0]+eps)
    target_spectra = torch.sqrt(target_spectra)
    
    model_spectra, target_spectra, nan_mask = normalize_spectra(model_spectra, target_spectra, threshold, eps, torch_device)
    mask_1= torch.gt(model_spectra*target_spectra, 0)
    
    mask_model_spectra_ge_0 = torch.ge(model_spectra, 0)
    mask_traget_eq_0 = torch.eq(target_spectra, 0)
    mask_2 = torch.logical_and(mask_model_spectra_ge_0, mask_traget_eq_0)
    mask_3 = torch.logical_and(torch.logical_not(mask_1), torch.logical_not(mask_2))

    loss = torch.where(mask_1, target_spectra*torch.log(torch.div(target_spectra, model_spectra)) + torch.abs(model_spectra - target_spectra), model_spectra)
    loss = torch.where(mask_3, float('inf'), loss)
    loss = torch.mean(loss)
    return loss