# src/utils/sweep_utils.py
import os
from decimal import Decimal
import torch
from torch_geometric.loader import DataLoader
from src.utils.data_utils import DatasetAttentiveFP, GenSplit
from src.models.AttentiveFP_v2 import AttentiveFP
import time
import logging


# Configure logging to include line number
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')



# Function to get memory usage of a GPU
def get_memory_usage(device, gpu_id):
    if device == 'cpu':
        logging.info('CPU memory usage not supported')
        raise NotImplementedError
    else:
        # Use nvidia-smi to get the memory usage information
        result = os.popen(f'nvidia-smi -i {gpu_id} | grep MiB').readlines()
        result = result[0].split()
        mem_usage = result[8]
        max_mem = result[10]
        mem_usage = mem_usage[:-3]
        max_mem = max_mem[:-3]
        mem_usage = float(Decimal(mem_usage)) * 1.048576 / 1000
        max_mem = float(Decimal(max_mem)) * 1.048576 / 1000
        return mem_usage, max_mem

""" A function to find the maximum batch size that fits in GPU memory by
    iteratively increasing the batch size and checking memory usage.
    Args:
        _model: The model to use for memory usage checks
        device: The device to use ('cpu' or 'cuda')
        gpu_id: The GPU ID to use
        on_disk_data: The dataset to use for memory usage checks
    Returns:
        batch_size: The maximum batch size that fits in GPU memory
"""
# Function to find the maximum batch size that fits in GPU memory
def find_batch_size(_model, device, gpu_id, on_disk_data):
    # Get initial memory usage and capacity
    mem_usage, mem_capacity = get_memory_usage(device, gpu_id)
    logging.info(f'Memory capacity: {mem_capacity}')
    
    batch_size = 64  # Start with an initial batch size
    num_trys = 0  # Counter to limit the number of tries
    
    while True:
        num_trys += 1
        mem_usage_max = 0  # Track maximum memory usage during the loop
        model = _model
        model.to(device)
        model.train()
        
        # Create DataLoader with the current batch size
        loader = DataLoader(on_disk_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
        
        iter = 0
        for data in loader:
            iter += 1
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            time.sleep(0.25)  # Allow time for memory usage to stabilize
            mem_usage, _ = get_memory_usage(device, gpu_id)
            
            if iter == 1:  # Only measure memory usage for the first iteration
                break

        mem_usage_max = mem_usage  # Update maximum memory usage
        
        # Remove the model from GPU memory
        data = data.detach()
        out = out.detach()
        model = model.cpu()

        # Check if the memory usage exceeds the capacity
        if mem_usage_max * 1.15 >= mem_capacity:
            break
        else:
            # Increase the batch size exponentially if there's a lot of free memory
            while mem_usage_max * 2.05 < mem_capacity:
                mem_usage_max = mem_usage_max * 2
                batch_size = batch_size * 2
            # Increase the batch size by 30% if there's moderate free memory
            while mem_usage_max * 1.35 < mem_capacity:
                mem_usage_max = mem_usage_max * 1.3
                batch_size = batch_size * 1.3
                batch_size = int(batch_size)
            # Increase the batch size by 10% if there's a little free memory
            while mem_usage_max * 1.15 < mem_capacity:
                mem_usage_max = mem_usage_max * 1.1
                batch_size = batch_size * 1.1
                batch_size = int(batch_size)
            if num_trys > 2:  # Limit the number of tries to prevent infinite loops
                break

        # Cleanup to free memory
        out = out.detach()
        data = data.detach()
        model = model.cpu()
        
        del out
        del data
        del model
        del loader
        time.sleep(0.25)
        torch.cuda.empty_cache()
        
    logging.info(f'Max batch size: {batch_size}', f'Max memory usage: {mem_usage_max}', f'Memory usage (current): {mem_usage}')

    return batch_size
