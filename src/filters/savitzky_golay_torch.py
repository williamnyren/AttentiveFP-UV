from scipy.signal import savgol_coeffs
import torch
import torch.nn

# Initialize coefficients using Scipy and prepare them for use in PyTorch
def initialize_savgol_filter(device, window_length=5, polyorder=3, deriv=1, delta=1.0, padding=False):
    assert deriv in [1, 2]
    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
    coeffs = torch.tensor(coeffs, dtype=torch.float32)
    coeffs = coeffs[None, None, :].to(device)
    deriv_tensor = torch.tensor(deriv*(deriv - 2) + deriv*(deriv-1)).to(device)
    dev = device
    if padding:
        def savgol_filter_torch(input_tensor):
            pad = torch.nn.ReplicationPad2d((1, 1, 0, 0)).to(dev)
            input_tensor = input_tensor.view(input_tensor.size(0), 1, -1)
            input_tensor = pad(input_tensor)
            
            input_tensor = torch.nn.functional.conv1d(input_tensor, coeffs, padding='same')
            input_tensor = input_tensor.view(input_tensor.size(0), -1)
            input_tensor = input_tensor * deriv_tensor
            input_tensor = input_tensor[:, 1:-1]
            input_tensor = input_tensor/torch.max(torch.abs(input_tensor), dim=1, keepdim=True)[0]
            return input_tensor
    else:
        def savgol_filter_torch(input_tensor):
            input_tensor = input_tensor.view(input_tensor.size(0), 1, -1)
            input_tensor = torch.nn.functional.conv1d(input_tensor, coeffs, padding='same')
            input_tensor = input_tensor.view(input_tensor.size(0), -1)
            input_tensor = input_tensor * deriv_tensor
            input_tensor = input_tensor/torch.max(torch.abs(input_tensor), dim=1, keepdim=True)[0]
            return input_tensor
    return savgol_filter_torch