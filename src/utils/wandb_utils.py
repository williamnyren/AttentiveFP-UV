import wandb
""" 
    Script to log data to wandb by calling the log_to_wandb function.
    If more data is to be logged, add the respective keys to the dictionary and call the log_to_wandb function.
"""
def log_to_wandb(run_wandb, data, METRIC_FUNCTION, LOSS_FUNCTION, image_log):
    """                    
    data = {'val': validate_average_metric,
            'train': train_average_metric,
            'epoch': epoch,
            'lr': lr,
            'batch_size': self.config['batch_size'],
            'spectra': fig,
            'density_plot': density_fig_name}
    """
    assert METRIC_FUNCTION in ['mse_metric', 'srmse'], 'Invalid METRIC_FUNCTION'
    if image_log:
        if METRIC_FUNCTION == 'mse_metric':
            run_wandb.log({'val_mse': data['val'],
                            'train_mse': data['train'],
                            'epoch': data['epoch'],
                            'lr': data['lr'],
                            'batch_size': data['batch_size'],
                            'Spectra': data['spectra'],
                            'density_plot': wandb.Image(data['density_plot'])})
        elif METRIC_FUNCTION == 'srmse':
            run_wandb.log({'val_srmse': data['val'],
                       'train_srmse': data['train'],
                       'epoch': data['epoch'],
                       'lr': data['lr'],
                       'batch_size': data['batch_size'],
                       'Spectra': data['spectra'],
                       'density_plot': wandb.Image(data['density_plot'])})
        else:
            raise ValueError('Invalid METRIC_FUNCTION')
    else:
        if METRIC_FUNCTION == 'mse_metric':
            run_wandb.log({'val_mse': data['val'],
                            'train_mse': data['train'],
                            'epoch': data['epoch'],
                            'lr': data['lr'],
                            'batch_size': data['batch_size']
                            })
        elif METRIC_FUNCTION == 'srmse':
            run_wandb.log({'val_srmse': data['val'],
                       'train_srmse': data['train'],
                       'epoch': data['epoch'],
                       'lr': data['lr'],
                       'batch_size': data['batch_size']
                       })
        else:
            raise ValueError('Invalid METRIC_FUNCTION')