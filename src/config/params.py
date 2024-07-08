WANDB_PROJECT = 'example_project'
ENTITY = 'nyrenw'


"""
 Some file paths for different systems used in the development of the project.
 WORKSPACE_PATH: Path to the workspace directory where the data is stored while traning a model.
 PERISTENT_STORAGE_PATH: Path to the directory where the models are stored after training.
"""
##GTX 1650
WORKSPACE_PATH = '/home/nyrenw/AttentiveFP-UV/data/workspace'
PERSISTENT_STORAGE_PATH = '/home/nyrenw/AttentiveFP-UV'

## RTX 4090
#WORKSPACE_PATH = '/home/nyrenw/Documents/workspace'
#PERSISTENT_STORAGE_PATH = '/home/nyrenw/Documents/AttentiveFP'

## RTX 6000
#WORKSPACE_PATH = '/home/nyrenw/Documents/workspace/AttentiveFP'
#PERSISTENT_STORAGE_PATH = '/home/nyrenw/Documents/AttentiveFP'

## Server. Storage in RAM
#WORKSPACE_PATH = ['/dev/shm/workspace', '/dev/shm/workspace', '/sys/fs/cgroup/workspace', '/sys/fs/cgroup/workspace']
#PERSISTENT_STORAGE_PATH = '/data/William/AttentiveFP'

## RTX 3090
#WORKSPACE_PATH = '/media/nyrenw/workspace/data'
#PERSISTENT_STORAGE_PATH = '/media/nyrenw/AttentiveFP'