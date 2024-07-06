from setuptools import setup, find_packages

setup(
    name='AttentiveFP-UV',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torch_geometric',
        'numpy',
        'pandas',
        'tqdm',
        'wandb',
        'matplotlib',
        'plotnine',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'train_ddp = train_ddp:main',
        ],
    },
)
