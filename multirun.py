#!/usr/bin/env python3
import yaml
from pathlib import Path
import subprocess

to_run = [
    {
        'beta': 2,
        'gamma': 0.01,
        'nvaes': 3,
    },
    {
        'beta': 1,
        'gamma': 0.01,
        'nvaes': 3,
    },
    {
        'beta': 2,
        'gamma': 0.01,
        'nvaes': 4,
    },
    {
        'beta': 1,
        'gamma': 0.01,
        'nvaes': 4,
    },



    {
        'beta': 2,
        'gamma': 0.005,
        'nvaes': 3,
    },
    {
        'beta': 1,
        'gamma': 0.005,
        'nvaes': 3,
    },
    {
        'beta': 2,
        'gamma': 0.005,
        'nvaes': 4,
    },
    {
        'beta': 1,
        'gamma': 0.005,
        'nvaes': 4,
    },
]

def run_once(cfg: dict):
    p = Path('cfg.yaml')
    p.write_text(
        yaml.dump(cfg, default_flow_style=False)
    )

    name = '_'.join([
        f'{key}={value}' for (key, value) in cfg.items()
    ]) + '-coordconv'

    proc = subprocess.run(
        ['./run.sh', '--runs', '4',
         '--name', name,
         '--desc', 'test',
        ],
        check=True
    )


for cfg in to_run:
    run_once(cfg)
