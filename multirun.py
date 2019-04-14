#!/usr/bin/env python3
import yaml
from pathlib import Path
import subprocess

gammas = [0.002, 0.005, 0.007, 0.01, 0.02, 0.04]
betas = [0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
nvaes = [4]

to_run = [{
        'beta': beta,
        'gamma': gamma,
        'nvaes': nvae,
    }
        for beta in betas
        for gamma in gammas
        for nvae in nvaes
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
        ['./run.sh', '--runs', '2',
         '--name', name,
         '--desc', 'test',
        ],
        check=True
    )


for cfg in to_run:
    run_once(cfg)