#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
import subprocess

gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
betas = [0.01, 0.05, 0.1, 0.5, 1]
nvaes = [1]
nlayers = [2, 4]

to_run = [{
        'beta': beta,
        'gamma': gamma,
        'nvaes': nvae,
        'clevr': '/cluster/scratch/ericst/clevr',
        'epochs': 500,
        'nlayers': nl,
    }
        for beta in betas
        for gamma in gammas
        for nvae in nvaes
        for nl in nlayers
]


def run_once(cfg: dict, runs: int):
    p = Path('cfg.yaml')
    p.write_text(
        yaml.dump(cfg, default_flow_style=False)
    )

    name = '_'.join([
        f'{key}={value}' for (key, value) in cfg.items() if key != 'nvaes'
    ])


    proc = subprocess.run(
        ['./run.sh', '--runs', str(runs),
         '--name', name,
         '--desc', 'test',
        ],
        check=True
    )

def main():
    parser = argparse.ArgumentParser(description='Launch multiple experiments')
    parser.add_argument(
        '--runs',
        type=int,
        required=True,
        help='How many times to repeat each experiment.'
    )

    subprocess.run([
        'rsync',
        '-a',
        '../clevr/',
        'ericst@login.leonhard.ethz.ch:/cluster/scratch/ericst/clevr/',
    ], check=True)

    args = parser.parse_args()
    for cfg in to_run:
        run_once(cfg, args.runs)


if __name__ == '__main__':
    main()
