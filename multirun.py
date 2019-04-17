#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
import subprocess

gammas = [0.005, 0.007, 0.01, 0.02]
betas = [0.5, 0.7, 0.9, 1.0]
nvaes = [5]

to_run = [{
        'beta': beta,
        'gamma': gamma,
        'nvaes': nvae,
    }
        for beta in betas
        for gamma in gammas
        for nvae in nvaes
]


def run_once(cfg: dict, runs: int):
    p = Path('cfg.yaml')
    p.write_text(
        yaml.dump(cfg, default_flow_style=False)
    )

    name = '_'.join([
        f'{key}={value}' for (key, value) in cfg.items()
    ]) + '-coordconv'


    proc = subprocess.run(
        ['./run.sh', '--runs', str(runs),
         '--name', name,
         '--desc', 'test',
        ],
        check=True
    )

def main():
    argparse = argparse.ArgumentParser(help='Launch multiple experiments')
    argparse.add_argument('--runs', type=int, required=True, help='How many times to repeat each experiment.')

    args = argparse.parse_args()
    for cfg in to_run:
        run_once(cfg, args.runs)


if __name__ == '__main__':
    main()
