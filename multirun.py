#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
import subprocess

gammas = [0.009, 0.011, 0.013, 0.015]
betas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
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
    parser.add_argument('--runs', type=int, required=True, help='How many times to repeat each experiment.')

    args = parser.parse_args()
    for cfg in to_run:
        run_once(cfg, args.runs)


if __name__ == '__main__':
    main()
