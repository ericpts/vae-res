#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
import subprocess

gammas = [0.005, 0.007, 0.009, 0.011]
betas = [0.9]
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
        f'{key}={value}' for (key, value) in cfg.items()
    ]) + '-new-optimizer-lower-lr'


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
