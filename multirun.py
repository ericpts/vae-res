#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
import subprocess
import git

gammas = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032]
betas = [0.1, 0.2, 0.4, 0.8, 1.6]
nvaes = [3]

to_run = [{
    'beta': beta,
    'gamma': gamma,
    'nvaes': nvae,
} for beta in betas for gamma in gammas for nvae in nvaes]


def get_commit_name():
    repo = git.Repo()
    sha = repo.head.object.hexsha[:6]
    branch = repo.head.ref.name
    return f'{branch}-{sha}'


def run_once(cfg: dict, runs: int):
    p = Path('cfg.yaml')
    p.write_text(yaml.dump(cfg, default_flow_style=False))

    name = '_'.join(
        [f'{key}={value}' for (key, value) in cfg.items() if key != 'nvaes']) + '-' + get_commit_name()

    proc = subprocess.run(['./run.sh', '--runs', str(runs), '--name', name, '--desc', 'test',], check=True)


def main():
    parser = argparse.ArgumentParser(description='Launch multiple experiments')
    parser.add_argument(
        '--runs',
        type=int,
        required=True,
        help='How many times to repeat each experiment.')

    args = parser.parse_args()
    for cfg in to_run:
        run_once(cfg, args.runs)


if __name__ == '__main__':
    main()
