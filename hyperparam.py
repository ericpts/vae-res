#!/usr/bin/env python3
import subprocess


# TODO(): Why are these bad? Is it because of the training, or because of an inherent balance?
bad_combos = [
    (1, 0.25),
    (2, 2),
    (1, 2),
    (2, 1),
    (2, 0.5),
    (1, 1),
    (0.125, 1),
    (2, 0.25),
    (0.5, 2),
    (0.5, 0.125),
]

betas = [1/8, 1/4, 1/2, 1, 2]
gammas = [1/8, 1/4, 1/2, 1, 2]

for beta in betas:
    for gamma in gammas:
        print('Training for beta={}, gamma = {}'.format(beta, gamma))
        args = [
            'python3', 'train.py',
            '--beta', str(beta),
            '--gamma', str(gamma),
            '--name', 'SuperVAE-beta-{}-gamma-{}'.format(beta, gamma)
        ]
        p = subprocess.run(args, check=True)
