#!/usr/bin/env python3
import subprocess

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
