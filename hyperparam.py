#!/usr/bin/env python3
import subprocess

betas = [1, 2, 4, 6, 8]

for beta in betas:
    args = ['python3', 'train.py', '--beta', str(beta), '--name', 'SuperVAE-beta={}'.format(beta)]
    p = subprocess.run(args, check=True)
