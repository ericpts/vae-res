#!/usr/bin/env python3
import subprocess

betas = [2**i for i in range(10 + 1)]

for beta in betas:
    args = ['python3', 'train.py', '--beta', str(beta), '--name', 'SuperVAE-beta={}'.format(beta)]
    p = subprocess.run(args, check=True)
