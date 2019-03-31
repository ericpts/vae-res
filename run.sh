#!/bin/bash

shopt -s globstar

dt=$(date '+%d-%m-%Y-%H-%M-%S')

read -p "Enter experiment name: " expname
read -p "Enter experiment description: " desc

remote_dir="vae-res-${expname}-${dt}"

rsync -rav -R **/*py preq.sh ericst@login.leonhard.ethz.ch:~/${remote_dir}

ssh ericst@login.leonhard.ethz.ch <<EOF
cd ${remote_dir}
echo "${expname}" > experiment_name.txt
echo "${desc}$" > experiment_desc.txt
bsub -W 8:00 -n 4 -R "rusage[mem=4000,ngpus_excl_p=1]" "python3 train.py --name leonhard"
EOF

