#!/bin/bash

shopt -s globstar

runs=1

function print_help() {
    echo "Usage: $0 [args]"
    echo "-r|--run runs: specify how many times to rerun the model. This is useful for consistency checking."
}

function parse_args() {
    while [ $# -gt 0 ]; do
        key="$1"
        shift
        case "$key" in
            -h|--help)
                print_help
                exit 0
                ;;
            -r|--runs)
                runs="$1"
                shift
                ;;
            *)
                echo "Unrecognized option: ${key}"
                print_help
                exit -1
        esac
    done
}

function get_experiment_data() {
    read -p "Enter experiment name: " expname
    read -p "Enter experiment description: " desc
}

function load_files_onto_remote() {
    rsync -rav -R **/*py *.sh ericst@login.leonhard.ethz.ch:~/${remote_dir}
}


function run_experiment() {
    load_files_onto_remote

    ssh ericst@login.leonhard.ethz.ch <<EOF
cd ${remote_dir}
echo "${expname}" > experiment_name.txt
echo "${desc}$" > experiment_desc.txt
bsub -W 8:00 -n 4 -R "rusage[mem=4000,ngpus_excl_p=1]" "python3 train.py --name leonhard"
EOF
}

parse_args $@

get_experiment_data

for r in $(seq 1 ${runs}); do
    echo "Launching run nr. ${r}"
    remote_dir="experiments/${expname}_run-${r}"
    run_experiment
done
