#!/bin/bash


set -e
shopt -s globstar

runs=1
expname=""
desc=""

function print_help() {
    echo "Usage: $0 [args]"
    echo "-r|--run runs: specify how many times to rerun the model. This is useful for consistency checking."
    echo "-n|--name expname: specify the name of the experiment."
    echo "-d|--desc desc: specify a description for the experiment"
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
            -n|--name)
                expname="$1"
                shift
                ;;
            -d|--desc)
                desc="$1"
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
    if [ "$expname" = "" ]; then
        read -p "Enter experiment name: " expname
    fi

    if [ "$desc" = "" ]; then
        read -p "Enter experiment description: " desc
    fi
}

function load_files_onto_remote() {
    rsync -rav -R **/*py *.sh ericst@login.leonhard.ethz.ch:~/${remote_dir}
}


function run_experiment() {
    ssh ericst@login.leonhard.ethz.ch <<EOF
mkdir -p ${remote_dir}
EOF

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
    remote_dir="experiments/${expname}/run-${r}"
    run_experiment
done
