#!/bin/bash

apt install tmux vim zsh
apt-get install cuda
pip3 install tf-nightly-gpu-2.0-preview

sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

