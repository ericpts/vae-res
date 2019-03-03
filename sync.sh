#!/bin/bash

export nport=16911

while true; do
    rsync -rav -e "ssh -p$nport" root@0.tcp.ngrok.io:~/checkpoints/ checkpoints
    rsync -rav -e "ssh -p$nport" root@0.tcp.ngrok.io:~/image_at_epoch_\*.png images/
    sleep 60
done
