#!/bin/bash

while true; do
    rsync -rav -e "sshpass -p 'root' ssh -p$nport" root@0.tcp.ngrok.io:~/checkpoints/ checkpoints
    rsync -rav -e "sshpass -p 'root' ssh -p$nport" root@0.tcp.ngrok.io:~/images/ images/
    sleep 60
done
