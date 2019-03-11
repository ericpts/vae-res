# VAE Project

Test out if it is possible to make VAE's each learn to recognize a different
object.

For example, let us consider the MNIST dataset restricted only to the digits 0
and 1.
We would like to have 2 VAE's, and have each one of them learn to represent the
digit 0, and the other one learn to represent the digit 1.


## Project structure

`train.py` -- run this in order to train the model.
`vae.py` -- this is the base variational autoencoder model/
`supervae.py` -- this is the model which contains multiple vae's inside, each of
which is supposed to learn a different concept.


## Normal running

1) First, copy the files to the machine where training needs to happen.
`rsync -rav -R -e "sshpass -p 'root' ssh -p$nport" **/*py preq.sh root@0.tcp.ngrok.io:~/ `

2) Run the `preq.sh` script on the target machin\e.

3) Start `train.py` in a tmux window.

4) Run `sync.sh` to continuously grab partial results.
