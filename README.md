# SuperVAE repository

## Purpose
Test out if it is possible to make VAE's each learn to recognize a different
object.

For example, let us consider the MNIST dataset restricted only to the digits 0
and 1.
We would like to have 2 VAE's, and have each one of them learn to represent the
digit 0, and the other one learn to represent the digit 1.


## Project structure
`sample.py` -- once a model has been trained, use this to sample images.

`train.py` -- run this in order to train the model.

`vae.py` -- this is the base variational autoencoder model.

`supervae.py` -- this is the model which contains multiple vae's inside, each of
   which is supposed to learn a different concept.



How to run:

All options can either be given on the command line, or they can be provided via a config file.
In order to see a list of the possible options, take a look at `cfg_sample.yaml`:

```yaml
# How many blocks to concatenate per width.
# expand_per_width: 2

# How many blocks to concatenate per height.
# expand_per_height: 1

# Size of the latent dimension.
# latent_dim: 128

# KL loss weight.
# beta: 2.0

# Entropy loss weight.
# gamma: 0.05

# How many CNN layers the model should have.
# nlayers: 4

# How many VAEs the module should include.
# nvaes: 2

# How many parameter updates an epoch should contain.
# epoch_length: 500

# How many epochs a single stage should last for.
# stage_length: 20

# How many stages to execute in total.
# nstages: 100

# Use the clevr dataset, and find it at this path.
# clevr: None
```

Once the model has started training, it will save all configuration parameters to `cfg_all.yaml`.
 
Training example:

```bash
python3 train.py --name leonhard --config cfg.yaml --epochs 10 --clevr ../clevr/ --nvaes 1
```

Currently, the model is adapted to work only for the `clevr` dataset.