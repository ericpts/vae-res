class Config:
    pass


config = Config()

config.expand_per_width = 2
config.expand_per_height = 1
config.latent_dim = 2
config.num_examples = 64
config.epochs = 40
config.batch_size = 64
config.nlayers = 1
config.nvaes = 2


# How many epochs to train VAE_0 for, and then VAE_1 for.
config.epochs = [20, 60]

# KL-loss coefficient.
config.beta = 1.0

# Entropy loss coefficient.
config.gamma = 10.0
