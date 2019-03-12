class Config:
    pass


config = Config()

config.expand_per_width = 2
config.expand_per_height = 1
config.latent_dim = 10
config.num_examples = 64
config.epochs = 40
config.batch_size = 64
config.nlayers = 2
config.nvaes = 2
config.beta = 10.0
