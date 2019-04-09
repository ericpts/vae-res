import argparse
from pathlib import Path
import yaml

class Config:
    pass


config = Config()

_config_argparse_fields = []

def setup_arg_parser(parser: argparse.ArgumentParser):
    sample_config = open('cfg_sample.yaml', 'w+t')
    sample_config.write('## Uncomment options to override command line values.\n\n\n')

    def add_config_argument(
            argname: str,
            default_value,
            help: str,
            type,
            parser: argparse.ArgumentParser,
            **kwargs):

        _config_argparse_fields.append(argname)

        setattr(config, argname, default_value)
        parser.add_argument(
            f'--{argname}',
            type=type,
            help=help,
            **kwargs)

        sample_config.write(
            f'\
# {help}\n\
# {argname}: {default_value}\n\n'
        )

    add_config_argument(
        'expand_per_width',
        2,
        help='How many blocks to concatenate per width.',
        type=int,
        parser=parser)

    add_config_argument(
        'expand_per_height',
        1,
        help='How many blocks to concatenate per height.',
        type=int,
        parser=parser)

    add_config_argument(
        'latent_dim',
        32,
        help='Size of the latent dimension.',
        type=int,
        parser=parser)

    add_config_argument(
        'epochs',
        None,
        help='How many epochs to train VAE_i for.',
        type=int,
        parser=parser,
        nargs='+')

    add_config_argument(
        'beta',
        2.0,
        help='KL loss weight.',
        type=float,
        parser=parser)

    add_config_argument(
        'gamma',
        0.05,
        help='Entropy loss weight.',
        type=float,
        parser=parser)

    add_config_argument(
        'nlayers',
        2,
        help='How many CNN layers the model should have.',
        type=int,
        parser=parser)

    add_config_argument(
        'nvaes',
        2,
        help='How many VAEs the module should include.',
        type=int,
        parser=parser)

    sample_config.close()



def update_config_from_parsed_args(args):
    for field_name in _config_argparse_fields:
        val = getattr(args, field_name)
        if val:
            setattr(config, field_name, val)


def update_config_from_yaml(cfg: Path):
    doc = yaml.load(cfg.read_text())
    for (k, v) in doc.items():
        assert hasattr(config, k)
        setattr(config, k, v)


# These options probably don't need to be set often.
config.num_examples = 16

config.batch_size = 64

config.checkpoint_dir = Path('checkpoints')
