import argparse
from pathlib import Path
import yaml

class Config:
    pass


global_config = Config()

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

        setattr(global_config, argname, default_value)
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
        128,
        help='Size of the latent dimension.',
        type=int,
        parser=parser)

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
        4,
        help='How many CNN layers the model should have.',
        type=int,
        parser=parser)

    add_config_argument(
        'nvaes',
        2,
        help='How many VAEs the module should include.',
        type=int,
        parser=parser)

    add_config_argument(
        'epoch_length',
        500,
        help='How many parameter updates an epoch should contain.',
        type=int,
        parser=parser)

    add_config_argument(
        'stage_length',
        20,
        help='How many epochs a single stage should last for.',
        type=int,
        parser=parser)

    add_config_argument(
        'nstages',
        100,
        help='How many stages to execute in total.',
        type=int,
        parser=parser)

    add_config_argument(
        'clevr',
        None,
        help='Use the clevr dataset, and find it at this path.',
        type=str,
        parser=parser)

    sample_config.close()



def update_config_from_parsed_args(args):
    for field_name in _config_argparse_fields:
        val = getattr(args, field_name)
        if val:
            setattr(global_config, field_name, val)


def update_config_from_yaml(cfg: Path):
    doc = yaml.load(cfg.read_text())
    for (k, v) in doc.items():
        assert hasattr(global_config, k)
        setattr(global_config, k, v)


def dump_config_to_yaml(cfg: Path):
    with cfg.open('w+t') as f:
        for field_name in _config_argparse_fields:
            val = getattr(global_config, field_name)
            if not val:
                continue
            f.write(f'{field_name}: {val}\n')


# These options probably don't need to be set often.
global_config.num_examples = 16
global_config.batch_size = 32
global_config.checkpoint_dir = Path('checkpoints')

global_config.img_width = 28
global_config.img_height = 28
global_config.img_channels = 1
