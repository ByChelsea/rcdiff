from easyvolcap.engine import get_parser, parse_cfg
from easyvolcap.engine.registry import Registry
ENCODERS = Registry('encoders')
DECODERS = Registry('decoders')


def discover_modules():
    from easyvolcap import engine

    import rcdiff.dataloaders
    import rcdiff.models
    import rcdiff.runners

    from easyvolcap.runners import optimizers


def parse_args_list(args_list):
    parser = get_parser()
    args, argv = parser.parse_known_args(args_list)  # commandline arguments
    argv = [v.strip('-') for v in argv]  # arguments starting with -- will not automatically go to the ops dict, need to parse them again
    argv = parser.parse_args(argv)  # the reason for -- arguments is that almost all shell completion requires a prefix for optional arguments
    args.opts.update(argv.opts)
    cfg = parse_cfg(args)
    return cfg
