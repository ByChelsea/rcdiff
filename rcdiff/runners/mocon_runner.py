from easyvolcap.engine import cfg, call_from_cfg
from easyvolcap.engine import RUNNERS

from rcdiff.runners.lowlevel_mopred_runner import LowLevelMoPredRunner
from rcdiff.utils.net_utils import load_other_network
from rcdiff.utils.engine_utils import parse_args_list


@RUNNERS.register_module()
class MoConRunner(LowLevelMoPredRunner):
    def __init__(self,
                 **kwargs
                 ):
        
        call_from_cfg(super().__init__, kwargs)
        
        motoken_cfg = parse_args_list(['-c', cfg.motoken_cfg_file])
        cfg.motoken_net = load_other_network(motoken_cfg)

        if hasattr(cfg, 'transl_cfg_file'):
            transl_cfg = parse_args_list(['-c', cfg.transl_cfg_file])
            cfg.transl_net = load_other_network(transl_cfg)

        if hasattr(cfg, 'contact_cfg_file'):
            contact_cfg = parse_args_list(['-c', cfg.contact_cfg_file])
            cfg.contact_net = load_other_network(contact_cfg)
