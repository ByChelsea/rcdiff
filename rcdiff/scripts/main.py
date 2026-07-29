# fmt: off
from easyvolcap.utils.console_utils import *
import torch
import torch.distributed as dist
import tqdm

from typing import Callable
from torch.nn.parallel import DistributedDataParallel as DDP

from easyvolcap.engine import args, cfg
from easyvolcap.engine import RUNNERS, MODELS, DATALOADERS
from easyvolcap.engine import callable_from_cfg

from rcdiff.utils.engine_utils import discover_modules
discover_modules()


from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.dist_utils import synchronize, get_rank, get_distributed
from easyvolcap.utils.net_utils import setup_deterministic, number_of_params
from easyvolcap.utils.prof_utils import setup_profiler, profiler_start, profiler_stop
# fmt: on

def launch_runner(runner_function: Callable,
                  runner_object = None,
                  exp_name='nerft',
                  detect_anomaly: bool = False,
                  profiling_cfg: dotdict = dotdict(),

                  *args,
                  **kwargs
                  ):

    setup_profiler(**profiling_cfg)
    prev_anomaly = torch.is_anomaly_enabled()
    torch.set_anomaly_enabled(detect_anomaly)
    profiler_start()

    log('Launching experiment:', magenta(exp_name))
    cfg.runner = runner_object
    runner_function()

    profiler_stop()
    torch.set_anomaly_enabled(prev_anomaly)


@callable_from_cfg
def data_test(
    dataloader_cfg: dotdict = dotdict(type="DefaultDataloader"),
    val_dataloader_cfg: dotdict = dotdict(type="DefaultDataloader"),
    test_dataloader_cfg: dotdict = None,
):
    if test_dataloader_cfg is not None:
        test_dataloader = DATALOADERS.build(test_dataloader_cfg)
        for iter, data in enumerate(test_dataloader):
            if iter > 10: break
        return
    dataloader = DATALOADERS.build(dataloader_cfg)
    for iter, data in enumerate(dataloader):
        if iter > 10: break
    val_dataloader = DATALOADERS.build(val_dataloader_cfg)
    for iter, data in enumerate(val_dataloader):
        if iter > 10: break


def preflight(
    fix_random: bool = False,
    allow_tf32: bool = True,
    deterministic: bool = False,
    benchmark: Union[bool, str] = True,
    ignore_breakpoint: bool = False,
    hide_progress: bool = False,
    less_verbose: bool = False,
    hide_output: bool = False,
    **kwargs,
):
    if ignore_breakpoint: disable_breakpoint()
    if hide_progress: disable_progress()
    if hide_output: disable_console()
    if less_verbose: disable_verbose_log()
    if benchmark == 'train': benchmark = args.type == 'train'

    setup_deterministic(fix_random, allow_tf32, deterministic, benchmark)

    log(f"Starting experiment: {magenta(cfg.exp_name)}, command: {magenta(args.type)}")


@callable_from_cfg
def test(
    model_cfg: dotdict = dotdict(),
    val_dataloader_cfg: dotdict = dotdict(type="DefaultDataloader"),
    test_dataloader_cfg: dotdict = None,
    runner_cfg: dotdict = dotdict(),

    base_device: str = 'cuda',

    record_images_to_tb: bool = False,
    print_test_progress: bool = True,
    dry_run: bool = False,

    **kwargs,
):
    preflight(**kwargs)

    if test_dataloader_cfg is not None:
        val_dataloader = DATALOADERS.build(test_dataloader_cfg)
    else:
        val_dataloader = DATALOADERS.build(val_dataloader_cfg)

    model = MODELS.build(model_cfg)
    model = model.to(base_device, non_blocking=True)

    runner = RUNNERS.build(runner_cfg,
                        model=model,
                        dataloader=None,
                        record_images_to_tb=record_images_to_tb,
                        print_test_progress=print_test_progress,
                        val_dataloader=val_dataloader)

    if dry_run: return runner

    launch_runner(**kwargs, runner_function=runner.test, runner_object=runner)


@callable_from_cfg
def train(
    model_cfg: dotdict = dotdict(),
    dataloader_cfg: dotdict = dotdict(type="DefaultDataloader"),
    val_dataloader_cfg: dotdict = dotdict(type="DefaultDataloader"),
    runner_cfg: dotdict = dotdict(),

    distributed: bool = False,
    find_unused_parameters: bool = False,

    dry_run: bool = False,
    print_model: bool = True,

    base_device: str = 'cuda',

    **kwargs,
):
    preflight(**kwargs)

    dataloader = DATALOADERS.build(dataloader_cfg)
    if not get_rank(): val_dataloader = DATALOADERS.build(val_dataloader_cfg)
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    rank = get_rank()
    device = torch.device(f'{base_device}:{rank}')
    torch.cuda.set_device(device)

    model = MODELS.build(model_cfg)
    model.to(device, non_blocking=True)

    if get_distributed():
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=find_unused_parameters,
        )

    runner = RUNNERS.build(runner_cfg,
                        model=model,
                        dataloader=dataloader,
                        val_dataloader=val_dataloader if not get_rank() else None)

    if print_model and not get_rank():
        pprint(model)
        try:
            nop = number_of_params(model)
            log(f'Number of parameters: {nop} ({nop / 1e6:.2f} M)')
        except ValueError as e:
            pass

    if dry_run: return runner

    launch_runner(**kwargs, runner_function=runner.train, runner_object=runner)


def main_entrypoint():
    globals()[args.type](cfg)


if __name__ == '__main__':
    main_entrypoint()
