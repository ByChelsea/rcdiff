import time
import torch
import datetime
import random
import contextlib
import os
from tqdm import tqdm

from easyvolcap.engine import cfg, args

from easyvolcap.engine import RUNNERS, OPTIMIZERS, SCHEDULERS, RECORDERS, EVALUATORS, \
    MODERATORS
from easyvolcap.utils.net_utils import save_model, load_model, load_network, setup_deterministic
from easyvolcap.utils.data_utils import add_iter, to_cuda
from easyvolcap.utils.prof_utils import profiler_step
from easyvolcap.utils.dist_utils import get_rank
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *
from rcdiff.runners.ema_utils import ModelEMA


@RUNNERS.register_module()
class LowLevelMoPredRunner:
    def __init__(self,
                 model,
                 dataloader,
                 val_dataloader,
                 optimizer_cfg: dotdict = dotdict(),
                 scheduler_cfg: dotdict = dotdict(),

                 moderator_cfg: dotdict = dotdict(),
                 recorder_cfg: dotdict = dotdict(),
                 evaluator_cfg: dotdict = dotdict(),
                 ema_cfg: dotdict = dotdict(),

                 epochs: int = 400,
                 ep_iter: int = 1000,
                 eval_ep: int = 10,
                 save_ep: int = 20,
                 save_latest_ep: int = 10,
                 log_interval: int = 1,
                 record_interval: int = 1,
                 strict: bool = True,

                 resume: bool = True,
                 trained_model_dir: str = f'data/trained_model/{cfg.exp_name}',
                 load_epoch: int = -1,

                 clip_grad_norm: float = -1,
                 clip_grad_value: float = -1,
                 record_images_to_tb: bool = True,
                 print_test_progress: bool = True,
                 ):
        self.model = model
        self.dataloader = dataloader
        if dataloader:
            ep_iter = ep_iter
        else:
            ep_iter = len(val_dataloader.dataset)

        self.optimizer = OPTIMIZERS.build(optimizer_cfg, params=model.parameters())
        if scheduler_cfg.type == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=scheduler_cfg.milestones,
                                                                  gamma=scheduler_cfg.gamma)
        else:
            self.scheduler = SCHEDULERS.build(scheduler_cfg, optimizer=self.optimizer,
                                              decay_iter=epochs * ep_iter)

        if not get_rank():
            self.val_dataloader = val_dataloader
            self.recorder = RECORDERS.build(recorder_cfg, resume=resume)
            self.evaluator = EVALUATORS.build(evaluator_cfg)

        self.moderator = MODERATORS.build(moderator_cfg, runner=self, total_iter=epochs * ep_iter)

        self.epochs = epochs
        self.ep_iter = ep_iter
        self.eval_ep = eval_ep
        self.save_ep = save_ep
        self.save_latest_ep = save_latest_ep
        self.log_interval = log_interval
        self.record_interval = record_interval

        self.resume = resume
        self.strict = strict
        self.load_epoch = load_epoch
        self.trained_model_dir = trained_model_dir

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        self.gscaler = torch.cuda.amp.GradScaler(enabled=False)

        self.record_images_to_tb = record_images_to_tb
        self.print_test_progress = print_test_progress

        cfg.runner = self

        self.ema = None
        if ema_cfg.get('enabled', False):
            self.ema = ModelEMA(
                model=self.model,
                decay=ema_cfg.get('decay', 0.999),
                use_fp32=ema_cfg.get('use_fp32', True),
                device=next(self.model.parameters()).device,
                update_every=ema_cfg.get('update_every', 1),
                warmup_steps=ema_cfg.get('warmup_steps', 0),
            )

    @property
    def total_iter(self):
        return self.epochs * self.ep_iter

    def load_network(self):
        epoch = load_network(model=self.model,
                             model_dir=self.trained_model_dir,
                             resume=self.resume,
                             epoch=self.load_epoch,
                             strict=self.strict,
                             )
        if self.resume:
            self.load_ema(epoch if self.load_epoch < 0 else self.load_epoch)
        return epoch

    def load_model(self):
        epoch = load_model(model=self.model,
                           optimizer=self.optimizer,
                           scheduler=self.scheduler,
                           moderator=self.moderator,
                           model_dir=self.trained_model_dir,
                           resume=self.resume,
                           epoch=self.load_epoch,
                           strict=self.strict,
                           )
        if self.resume:
            self.load_ema(epoch if self.load_epoch < 0 else self.load_epoch)
        return epoch

    def save_network(self, epoch, latest: bool = True):
        save_model(model=self.model,
                   model_dir=self.trained_model_dir,
                   epoch=epoch,
                   latest=latest,
                   save_lim=100,
                   )
        self.save_ema(epoch, latest=latest)

    def save_model(self, epoch: int, latest: bool = True):
        save_model(model=self.model,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   moderator=self.moderator,
                   model_dir=self.trained_model_dir,
                   epoch=epoch,
                   latest=latest,
                   save_lim=100,
                   )
        self.save_ema(epoch, latest=latest)

    def test(self):
        setup_deterministic(fix_random=True, allow_tf32=False, deterministic=True, benchmark=True, seed=66)
        epoch = self.load_network()
        self.test_epoch(epoch)
        setup_deterministic(fix_random=False, allow_tf32=False, deterministic=False, benchmark=False)

    def train(self):
        epoch = self.load_model()

        train_generator = self.train_generator(epoch, self.ep_iter)

        for epoch in range(epoch, self.epochs):
            next(train_generator)

            if (epoch + 1) % self.save_ep == 0 and not get_rank():
                self.save_model(epoch, latest=False)

            if (epoch + 1) % self.save_latest_ep == 0 and not get_rank():
                self.save_model(epoch, latest=True)

            if (epoch + 1) % self.eval_ep == 0 and not get_rank():
                try:
                    self.test_epoch(epoch + 1)
                except Exception as e:
                    log(red('Error in validation pass, ignored and continuing'))
                    stacktrace()
                    stop_prog()
                    pass

    def train_epoch(self, epoch: int):
        train_generator = self.train_generator(epoch, self.ep_iter)
        for _ in train_generator: pass

    def train_generator(self, begin_epoch: int, yield_every: int = 1):
        epoch = begin_epoch
        self.model.train()
        start_time = time.perf_counter()
        for index, batch in enumerate(self.dataloader):
            iter = begin_epoch * self.ep_iter + index
            batch = add_iter(batch, iter, self.total_iter)
            batch = to_cuda(batch)
            data_time = time.perf_counter() - start_time

            with torch.cuda.amp.autocast(enabled=False):
                output: dotdict = self.model(batch, self.dataloader)
            loss: torch.Tensor = output.loss
            scalar_stats: dotdict = output.scalar_stats

            self.optimizer.zero_grad(set_to_none=True)
            loss = loss.mean()
            self.gscaler.scale(loss).backward()
            if self.clip_grad_norm > 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            if self.clip_grad_value > 0: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
            self.optimizer.step()
            self.scheduler.step()
            self.moderator.step()
            if self.ema is not None:
                self.ema.maybe_update(self.model, global_step=iter)

            end_time = time.perf_counter()
            batch_time = end_time - start_time
            start_time = end_time
            if (iter + 1) % self.log_interval == 0 and not get_rank():
                scalar_stats = dotdict(
                    {k: v.mean().item() for k, v in scalar_stats.items()})

                lr = self.optimizer.param_groups[0]['lr']
                max_mem = torch.cuda.max_memory_allocated() / 2 ** 20
                scalar_stats.data = data_time
                scalar_stats.batch = batch_time
                scalar_stats.lr = lr
                scalar_stats.max_mem = max_mem

                self.recorder.iter = iter
                self.recorder.epoch = epoch
                self.recorder.update_scalar_stats(scalar_stats)

                eta_seconds = self.recorder.scalar_stats.batch.global_avg * (self.total_iter - self.recorder.iter)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_stats = dotdict()
                log_stats.eta = eta_string
                log_stats.update(self.recorder.log_stats)

                display_table(log_stats)

                if (iter + 1) % self.record_interval == 0:
                    self.recorder.record(self.dataloader.dataset.split)

            if yield_every > 0 and (iter + 1) % yield_every == 0:
                yield output
                self.model.train()

            if (iter + 1) % self.ep_iter == 0:
                epoch = epoch + 1

            profiler_step()

    def test_epoch(self, epoch: int):
        test_generator = self.test_generator(epoch, -1)
        for _ in test_generator: pass

    def test_generator(self, epoch: int, yield_every: int = 1):
        self.model.eval()
        save_dir = f'data/trained_model/{cfg.exp_name}/eval/{epoch}'
        for index, batch in enumerate(tqdm(self.val_dataloader, disable=not self.print_test_progress)):
            name, unique_files = batch.meta.file_name[0], []
            if os.path.exists(save_dir):
                files = [f[:-7] for f in os.listdir(save_dir)]
                unique_files = list(set(files))
            if name in unique_files:
                continue
            iter = epoch * self.ep_iter - 1
            batch = add_iter(batch, iter, self.total_iter)
            batch = to_cuda(batch)

            apply_ctx = self.ema.apply_to(self.model) if self.ema is not None else contextlib.nullcontext()
            with apply_ctx, torch.inference_mode() and torch.cuda.amp.autocast(enabled=False):
                output: dotdict = self.model.inference(batch, self.val_dataloader)
                scalar_stats = self.evaluator.evaluate(output, batch, self.val_dataloader)
                self.evaluator.save_and_visualize(output, batch, cfg.exp_name, epoch, index, self.val_dataloader)

            self.recorder.iter = iter
            self.recorder.epoch = epoch
            self.recorder.update_scalar_stats(scalar_stats)
            self.recorder.record(self.val_dataloader.dataset.split)

            if yield_every > 0 and (iter + 1) % yield_every == 0:
                yield output
                self.model.eval()

            profiler_step()

        scalar_stats = self.evaluator.summarize()
        self.recorder.update_scalar_stats(scalar_stats)
        self.recorder.record(self.val_dataloader.dataset.split)

    def _ema_path(self, epoch, latest=False):
        if latest:
            return os.path.join(self.trained_model_dir, 'latest_ema.pth')
        return os.path.join(self.trained_model_dir, f'epoch_{epoch:04d}_ema.pth')

    def save_ema(self, epoch, latest=True):
        if self.ema is None or get_rank():
            return
        path = self._ema_path(epoch, latest=latest)
        os.makedirs(self.trained_model_dir, exist_ok=True)
        torch.save(self.ema.state_dict(), path)

    def load_ema(self, epoch):
        if self.ema is None:
            return
        candidates = []
        if epoch is not None and epoch >= 0:
            candidates.append(self._ema_path(epoch, latest=False))
        candidates.append(self._ema_path(epoch, latest=True))
        for p in candidates:
            if os.path.exists(p):
                state = torch.load(p, map_location='cpu')
                self.ema.load_state_dict(state)
                log(green(f'Loaded EMA from: {p}'))
                return
        log(yellow('EMA state not found; starting fresh.'))
