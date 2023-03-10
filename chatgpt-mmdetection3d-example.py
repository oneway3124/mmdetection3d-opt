import os.path as osp
import tempfile
import warnings

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer, build_runner, obj_from_dict)
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet3d.apis.train import train_detector
# from mmdet3d.core import eval_map

# import torch.distributed as dist
# import torch.multiprocessing as mp

import os
import torch.distributed as dist

os.environ['WORLD_SIZE'] = '8'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '1234'

# 构建训练和验证集
cfg_path = 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'
cfg = mmcv.Config.fromfile(cfg_path)
train_dataset = build_dataset(cfg.data.train)

# train_dataloader_cfg = cfg.data
# train_dataloader_cfg['shuffle'] = True
# train_dataloader_cfg['workers_per_gpu'] = 2
# train_loader = build_dataloader(train_dataset, **train_dataloader_cfg)
# val_dataset = build_dataset(cfg.data.val)
# val_dataloader_cfg = cfg.data
# val_dataloader_cfg['shuffle'] = False
# val_dataloader_cfg['workers_per_gpu'] = 2
# val_loader = build_dataloader(val_dataset, **val_dataloader_cfg)

train_loader = build_dataloader(cfg.data.train,
                                samples_per_gpu = cfg.data.samples_per_gpu,
                                workers_per_gpu = cfg.data.workers_per_gpu,shuffle=False)
val_loader = build_dataloader(cfg.data.val,
                                samples_per_gpu = cfg.data.samples_per_gpu,
                                workers_per_gpu = cfg.data.workers_per_gpu,shuffle=False)

dist.init_process_group(backend='nccl', init_method='env://')
# 构建模型
model = build_detector(cfg.model)
if torch.cuda.is_available():
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
else:
    model = MMDataParallel(model)

# 构建优化器和学习率调度器
optimizer_cfg = cfg.optimizer
optimizer = build_optimizer(model, optimizer_cfg)

lr_cfg = cfg.lr_config
lr_scheduler_cfg = lr_cfg.policy
lr_scheduler = obj_from_dict(lr_scheduler_cfg, torch.optim.lr_scheduler,
                             dict(optimizer=optimizer, **lr_cfg))

# 构建训练器
runner_cfg = cfg.runner
if 'max_iters' in runner_cfg:
    warnings.warn('max_iters in runner is deprecated, '
                  'please consider using max_epochs instead')
if runner_cfg['type'] == 'EpochBasedRunner':
    runner_cfg = dict(max_epochs=runner_cfg['max_epochs'])
runner = build_runner(runner_cfg)
runner.register_hook(DistSamplerSeedHook())

# 构建检查点保存和评估
checkpoint_cfg = cfg.checkpoint_config
checkpoint_cfg['interval'] = 1
checkpoint_cfg['save_optimizer'] = False
runner.register_hook(
    OptimizerHook(interval=checkpoint_cfg['interval'], save_optimizer=False))
runner.register_hook(mmcv.visualization.TensorboardLoggerHook())

# 开始训练
work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(cfg_path))[0])
train_detector(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    runner,
    checkpoint_cfg,
    work_dir=work_dir,
    log_level=cfg.log_level)
