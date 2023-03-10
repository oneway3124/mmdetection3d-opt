from mmcv import Config
from mmcv.runner import Runner
from mmdet3d.models import build_model
from mmcv.runner import (build_optimizer,build_runner)
from mmdet.datasets import build_dataloader,build_dataset
from mmdet.datasets import DATASETS
from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdet.utils import get_root_logger as get_mmdet_root_logger

DATASETS.register_module(KittiDataset, force=True)


# 定义配置文件路径
# cfg_file = 'configs/classification/my_config.py'
cfg_file = 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'

# 加载配置文件
cfg = Config.fromfile(cfg_file)

# 创建模型和优化器
model = build_model(cfg.model)
optimizer = build_optimizer(model, cfg.optimizer)

train_dataset = build_dataset(cfg.data.train)

# 创建数据加载器
train_loader = build_dataloader(cfg.data.train,
                                samples_per_gpu = cfg.data.samples_per_gpu,
                                workers_per_gpu = cfg.data.workers_per_gpu,shuffle=False)
val_loader = build_dataloader(cfg.data.val,
                                samples_per_gpu = cfg.data.samples_per_gpu,
                                workers_per_gpu = cfg.data.workers_per_gpu,shuffle=False)

# 创建运行器
# runner = Runner(model, batch_processor, optimizer, cfg.work_dir, logger)

logger = get_mmdet_root_logger(log_level=cfg.log_level)
runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger))

cfg.total_epochs = 20
# 开始训练
# runner.run(train_loader, val_loader, cfg.total_epochs)
#有问题，再说吧