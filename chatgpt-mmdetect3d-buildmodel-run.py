from mmdet3d.models import build_detector
from mmcv import Config
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset

# 定义配置文件路径
config_file = 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'
# 加载配置文件
cfg = Config.fromfile(config_file)
# 构建模型
model = build_detector(cfg.model)
# 打印模型结构
print(model)

train_dataset = 'data/kitt/kitti_infos_train.pkl'
val_dataset = 'data/kitt/kitti_infos_val.pkl'
train_pipeline = cfg.train_pipeline
val_pipeline = cfg.eval_pipeline
train_cfg = cfg.model.train_cfg
test_cfg = cfg.model.test_cfg

# 构建数据集
train_dataset = build_dataset(cfg.data.train)
datasets = [build_dataset(cfg.data.train)]
# Train the detector
train_detector(
    model,
    datasets,
    train_cfg,
    distributed=False,
    validate=True,
    timestamp=None,
    meta=None
)