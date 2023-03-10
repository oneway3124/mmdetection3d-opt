from mmdet3d.models import build_detector
from mmcv import Config

# 定义配置文件路径
config_file = 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'
# 加载配置文件
cfg = Config.fromfile(config_file)
# 构建模型
model = build_detector(cfg.model)
# 打印模型结构
print(model)
