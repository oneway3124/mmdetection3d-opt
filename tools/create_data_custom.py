import argparse
from pathlib import Path
from concurrent import futures as futures
import os
import numpy as np
import json
import pickle
from tools.create_gt_database_custom import create_groundtruth_database_custom


def _read_sampleset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def _get_lidar_path(samples, idx):
    return os.path.join('bins', samples[idx]+'.bin')


def _get_label_info(root_path, samples, idx):
    label_rel_path = os.path.join('labels', samples[idx] + '.json')
    label_path = root_path / label_rel_path
    label_str = ""
    with open(label_path, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            label_str = label_str + line.replace(" ","").replace("\t","").strip()
    label_info = json.loads(label_str)
    num_objs = len(label_info)
    gt_bboxes = []
    gt_names = []
    for obj in label_info:
        bbox = [
            obj["psr"]["position"]["x"],obj["psr"]["position"]["y"], obj["psr"]["position"]["z"],
            obj["psr"]["scale"]["x"], obj["psr"]["scale"]["y"], obj["psr"]["scale"]["z"],
            obj["psr"]["rotation"]["z"]
        ]
        gt_bboxes.append(bbox)
        obj_name = obj["obj_type"]
        gt_names.append(obj_name)
    annos = {}
    annos['box_type_3d'] = 'LiDAR'
    annos['gt_bboxes_3d'] = np.array([i for i in gt_bboxes])
    annos['gt_names'] = gt_names

    return annos


def get_info(path,
             samples=[],
             sample_ids=[],
             num_worker=8
             ):

    root_path = Path(path)
    if not isinstance(sample_ids, list):
        sample_ids = list(range(sample_ids))

    def map_func(idx):
        info = {}
        info['sample_idx'] = idx
        info['lidar_points'] = {'lidar_path': _get_lidar_path(samples, idx)}
        info['annos'] = _get_label_info(root_path, samples, idx)

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        lable_infos = executor.map(map_func, sample_ids)

    return list(lable_infos)


def create_custom_info_file(data_path, save_path=None, relative_path=True, class_names=None):
    sampleset_folder = Path(data_path) / 'SampleSets'
    train_samples = _read_sampleset_file(str(sampleset_folder / 'train.txt'))
    val_samples = _read_sampleset_file(str(sampleset_folder / 'val.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    infos_train = get_info(data_path, samples=train_samples, sample_ids=len(train_samples))
    infos_val = get_info(data_path, samples=val_samples, sample_ids=len(val_samples))

    infos_train_save_path = save_path / 'train_annotation.pkl'
    infos_val_save_path = save_path / 'val_annotation.pkl'

    if not os.path.exists(infos_train_save_path):
        with open(infos_train_save_path, 'wb') as f:
            pickle.dump(infos_train, f)

    if not os.path.exists(infos_val_save_path):
        with open(infos_val_save_path, 'wb') as f:
            pickle.dump(infos_val, f)

    create_groundtruth_database_custom(
        'Custom3DDataset',
        data_path,
        info_path=infos_train_save_path,
        relative_path=False,
        classes=class_names
    )


def custom_data_prep(root_path, out_dir, class_names):
    create_custom_info_file(root_path, class_names=class_names)


def main():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./toy_kitti',
        help='specify the root path of dataset')

    parser.add_argument(
        '--out-dir',
        type=str,
        default='./toy_kitti',
        required=False,
        help='name of info pkl')

    args = parser.parse_args()

    class_names_txt = os.path.join(args.root_path, 'classnames.txt')
    class_names = []
    with open(class_names_txt, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            class_names.append(line.strip())

    custom_data_prep(
        root_path=args.root_path,
        out_dir=args.out_dir,
        class_names=class_names
    )


if __name__ == "__main__":
    main()
