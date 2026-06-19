import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

DATASETS = {
    'mvtec': ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
              'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood'],
    'visa':  ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
              'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
    'btad':  ['01', '02', '03'],
}


def get_class_list(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: {list(DATASETS.keys())}")
    obj_list = DATASETS[dataset_name]
    class_name_map_class_id = {k: i for i, k in enumerate(obj_list)}
    return obj_list, class_name_map_class_id


class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        meta_info = json.load(open(os.path.join(root, 'meta.json'), 'r'))
        meta_info = meta_info[mode]

        self.data_all = []
        for cls_name in meta_info:
            self.data_all.extend(meta_info[cls_name])

        self.obj_list, self.class_name_map_class_id = get_class_list(dataset_name)

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        item = self.data_all[index]
        img_path = item['img_path']
        mask_path = item['mask_path']
        cls_name = item['cls_name']
        anomaly = item['anomaly']

        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and img_mask is not None:
            img_mask = self.target_transform(img_mask)
        if img_mask is None:
            img_mask = []

        return {
            'img': img,
            'img_mask': img_mask,
            'cls_name': cls_name,
            'anomaly': anomaly,
            'img_path': os.path.join(self.root, img_path),
            'cls_id': self.class_name_map_class_id[cls_name],
        }


class RefDataset(data.Dataset):
    '''Dataset that returns a query image alongside a randomly sampled normal reference.'''

    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        meta_info = json.load(open(os.path.join(root, 'meta.json'), 'r'))
        modes_to_load = ['train', 'test'] if mode == 'all' else [mode]

        self.data_all = []
        self.cls_names = []
        for m in modes_to_load:
            if m in meta_info:
                for cls_name, items in meta_info[m].items():
                    if cls_name not in self.cls_names:
                        self.cls_names.append(cls_name)
                    self.data_all.extend(items)

        self.obj_list, self.class_name_map_class_id = get_class_list(dataset_name)

        self.normal_indices = {cls: [] for cls in self.cls_names}
        for i, item in enumerate(self.data_all):
            if item['anomaly'] == 0:
                self.normal_indices[item['cls_name']].append(i)

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        item = self.data_all[index]
        img_path = item['img_path']
        mask_path = item['mask_path']
        cls_name = item['cls_name']
        anomaly = item['anomaly']

        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and img_mask is not None:
            img_mask = self.target_transform(img_mask)
        if img_mask is None:
            img_mask = []

        normal_pool = [i for i in self.normal_indices[cls_name] if i != index]
        ref_index = random.choice(normal_pool) if normal_pool else index
        ref_item = self.data_all[ref_index]
        ref_img = Image.open(os.path.join(self.root, ref_item['img_path']))
        if self.transform is not None:
            ref_img = self.transform(ref_img)

        return {
            'img': img,
            'ref_img': ref_img,
            'img_mask': img_mask,
            'cls_name': cls_name,
            'anomaly': anomaly,
            'img_path': os.path.join(self.root, img_path),
            'cls_id': self.class_name_map_class_id[cls_name],
        }
