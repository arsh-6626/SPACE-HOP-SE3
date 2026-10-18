"""ISRO GSAT-12R loader for keypoint-guided JEPA pretraining.

Reuses the pose stage's crop (SPACE-HOP/src/isro_data.py) so the keypoints handed to
JointMaskCollator are in the same 224x224 frame as the image. Only pixel-level augmentations
may be applied after cropping; a geometric transform here would misalign the keypoint masks.
"""
import os
import sys
from logging import getLogger

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

_POSE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src')
if _POSE_SRC not in sys.path:
    sys.path.append(_POSE_SRC)
from isro_data import IsroPoseDataset, seed_worker  # noqa: E402

logger = getLogger()


class IsroJepa(Dataset):
    def __init__(self, root, camera_file, split='train', fold=1, transform=None, crop_size=224):
        self.ds = IsroPoseDataset(root, split, camera_file, fold=fold, train=True, crop_size=crop_size,
                                  return_keypoints=True, pixel_transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        s = self.ds[idx]
        return s['pixel_values'], s['keypoints'], {'idx': idx}


def make_isro(root, camera_file, batch_size, transform=None, collator=None, pin_mem=True, num_workers=8,
              world_size=1, rank=0, training=True, crop_size=224, split='train', fold=1, drop_last=True):
    dataset = IsroJepa(root, camera_file, split=split, fold=fold, transform=transform, crop_size=crop_size)
    logger.info(f'ISRO JEPA dataset created: {len(dataset)} samples (split={split}, fold={fold})')
    sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=training)
    loader = DataLoader(dataset, collate_fn=collator, sampler=sampler, batch_size=batch_size,
                        drop_last=drop_last, pin_memory=pin_mem, num_workers=num_workers,
                        persistent_workers=num_workers > 0,
                        worker_init_fn=seed_worker)
    return dataset, loader, sampler
