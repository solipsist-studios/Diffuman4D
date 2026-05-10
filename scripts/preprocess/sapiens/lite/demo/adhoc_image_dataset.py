# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image


class AdhocImageDataset(torch.utils.data.Dataset):
    """
    Dataset that loads images at their original resolution without preprocessing.
    
    The pose estimation pipeline follows a bbox-based approach:
    1. Images are loaded at original resolution
    2. Person bboxes are detected on the full image
    3. Each bbox is cropped and resized to the model's input size (e.g., 1024x1024)
    4. Pose estimation runs on the resized crops
    5. Keypoints are remapped back to original image coordinates
    
    This approach allows processing images of arbitrary resolution while using
    a model with fixed input dimensions (TorchScript limitation).
    """
    def __init__(self, image_paths, fmask_paths=None, shape=None):
        self.image_paths = image_paths
        self.fmask_paths = fmask_paths
        if self.fmask_paths is not None and len(self.fmask_paths) != len(
            self.image_paths
        ):
            raise ValueError("fmask_paths and image_paths must have the same length")
        if shape:
            assert len(shape) == 2
        self.shape = shape if shape else (1024, 1024)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Load image at original resolution (no resizing yet)
        image = Image.open(image_path)
        if self.fmask_paths is not None:
            fmask_path = self.fmask_paths[idx]
            fmask = Image.open(fmask_path)
            bg = Image.new(image.mode, image.size, (0, 0, 0))
            image = Image.composite(image, bg, fmask)

        # Convert to numpy array but keep original resolution
        image = np.array(image)
        return image_path, image
