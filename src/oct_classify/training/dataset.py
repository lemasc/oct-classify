from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transforms

from oct_classify.data.models import ImageRecord
from oct_classify.data.preprocessing import PreprocessingSpec, preprocess_image
from oct_classify.data.taxonomy import UnifiedLabel
from oct_classify.training.config import AugmentationConfig

CLASS_ORDER = (UnifiedLabel.NORMAL, UnifiedLabel.AMD, UnifiedLabel.DME)


def labels_for_available(available_labels: Iterable[UnifiedLabel]) -> tuple[UnifiedLabel, ...]:
    available = frozenset(available_labels)
    labels = tuple(label for label in CLASS_ORDER if label in available)
    if len(labels) < 2:
        raise ValueError("A supervised source must have at least two available labels.")
    return labels


def load_split_records(path: Path, split: str) -> list[ImageRecord]:
    from oct_classify.data.manifest import read_jsonl

    records = [record for record in read_jsonl(path) if record.split == split]
    if not records:
        raise ValueError(f"No {split!r} records in split manifest: {path}")
    return records


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        root: Path,
        records: Iterable[ImageRecord],
        class_labels: tuple[UnifiedLabel, ...],
        preprocessing: PreprocessingSpec,
        mean: np.ndarray,
        stdev: np.ndarray,
        augmentation: AugmentationConfig,
        *,
        training: bool,
    ) -> None:
        self.root = root
        self.class_labels = class_labels
        self.preprocessing = preprocessing
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.stdev = torch.tensor(np.maximum(stdev, 1e-6), dtype=torch.float32).view(3, 1, 1)
        self.augmentation = augmentation
        self.training = training
        class_indices = {label: index for index, label in enumerate(class_labels)}
        self.records = [record for record in records if record.label in class_indices]
        if not self.records:
            raise ValueError("No records remain after filtering to the requested class labels.")
        self.targets = [class_indices[record.label] for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record = self.records[index]
        with Image.open(self.root / record.path) as image:
            array = preprocess_image(image, self.preprocessing)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if self.training:
            # OCT B-scans have no clinically meaningful left-to-right orientation for this task.
            if random.random() < self.augmentation.horizontal_flip_probability:
                tensor = torch.flip(tensor, dims=(2,))
            if self.augmentation.rotation_degrees:
                tensor = transforms.rotate(
                    tensor,
                    random.uniform(
                        -self.augmentation.rotation_degrees, self.augmentation.rotation_degrees
                    ),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
            if self.augmentation.brightness_jitter:
                tensor = transforms.adjust_brightness(
                    tensor,
                    random.uniform(
                        1 - self.augmentation.brightness_jitter,
                        1 + self.augmentation.brightness_jitter,
                    ),
                )
            if self.augmentation.contrast_jitter:
                tensor = transforms.adjust_contrast(
                    tensor,
                    random.uniform(
                        1 - self.augmentation.contrast_jitter,
                        1 + self.augmentation.contrast_jitter,
                    ),
                )
            if random.random() < self.augmentation.gaussian_blur_probability:
                tensor = transforms.gaussian_blur(
                    tensor, kernel_size=[3, 3], sigma=self.augmentation.gaussian_blur_sigma
                )
        return (tensor - self.mean) / self.stdev, self.targets[index], record.path
