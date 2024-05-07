import logging
import warnings
from typing import Optional

import decord
from torch.utils.data import DataLoader, Dataset, Sampler

from dataloaders.decord_dataset import DecordDataset


class DecordVideoLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        batch_sampler: Optional[Sampler] = None,
        device: str = "cpu",
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = False,
        shuffle: Optional[bool] = False,
        sampler: Optional[Sampler] = None,
        prefetch_factor: int = 2,
        **kwargs,
    ):
        # Check if the dataset is a DecordDataset and the device is GPU
        if isinstance(dataset, DecordDataset):
            self._is_decord_dataset = True
        else:
            print(dataset)
            self._is_decord_dataset = False
        if num_workers > 0 and self._is_decord_dataset:
            if device == "gpu":
                warnings.warn(
                    "Using multiple workers with DecordVideoLoader and device='gpu' "
                    "is not supported yet. Please set num_workers=1."
                )
            elif device == "cpu" and sampler is None and batch_sampler is None:
                warnings.warn(
                    "Using multiple workers with DecordVideoLoader and device='cpu' "
                    "without a sampler may cause high RAM usage. If you are encountering "
                    "such problems, Please set num_workers=1 or use the DecordSampler."
                )
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            prefetch_factor=prefetch_factor,
            **kwargs,
        )

    def __iter__(self):
        # If the dataset is a DecordDataset, shuffle the dataset.
        if self._is_decord_dataset:
            self.dataset.shuffle()
        self._base_iterator = super().__iter__()
        while True:
            try:
                # Yield the next batch from the parent iterator
                yield next(self._base_iterator)
            except decord._ffi.base.DECORDError as e:
                # If the parent iterator raises a decord error, log it and continue
                logging.error(e)
            except StopIteration:
                # If the parent iterator raises StopIteration, break the loop
                break
