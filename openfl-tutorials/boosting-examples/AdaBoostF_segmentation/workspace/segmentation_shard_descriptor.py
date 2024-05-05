# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""

import logging
from typing import List

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from sklearn.datasets import load_svmlight_file

logger = logging.getLogger(__name__)


class segmentationShardDataset(ShardDataset):
    """Mnist Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1, complete=False):
        """Initialize TinyImageNetDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x if complete else x[self.rank - 1::self.worldsize]
        self.y = y if complete else y[self.rank - 1::self.worldsize]

    def get_data(self):
        return self.x, self.y

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)


class segmentationShardDescriptor(ShardDescriptor):
    """Mnist Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize MnistShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        x_train, x_test, y_train, y_test = self.download_data()

        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_test, y_test)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train', complete=False):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return segmentationShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize,
            complete=complete
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['19']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Adult dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

    def download_data(self):
        X_tr, y_tr = load_svmlight_file("../segmentation_dataset/segmentation.tr.svmlight")
        X_te, y_te = load_svmlight_file("../segmentation_dataset/segmentation.te.svmlight")
        X_tr = X_tr.toarray()
        X_te = X_te.toarray()
        return X_tr, X_te, y_tr, y_te
