from openfl.interface.interactive_api.experiment import DataInterface


class IrisDataset(DataInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def shard_descriptor(self):
        return self._shard_descriptor

    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        Local shard_descriptor  will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor

        self.train = shard_descriptor.get_dataset('train')
        self.test = shard_descriptor.get_dataset('val', complete=True)

    def __getitem__(self, index):
        return self.shard_descriptor[index]

    def __len__(self):
        return len(self.shard_descriptor)

    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        return self.train.get_data()

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        return self.test.get_data()

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        return len(self.train)

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.test)
