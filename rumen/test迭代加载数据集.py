from torch.utils.data import IterableDataset
import json

class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, 'rt') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample

train_data = IterableAFQMC('data/afqmc_public/train.json')

train_iterator = iter(train_data)
print(next(train_iterator))
print(next(train_iterator))
print(next(train_iterator))
print(next(train_iterator))
print(next(train_iterator))
print(next(train_iterator))
print(next(train_iterator))
print(next(train_iterator))
print(next(train_iterator))
print(next(train_iterator))

