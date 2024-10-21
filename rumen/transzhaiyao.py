from torch.utils.data import Dataset

max_dataset_size = 200000

class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                items = line.strip().split('!=!')
                assert len(items) == 2
                Data[idx] = {
                    'title': items[0],
                    'content': items[1]
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = LCSTS('data/lcsts_tsv/data1.tsv')
valid_data = LCSTS('data/lcsts_tsv/data2.tsv')
test_data = LCSTS('data/lcsts_tsv/data3.tsv')
