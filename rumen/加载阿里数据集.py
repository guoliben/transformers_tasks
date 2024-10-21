from torch.utils.data import Dataset
import json

class AFQMCXXX(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
                print(idx)
                print(line)
                print("=========")
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = AFQMCXXX('data/afqmc_public/train.json')
valid_data = AFQMCXXX('data/afqmc_public/dev.json')


#for d in train_data:
#	print(d)

print(train_data[0])
print(train_data[1])
print(train_data[2])
print(train_data[3])
print(train_data[4])
print(train_data[5])
print(train_data[6])
print("------------")
print(valid_data[0])
print(valid_data[1])
print(valid_data[2])
print(valid_data[3])
print(valid_data[4])
print(valid_data[5])
