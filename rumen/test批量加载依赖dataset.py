from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dataset = MyDataset(data)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

for batch in dataloader:
    print(batch)
