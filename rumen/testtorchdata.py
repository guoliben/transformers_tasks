from torch.utils.data import IterableDataset, DataLoader
import math

class MyIterableDataset(IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

from torch.utils.data import get_worker_info

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)



if __name__ == '__main__':

    ds = MyIterableDataset(start=3, end=7) # [3, 4, 5, 6]
    # Single-process loading
    print(list(DataLoader(ds, num_workers=0)))
    # Directly doing multi-process loading
    print(list(DataLoader(ds, num_workers=2)))

    # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
    print(list(DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
    # With even more workers
    print(list(DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))