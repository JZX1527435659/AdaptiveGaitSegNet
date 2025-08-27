import torch.utils.data as tordata
import random

class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            pid_list = list(self.dataset.label_set)
            if len(pid_list) < self.batch_size[0]:
                raise ValueError(f"❌ 当前数据集中 PID 数量不足 batch_size[0]，请调整 config：PID数={len(pid_list)}, batch_size[0]={self.batch_size[0]}")

            pid_sample = random.sample(pid_list, self.batch_size[0])
            sample_indices = []
            for pid in pid_sample:
                _index = self.dataset.index_dict.loc[pid, :, :].values
                _index = _index[_index > 0].flatten().tolist()
                _index = random.choices(_index, k=self.batch_size[1])
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size
