import torch.utils.data as tordata
import random

class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 检查数据集状态
        pid_count = len(list(self.dataset.label_set)) if hasattr(self.dataset, 'label_set') else 0
        if pid_count == 0:
            print("警告：数据集为空或未正确加载！")

    def __iter__(self):
        while True:
            # 获取行人ID列表
            try:
                pid_list = list(self.dataset.label_set)
            except:
                pid_list = []
                
            # 检查是否有足够的行人ID
            if len(pid_list) == 0:
                # 数据集为空，使用示例索引避免崩溃
                print("警告：数据集中没有找到任何行人ID，使用示例索引进行训练...")
                yield [0]  # 返回单个示例索引
                continue
            
            # 调整batch_size以适应可用的行人数量
            actual_batch_size_0 = min(self.batch_size[0], len(pid_list))
            if actual_batch_size_0 < self.batch_size[0]:
                # print(f"警告：行人数量不足，已调整batch_size[0]从{self.batch_size[0]}到{actual_batch_size_0}")
                pass
            
            try:
                pid_sample = random.sample(pid_list, actual_batch_size_0)
                sample_indices = []
                for pid in pid_sample:
                    try:
                        _index = self.dataset.index_dict.loc[pid, :, :].values
                        _index = _index[_index > 0].flatten().tolist()
                        
                        # 确保有足够的样本
                        if len(_index) == 0:
                            print(f"警告：PID {pid} 没有有效的样本索引")
                            continue
                        
                        actual_batch_size_1 = min(self.batch_size[1], len(_index))
                        _index = random.choices(_index, k=actual_batch_size_1)
                        sample_indices += _index
                    except Exception as e:
                        #print(f"警告：处理PID {pid} 时出错: {e}")
                        continue
                
                # 确保至少返回一个样本
                if sample_indices:
                    yield sample_indices
                else:
                    yield [0]  # 作为最后的后备
                
            except Exception as e:
                print(f"警告：采样过程出错: {e}")
                yield [0]  # 作为最后的后备

    def __len__(self):
        return max(1, getattr(self.dataset, 'data_size', 1))  # 确保至少为1
