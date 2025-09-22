import os
import math
import os.path as osp
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata

from .network import TripletLoss
from .utils import TripletSampler
from .gaitset_focal_edge import SetNet


class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 img_size=64,
                 ce_weight=0.5,
                 triplet_weight=0.5,
                 early_stop_patience=100,
                 early_stop_delta=0.001):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.img_size = img_size

        self.encoder = SetNet(self.hidden_dim, num_classes=train_pid_num).float()
        self.encoder = nn.DataParallel(self.encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.encoder.cuda()
        self.triplet_loss.cuda()

        self.optimizer = optim.Adam(
            [{'params': self.encoder.parameters()},
             {'params': self.triplet_loss.parameters()}],
            lr=self.lr)

        self.ce_loss = nn.CrossEntropyLoss().cuda()

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        self.sample_type = 'all'
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight

        # 早停相关参数
        self.early_stop_patience = early_stop_patience  # 容忍多少次验证损失没有改善
        self.early_stop_delta = early_stop_delta  # 最小改善阈值
        self.best_loss = float('inf')  # 最佳验证损失
        self.early_stop_counter = 0  # 计数器
        self.best_model_iter = 0  # 最佳模型的迭代次数

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                        np.concatenate([seqs[i][j]
                                        for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                        if i < batch_size], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            seqs = [np.asarray([np.pad(seqs[j][_],
                                       ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                       'constant',
                                       constant_values=0)
                               for _ in range(gpu_num)])
                    for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.encoder.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1
            self.optimizer.zero_grad()

            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, label_prob = self.encoder(*seq, batch_frame)

            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            triplet_feature = feature.permute(1, 0, 2).contiguous()
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)
            if self.hard_or_full_trip == 'hard':
                triplet_loss = hard_loss_metric.mean()
            else:
                triplet_loss = full_loss_metric.mean()

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())

            ce_loss = self.ce_loss(label_prob, target_label)
            joint_loss = self.triplet_weight * triplet_loss + self.ce_weight * ce_loss

            if joint_loss > 1e-9:
                joint_loss.backward()
                self.optimizer.step()

            if self.restore_iter % 10 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()

            if self.restore_iter % 10 == 0:
                # 从数据配置中获取data_type和direction
                data_type = getattr(self.train_source, 'data_type', 'silhouettes') if hasattr(self.train_source, 'data_type') else 'silhouettes'
                direction = getattr(self.train_source, 'direction', 'front') if hasattr(self.train_source, 'direction') else 'front'
                self.save(data_type=data_type, direction=direction)
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                
                # 添加早停逻辑
                val_loss = self.validate()
                print(', validation_loss={0:.8f}'.format(val_loss), end='')
                if self.check_early_stopping(val_loss):
                    # 加载最佳模型并退出训练循环
                    print("\nLoading best model...")
                    self.load(self.best_model_iter, data_type=data_type, direction=direction)
                    break
                    
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, _ = self.encoder(*seq, batch_frame)
            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def transform_logits(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        prob_list = []
        label_list = []
        for i, x in enumerate(data_loader):
            seq, _, _, label, batch_frame = x
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            _, label_prob = self.encoder(*seq, batch_frame)
            prob_list.append(label_prob.data.cpu().numpy())
            label_list += label

        return np.concatenate(prob_list, 0), label_list

    def validate(self):
        """使用测试集作为验证集计算损失"""
        self.encoder.eval()
        # 使用测试数据作为验证数据
        val_loader = tordata.DataLoader(
            dataset=self.test_source,
            batch_size=self.batch_size[0] * self.batch_size[1],
            sampler=tordata.sampler.SequentialSampler(self.test_source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        val_label_set = list(self.test_source.label_set)
        val_label_set.sort()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for seq, view, seq_type, label, batch_frame in val_loader:
                for i in range(len(seq)):
                    seq[i] = self.np2var(seq[i]).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()

                feature, label_prob = self.encoder(*seq, batch_frame)

                target_label = [val_label_set.index(l) for l in label]
                target_label = self.np2var(np.array(target_label)).long()

                # 计算交叉熵损失
                ce_loss = self.ce_loss(label_prob, target_label)
                total_loss += ce_loss.item() * len(label)
                total_samples += len(label)

        self.encoder.train()
        return total_loss / total_samples if total_samples > 0 else float('inf')

    def check_early_stopping(self, val_loss):
        """
        检查是否满足早停条件
        :param val_loss: 当前验证损失
        :return: 是否应该早停
        """
        # 如果当前损失比最佳损失小一个阈值，则认为有改善
        if val_loss < self.best_loss - self.early_stop_delta:
            self.best_loss = val_loss
            self.early_stop_counter = 0
            self.best_model_iter = self.restore_iter
            # 保存最佳模型
            self.save()
        else:
            self.early_stop_counter += 1
            # 如果计数器达到耐心值，则触发早停
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping triggered at iteration {self.restore_iter}")
                print(f"Best validation loss: {self.best_loss:.6f} at iteration {self.best_model_iter}")
                return True
        return False

    def save(self, data_type=None, direction=None):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        if data_type is not None and direction is not None:
            torch.save(self.encoder.state_dict(),
                       osp.join('checkpoint', self.model_name,
                                '{}-{}-{}-{:0>5}-encoder.ptm'.format(
                                    self.save_name, data_type, direction, self.restore_iter)))
            torch.save(self.optimizer.state_dict(),
                       osp.join('checkpoint', self.model_name,
                                '{}-{}-{}-{:0>5}-optimizer.ptm'.format(
                                    self.save_name, data_type, direction, self.restore_iter)))
        else:
            torch.save(self.encoder.state_dict(),
                       osp.join('checkpoint', self.model_name,
                                '{}-{:0>5}-encoder.ptm'.format(
                                    self.save_name, self.restore_iter)))
            torch.save(self.optimizer.state_dict(),
                       osp.join('checkpoint', self.model_name,
                                '{}-{:0>5}-optimizer.ptm'.format(
                                    self.save_name, self.restore_iter)))

    def load(self, restore_iter, data_type=None, direction=None):
        import torch
        import os.path as osp
        if data_type is not None and direction is not None:
            # 根据实际存在的文件名格式构建路径
            ckpt_path = osp.join('checkpoint', self.model_name,
                                 '{}-{}-{}-{:0>5}-encoder.ptm'.format(self.save_name, data_type, direction, restore_iter))
        else:
            ckpt_path = osp.join('checkpoint', self.model_name,
                                 '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))
        print(f"Loading checkpoint from: {ckpt_path}")  # 添加调试信息
        state_dict = torch.load(ckpt_path, map_location='cpu')
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        if isinstance(self.encoder, torch.nn.DataParallel):
            final_state_dict = {'module.' + k: v for k, v in new_state_dict.items()}
        else:
            final_state_dict = new_state_dict
        self.encoder.load_state_dict(final_state_dict, strict=False)
