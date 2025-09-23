import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, batch_size: int, hard_or_full: str = 'both', margin: float = 1.0):
        """
        Triplet loss with options for hard or full mining.
        :param batch_size: number of samples per batch
        :param hard_or_full: 'hard', 'full', or 'both'
        :param margin: triplet loss margin
        """
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin
        self.hard_or_full = hard_or_full

    def forward(self, feature: torch.Tensor, label: torch.Tensor):
        """
        :param feature: Tensor of shape [n, m, d], where
                        n = number of classes (P), m = number of samples per class (M), d = feature dim
        :param label: Tensor of shape [n, m], label matrix corresponding to features
        :return: full_loss_mean, hard_loss_mean, mean_dist, full_loss_count
        """
        n, m, d = feature.size()

        # [n, m, m] pairwise distance matrix for each class block
        dist = self.batch_dist(feature)
        mean_dist = dist.mean(dim=1).mean(dim=1)  # [n], average distance per class block

        # Construct label mask matrices
        label_flat = label.view(-1)
        label_matrix = label_flat.unsqueeze(0) == label_flat.unsqueeze(1)  # [N, N]
        hp_mask = label_matrix.clone()
        hn_mask = ~label_matrix

        dist_flat = self.pairwise_dist(feature.view(-1, d))  # [N, N]

        # ---- Hard Mining ----
        try:
            hard_ap = (dist_flat * hp_mask.float()).view(n, m, -1).max(dim=2)[0]
            hard_an = (dist_flat + (1e5 * hp_mask.float())).view(n, m, -1).min(dim=2)[0]
            hard_loss = F.relu(self.margin + hard_ap - hard_an)
            hard_loss_mean = hard_loss.mean(dim=1)
        except Exception as e:
            print("⚠️ Hard triplet error:", e)
            hard_loss_mean = torch.zeros(n, device=feature.device)

        # ---- Full Mining ----
        try:
            ap_dist = dist_flat.unsqueeze(2)  # [N, N, 1]
            an_dist = dist_flat.unsqueeze(1)  # [N, 1, N]

            pos_mask = hp_mask.float()
            neg_mask = hn_mask.float()

            pos_dist = ap_dist * pos_mask.unsqueeze(2)  # [N, N, 1]
            neg_dist = an_dist * neg_mask.unsqueeze(1)  # [N, 1, N]

            full_loss = F.relu(self.margin + pos_dist - neg_dist)  # [N, N, N]
            full_loss_sum = full_loss.sum(dim=(1, 2))
            full_loss_count = (full_loss > 1e-6).sum(dim=(1, 2)).float() + 1e-5  # prevent division by zero
            full_loss_mean = full_loss_sum / full_loss_count
        except Exception as e:
            print("⚠️ Full triplet error:", e)
            full_loss_mean = torch.zeros(n, device=feature.device)
            full_loss_count = torch.ones(n, device=feature.device)

        return full_loss_mean, hard_loss_mean, mean_dist, full_loss_count

    def batch_dist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Euclidean distance within each sample group.
        :param x: [n, m, d]
        :return: [n, m, m]
        """
        n, m, d = x.size()
        x_exp = x.view(n, m, 1, d)
        y_exp = x.view(n, 1, m, d)
        dist = ((x_exp - y_exp) ** 2).sum(dim=3)
        return dist

    def pairwise_dist(self, x):
        """
        Compute pairwise Euclidean distance matrix.
        :param x: [N, D]
        :return: [N, N]
        """
        squared = torch.sum(x ** 2, dim=1, keepdim=True)
        dist = squared + squared.t() - 2 * torch.matmul(x, x.t())
        dist = torch.clamp(dist, min=1e-6)
        return dist.sqrt()

    def batch_dist(self, x):
        """
        Compute pairwise distance within each class batch.
        :param x: [n, m, d]
        :return: [n, m, m]
        """
        n, m, d = x.size()
        dist = []
        for i in range(n):
            xi = x[i]  # [m, d]
            dist.append(self.pairwise_dist(xi).unsqueeze(0))  # [1, m, m]
        return torch.cat(dist, dim=0)

