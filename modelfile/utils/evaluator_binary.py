import torch
import torch.nn.functional as F
import numpy as np


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def evaluation(data, config):
    dataset = config['dataset'].split('-')[0].upper()
    feature, view, seq_type, label = data
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}

    if dataset not in probe_seq_dict:
        probe_seq_dict[dataset] = [[str(seq_type[0]) if len(seq_type) > 0 else 'silhouettes']]
        gallery_seq_dict[dataset] = [[str(seq_type[0]) if len(seq_type) > 0 else 'silhouettes']]

    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x)
                    if probe_x.shape[0] == gallery_x.shape[0] and np.array_equal(probe_y, gallery_y):
                        eye = torch.eye(dist.size(0), device=dist.device)
                        dist = dist + eye * 1e6
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / max(1, dist.shape[0]), 2)

    return acc


def evaluate_binary_classification(test_feat):
    """
    输入: (label_prob_numpy, label_list)
    输出: (acc, precision, recall, f1, conf_matrix)
    """
    label_prob, label = test_feat
    y_true = np.array([int(l) for l in label])
    y_pred = np.argmax(label_prob, axis=1)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    acc = (tp + tn) / max(1, (tp + tn + fp + fn)) * 100.0
    precision = tp / max(1, (tp + fp)) * 100.0
    recall = tp / max(1, (tp + fn)) * 100.0
    f1 = (2 * precision * recall / max(1e-6, (precision + recall))) if (precision + recall) > 0 else 0.0

    conf_matrix = np.array([[tn, fp], [fn, tp]], dtype=int)
    return acc, precision, recall, f1, conf_matrix


def save_confusion_matrix(conf_matrix, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=160)
    im = ax.imshow(conf_matrix, cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Normal', 'Park'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Normal', 'Park'])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(conf_matrix[i, j]), va='center', ha='center', color='black')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
