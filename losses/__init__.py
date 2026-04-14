import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tensor_list.append((input_tensor == i * torch.ones_like(input_tensor)).float())
        return torch.cat(tensor_list, dim=1)

    @staticmethod
    def _dice_loss(score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), "predict & target shape do not match"
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes


def entropy_loss(probabilities, num_classes):
    entropy = -1 * torch.sum(probabilities * torch.log(probabilities + 1e-6), dim=1)
    entropy = entropy / torch.tensor(np.log(num_classes), device=probabilities.device)
    return torch.mean(entropy)


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_prob = torch.sigmoid(input_logits)
        target_prob = torch.sigmoid(target_logits)
    else:
        input_prob = F.softmax(input_logits, dim=1)
        target_prob = F.softmax(target_logits, dim=1)
    return (input_prob - target_prob) ** 2


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=1), F.softmax(q, dim=1), reduction="none").mean()
    q_loss = F.kl_div(F.log_softmax(q, dim=1), F.softmax(p, dim=1), reduction="none").mean()
    return (p_loss + q_loss) / 2


def build_supervised_loss_terms(args):
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(args.model.out_channels)
    return ce_loss, dice_loss
