import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELossWrapper(nn.Module):
    def __init__(self, w_spoof=1.0, w_bona_fide=9.0):
        super().__init__()
        self.w_spoof = w_spoof
        self.w_bona_fide = w_bona_fide

    def forward(self, scores, target, **batch):
        probs = F.sigmoid(scores)

        weight = torch.zeros(scores.size(0), device=scores.device)
        weight[target == 1] = self.w_bona_fide
        weight[target == 0] = self.w_spoof

        return F.binary_cross_entropy(probs, target.float(), weight=weight)
