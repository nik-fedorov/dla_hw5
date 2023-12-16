import numpy as np

from anti_spoof.base.base_metric import BaseMetric

from .calculate_eer import compute_eer


class EER(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bonafide_scores = np.array([])
        self.spoof_scores = np.array([])

    def accumulate(self, scores, target, **batch):
        self.bonafide_scores = np.concatenate((self.bonafide_scores, scores[target == 1].cpu().detach().numpy()))
        self.spoof_scores = np.concatenate((self.spoof_scores, scores[target == 0].cpu().detach().numpy()))

    def result(self):
        metric = compute_eer(self.bonafide_scores, self.spoof_scores)[0]

        self.bonafide_scores = np.array([])
        self.spoof_scores = np.array([])

        return metric
