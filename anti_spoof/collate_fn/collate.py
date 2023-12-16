import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = dict()

    result_batch["audio"] = torch.stack([item["audio"] for item in dataset_items], dim=0)
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    if 'target' in dataset_items[0]:
        result_batch['target'] = torch.tensor([item["target"] for item in dataset_items])

    return result_batch
