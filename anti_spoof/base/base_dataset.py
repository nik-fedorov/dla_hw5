import logging
from typing import List

import torch
import torchaudio
from torch.utils.data import Dataset

from anti_spoof.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            audio_len
    ):
        self.config_parser = config_parser
        self.audio_len = audio_len
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["audio_path"]
        audio_wave = self.load_audio(audio_path)
        audio_wave = self.cut_or_concat_audio(audio_wave)

        sample = {
            "audio_path": audio_path,
            "audio": audio_wave,
        }

        if 'target' in data_dict:
            sample['target'] = 1 if data_dict['target'] == 'bonafide' else 0

        return sample

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        '''
        Load audio from path, resample it if needed
        :return: 1st channel of audio (tensor of shape 1xL)
        '''
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def cut_or_concat_audio(self, audio):
        if audio.size(1) >= self.audio_len:
            return audio[:, :self.audio_len]

        n_repeats = self.audio_len // audio.size(1) + 1
        return audio.repeat(1, n_repeats)[:, :self.audio_len]
