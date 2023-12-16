import logging
from pathlib import Path

from anti_spoof.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, *args, **kwargs):
        index = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["audio_path"] = str(path)
            if entry:
                index.append(entry)
        super().__init__(index, *args, **kwargs)
