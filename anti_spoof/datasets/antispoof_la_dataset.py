import json
import logging
import os
import shutil
from pathlib import Path

from speechbrain.utils.data_utils import download_file

from anti_spoof.base.base_dataset import BaseDataset
from anti_spoof.utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "LA": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
}


class ASV2019AntispoofDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ASV2019Antispoof"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir

        index = self._get_or_load_index(part)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        if not (Path(self._data_dir) / 'downloaded').exists():
            self._load_dataset()

        labels_file = Path(self._data_dir) / 'ASVspoof2019_LA_cm_protocols' / f'ASVspoof2019.LA.cm.{part}.{"trn" if part == "train" else "trl"}.txt'
        with labels_file.open() as f:
            for line in f:
                _, file, _, _, target = line.strip().split()
                audio_path = Path(self._data_dir) / f'ASVspoof2019_LA_{part}/flac/{file}.flac'
                index.append(
                    {
                        "audio_path": str(audio_path.absolute().resolve()),
                        "target": target,
                    }
                )

        return index

    def _load_dataset(self):
        arch_path = self._data_dir / "LA.zip"
        print(f"Loading ASV2019Antispoof LA...")
        download_file(URL_LINKS["LA"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LA").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LA"))

        (Path(self._data_dir) / 'downloaded').touch()
