from pathlib import Path
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


DIR = os.path.dirname(os.path.realpath(__file__))
BOS_INDEX = 3
PAD_INDEX = 1
EOS_INDEX = 0


class FCDataset(Dataset):
    def __init__(self, dataset_dir: str, tokenizer: Tokenizer, max_length=120):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.tokenizer.enable_truncation(max_length=max_length)
        self.ntokens = tokenizer.get_vocab_size()

        self.samples = []

        print("🔥 files!")
        src_files = Path(dataset_dir).glob("*.npy")
        self.samples = list(src_files)
        print("🔥 done!")

    def _pad(self, sample: list):
        while len(sample) < self.max_length:
            sample.append(PAD_INDEX)

        return sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        data = np.load(self.samples[i])

        if data.shape[0] < self.max_length:
            data = np.hstack(
                (data, [PAD_INDEX for _ in range(self.max_length - data.shape[0])]))

        return torch.tensor(data)
