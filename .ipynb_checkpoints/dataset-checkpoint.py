import numpy as np
import torch
import pyBigWig

from torch.utils.data import Dataset

from config import *


class GenomeEmbeddingDataset(Dataset):

    def __init__(
        self,
        chrom_sizes,
        tokenstep,
        mmap_dir,
        bw_path
    ):

        self.bw = pyBigWig.open(bw_path)

        self.embeddings = {}

        for chrom, n_tokens in chrom_sizes.items():

            self.embeddings[chrom] = np.memmap(
                f"{mmap_dir}/{chrom}.fp16.mmap",
                mode="r",
                dtype=np.float16,
                shape=(n_tokens, D_MODEL)
            )

        self.coords = []

        for chrom, n_tokens in chrom_sizes.items():

            for i in range(
                0,
                n_tokens - REGION_TOKENS
            ):

                self.coords.append((chrom, i))

    def __len__(self):

        return len(self.coords)

    def __getitem__(self, idx):

        chrom, start_token = self.coords[idx]

        end_token = start_token + REGION_TOKENS

        emb = self.embeddings[chrom][
            start_token:end_token
        ]

        emb = torch.tensor(
            emb,
            dtype=torch.float32
        )

        start_bp = start_token * WINDOW_BP
        end_bp = end_token * WINDOW_BP

        vals = self.bw.values(
            chrom,
            start_bp,
            end_bp,
            numpy=True
        )

        vals = np.nan_to_num(vals)

        label = vals.reshape(
            -1,
            WINDOW_BP
        ).mean(axis=1)

        #label = np.log1p(label)

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return emb, label