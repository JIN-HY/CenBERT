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
                n_tokens - REGION_TOKENS, tokenstep
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




class GenomeInferenceDataset(Dataset):

    def __init__(
        self,
        chrom,
        chrom_sizes,
        tokenstep,
        mmap_dir,
    ):

        self.chrom = chrom

        self.n_tokens = chrom_sizes[chrom]

        self.embeddings = np.memmap(
            f"{mmap_dir}/{chrom}.fp16.mmap",
            mode="r",
            dtype=np.float16,
            shape=(self.n_tokens, D_MODEL)
        )

        self.coords = []

        for i in range(
            0,
            self.n_tokens - REGION_TOKENS,
            tokenstep
        ):

            self.coords.append(i)

        # self.pred_sum = np.zeros(
        #     self.n_tokens,
        #     dtype=np.float32
        # )

        # self.pred_count = np.zeros(
        #     self.n_tokens,
        #     dtype=np.float32
        # )

        self.all_preds = [
            [] for _ in range(self.n_tokens)
        ]

    def __len__(self):

        return len(self.coords)

    def __getitem__(self, idx):

        start_token = self.coords[idx]

        end_token = (
            start_token + REGION_TOKENS
        )

        emb = self.embeddings[
            start_token:end_token
        ]

        emb = torch.tensor(
            emb,
            dtype=torch.float32
        )

        return emb, start_token

    def add_prediction(self, start_token, pred):

        end_token = (
            start_token + REGION_TOKENS
        )

        for i in range(REGION_TOKENS):

            self.all_preds[
                start_token + i
            ].append(float(pred[i]))

    def get_mean_predictions(self):

        means = []

        for preds in self.all_preds:

            if len(preds) == 0:
                means.append(np.nan)
            else:
                means.append(np.mean(preds))

        return np.array(means)

    def get_variance_predictions(self):

        vars_ = []

        for preds in self.all_preds:

            if len(preds) <= 1:
                vars_.append(0)
            else:
                vars_.append(np.var(preds))

        return np.array(vars_)
        
    def get_predictions(self):

        return (
            self.pred_sum /
            np.maximum(
                self.pred_count,
                1
            )
        )

    def get_counts(self):

        return np.array([
            len(x)
            for x in self.all_preds
        ])