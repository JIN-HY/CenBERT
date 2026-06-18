import os
import sys
import numpy as np
import torch

from torch.utils.data import DataLoader

from config import *
from utils import *
from dataset import GenomeInferenceDataset
from model import GlobalCentFormer


CHECKPOINT = sys.argv[1]
OUTDIR = sys.argv[2]

model = GlobalCentFormer().to(DEVICE)
model.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE)
)
model.eval()

fa_dict, bw_dict = bw_map(BW_MAP)

samples = sys.argv[3:] if len(sys.argv) > 3 else fa_dict.keys()

for SAMPLE in samples:

    genome = load_genome(fa_dict[SAMPLE])
    chrom_sizes = get_chrom_sizes(genome, WINDOW_BP)

    for CHROM in chrom_sizes:

        dataset = GenomeInferenceDataset(
            chrom=CHROM,
            chrom_sizes=chrom_sizes,
            tokenstep=TOKEN_STEP,
            mmap_dir=EMBEDDING_DIR,
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        saliency_list = []

        for emb, start_token in loader:

            emb = emb.to(DEVICE)
            emb.requires_grad_(True)

            pred = model(emb)

            #
            # IMPORTANT:
            # choose ONE output location
            #

            center_idx = pred.shape[1] // 2

            score = pred[:, center_idx].sum()

            model.zero_grad()

            score.backward()

            grad = emb.grad

            saliency = (
                grad * emb
            ).abs().sum(dim=-1)

            saliency_list.append(
                saliency.squeeze(0).detach().cpu().numpy()
            )

        saliency_array = np.stack(saliency_list)

        os.makedirs(
            f"saliency/{OUTDIR}",
            exist_ok=True,
        )

        np.save(
            f"saliency/{OUTDIR}/{CHROM}.npy",
            saliency_array,
        )