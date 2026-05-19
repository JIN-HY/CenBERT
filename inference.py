
import sys
import numpy as np
import torch
import pandas as pd

from Bio import SeqIO

from config import *
from utils import *
from model import GlobalCentFormer


# =========================================================
# ARGUMENTS
# =========================================================

CHECKP
SAMPLE = sys.argv[2]CHROM = sys.argv[2]

OUTFILE = (
    sys.argv[3]
    if len(sysSAMPLE) > 3
    else f"{CHROM}.predictions.tsv"
)


# =========================================================
# LOAD GENOME
# =========================================================

genome = load_genome(GENOME_PATH)

chrom_len = len(genome[CHROM])

fa_dict, bw_dict = bw_map(BW_MAP)

if len(sys.argv) > 2: 
    samples = [sys.argv[2]]
else:
    samples = fa_dict.keys()

    for SAMPLE in samples:
        genome = load_genome(fa_dict[SAMPLE])
        chrom_sizes = get_chrom_sizes(
            genome,
            WINDOW_BP
        )


n_tokens = chrom_len // WINDOW_BP


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

embeddings = np.memmap(
    f"{EMBEDDING_DIR}/{CHROM}.fp16.mmap",
    mode="r",
    dtype=np.float16,
    shape=(n_tokens, D_MODEL)
)


# =========================================================
# LOAD MODEL
# =========================================================

model = GlobalCentFormer().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.eval()


# =========================================================
# INFERENCE
# =========================================================

pred_sum = np.zeros(n_tokens)
pred_count = np.zeros(n_tokens)

stride = 100

with torch.no_grad():

    for start in range(
        0,
        n_tokens - REGION_TOKENS,
        stride
    ):

        end = start + REGION_TOKENS

        x = embeddings[start:end]

        x = torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        pred = model(x)

        pred = pred.squeeze(0)

        pred = pred.cpu().numpy()

        # reverse log1p transform
        pred = np.expm1(pred)

        pred_sum[start:end] += pred
        pred_count[start:end] += 1


# =========================================================
# AVERAGE OVERLAPS
# =========================================================

final_pred = pred_sum / np.maximum(pred_count, 1)


# =========================================================
# SAVE
# =========================================================

starts = np.arange(n_tokens) * WINDOW_BP
ends = starts + WINDOW_BP

df = pd.DataFrame({
    "chrom": CHROM,
    "start": starts,
    "end": ends,
    "prediction": final_pred
})

df.to_csv(
    OUTFILE,
    sep="\t",
    index=False
)

print(df.head())

print(f"\nSaved: {OUTFILE}")