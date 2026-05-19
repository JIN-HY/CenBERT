
import sys
import numpy as np
import torch
import pandas as pd

from Bio import SeqIO

from config import *
from utils import *
from model import GlobalCentFormer
from dataset import GenomeInferenceDataset


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

dataset = GenomeInferenceDataset(
    chrom_sizes=chrom_sizes,
    tokenstep=TOKEN_STEP,
    mmap_dir=EMBEDDING_DIR,
    chrom=CHROM
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
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
# PREDICTION ARRAYS
# =========================================================

pred_sum = np.zeros(n_tokens)
pred_count = np.zeros(n_tokens)


# =========================================================
# INFERENCE
# =========================================================

with torch.no_grad():

    for emb, start_token in loader:

        emb = emb.to(DEVICE)

        pred = model(emb)

        pred = pred.cpu().numpy()

        pred = np.expm1(pred)

        for i in range(len(start_token)):

            s = start_token[i].item()

            e = s + REGION_TOKENS

            pred_sum[s:e] += pred[i]

            pred_count[s:e] += 1


# =========================================================
# FINALIZE
# =========================================================

final_pred = pred_sum / np.maximum(
    pred_count,
    1
)

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
