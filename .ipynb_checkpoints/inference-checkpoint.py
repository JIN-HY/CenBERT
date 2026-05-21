
import sys
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader, random_split, ConcatDataset
from config import *
from utils import *
from model import GlobalCentFormer
from dataset import GenomeInferenceDataset


# =========================================================
# ARGUMENTS
# =========================================================

CHECKPOINT = sys.argv[1]


model = GlobalCentFormer().to(DEVICE)
model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)
model.eval()

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

    for CHROM in chrom_sizes:
        dataset = GenomeInferenceDataset(
            chrom = CHROM,
            chrom_sizes=chrom_sizes,
            tokenstep=TOKEN_STEP,
            mmap_dir=EMBEDDING_DIR
        )

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0
        )


        with torch.no_grad():

            for emb, start_token in loader:

                emb = emb.to(DEVICE)

                pred = model(emb)

                pred = pred.cpu().numpy()

                #pred = np.expm1(pred)

                for i in range(len(start_token)):

                    dataset.add_prediction(
                        start_token[i].item(),
                        pred[i]
                    )

        # final_pred = dataset.get_predictions()

        mean_pred = dataset.get_mean_predictions()

        var_pred = dataset.get_variance_predictions()

        counts = dataset.get_counts()
        
        starts = (
            np.arange(dataset.n_tokens)
            * WINDOW_BP
        )

        ends = starts + WINDOW_BP

        df = pd.DataFrame({
            "chrom": CHROM,
            "start": starts,
            "end": ends,
            # "prediction": final_pred,
            "prediction_mean": mean_pred,
            "prediction_var": var_pred,
            "n_overlaps": counts
            })

        outfile = f"prediction/{CHROM}.tsv"

        df.to_csv(
            outfile,
            sep="\t",
            index=False
        )

        print(f"\nSaved: {outfile}")
