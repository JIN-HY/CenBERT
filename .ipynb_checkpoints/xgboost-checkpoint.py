import numpy as np
import pyBigWig
from xgboost import XGBRegressor
from config import *
from utils import *


def load_targets(
    bw_path,
    chrom_sizes, 
    window_bp
):

    targets = []

    bw = pyBigWig.open(bw_path)

    for chrom, length in chrom_sizes.items():

        for start in range(
            0,
            length,
            window_bp
        ):

            end = min(
                start + window_bp,
                length
            )

            vals = bw.values(
                chrom,
                start,
                end,
                numpy=True
            )

            targets.append(
                np.nanmean(vals)
            )

    bw.close()

    return np.array(
        targets,
        dtype=np.float16
    )



def main():
    
    fa_dict, bw_dict = bw_map(BW_MAP)
    samples = fa_dict.keys()

    if len(sys.argv) >= 3: # exclude sample
        val_samples = sys.argv[1:-1]
    else:
        train_samples = {sp for sp in fa_dict.keys() if sp not in val_samples}

    all_X = []
    all_y = []
    
    for SAMPLE in samples:
        genome = load_genome(fa_dict[SAMPLE])
        chrom_sizes = get_chrom_sizes(
            genome,
            WINDOW_BP
        )
        for chrom, n_tokens in chrom_sizes.items():

            x = np.memmap(
                f"{mmap_dir}/{chrom}.fp16.mmap",
                mode="r",
                dtype=np.float16,
                shape=(n_tokens, D_MODEL)
            )
            all_X.append(x)

        y = load_targets(
            bw_path=bw_dict[SAMPLE],
            chrom_sizes=chrom_sizes,
            window_bp=WINDOW_BP
        )

       all_y.append(y)

    X = np.concatenate(
        all_X,
        axis=0
    )

    y = np.concatenate(
        all_y,
        axis=0
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(
        X,
        y
    )


