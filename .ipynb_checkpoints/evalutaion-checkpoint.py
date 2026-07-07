from utils import *
import os
import sys
import numpy as np
import pandas as pd
import pyBigWig

def evaluate_prediction(
    pred,
    truth,
    bin_size_bp=512,
    threshold=1
):

    pred = np.asarray(pred)
    truth = np.asarray(truth)

    mask = (
        ~np.isnan(pred)
        &
        ~np.isnan(truth)
    )

    pred = pred[mask]
    truth = truth[mask]
    
    results = {}

    results.update(
        regression_metrics(
            pred,
            truth
        )
    )

    results.update(
        peak_metrics(
            pred,
            truth,
            bin_size_bp
        )
    )

    results.update(
        alignment_metrics(
            pred,
            truth
        )
    )

    results.update(
        overlap_metrics(
            pred,
            truth,
            threshold
        )
    )

    results.update(
        classification_metrics(
            pred,
            truth,
            threshold
        )
    )

    # results.update(
    #     dtw_metrics(
    #         pred,
    #         truth
    #     )
    # )

    return results

sample = sys.argv[1]
chrom = sys.argv[2]
schrom = f"{sample}.{chrom}" #
holdout_type = sys.argv[3]
PRED_TSV = f"prediction/{holdout_type}/{schrom}.tsv"
BW_MAP = "sample-bw.txt"

fa_dict, bw_dict = bw_map(BW_MAP)
BW_FILE = bw_dict[sample]
df = pd.read_csv(
    PRED_TSV,
    sep="\t"
)

bw = pyBigWig.open(BW_FILE)

truth = bw.values(
    schrom,
    0,
    int(df["end"].iloc[-1]),
    numpy=True
)

bw.close()

truth_bin = [] 
for start, end in zip(df.start, df.end): 
    vals = truth[start:end] 
    truth_bin.append( np.nanmean(vals) )
    
metrics = evaluate_prediction(
    pred=df["prediction_mean"].values,
    truth=truth_bin
)

print(metrics)

metrics["sample"] = sample
metrics["chrom"] = chrom
metrics["holdout_type"] = holdout_type

# remove large arrays
metrics.pop("aligned_prediction", None)

df_metrics = pd.DataFrame([metrics])

df_metrics.to_csv(
    "evaluation_metrics.csv",
    mode="a",
    header=not os.path.exists("evaluation_metrics.csv"),
    index=False
)