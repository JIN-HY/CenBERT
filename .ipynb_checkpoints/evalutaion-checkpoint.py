from utils import *
import numpy as np
import pandas as pd
import pyBigWig

def evaluate_prediction(
    pred,
    truth,
    bin_size_bp=512,
    threshold=0.5
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

chrom = "S001.Chr01" #
PRED_TSV = f"prediction/hold_sample/{chrom}.tsv"
BW_MAP = "sample-bw.txt"

fa_dict, bw_dict = bw_map(BW_MAP)
BW_FILE = bw_dict[chrom.split('.')[0]]
df = pd.read_csv(
    PRED_TSV,
    sep="\t"
)

bw = pyBigWig.open(BW_FILE)

truth = bw.values(
    chrom,
    0,
    int(df["end"].iloc[-1]),
    numpy=True
)

bw.close()

truth_bin = [] 
for start, end in zip(df.start, df.end): 
    vals = truth[start:end] 
    truth_bin.append( np.nanmean(vals) )
    
results = evaluate_prediction(
    pred=df["prediction_mean"].values,
    truth=truth_bin
)

print(results)
