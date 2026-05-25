from utils import *
def evaluate_prediction(
    pred,
    truth,
    bin_size_bp=512,
    threshold=0.5
):

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

    results.update(
        dtw_metrics(
            pred,
            truth
        )
    )

    return results

chrom = "S001.Chr01" #
PRED_TSV = f"hold_chrom/{chrom}.tsv"
BW_MAP = "../sample-bw.txt"
def bw_map(mapfile):
    fa_dict = {}
    bw_dict = {}
    with open(mapfile) as f:
        for l in f:
            line = l.strip().split(' ')
            fa_dict[line[0]] = line[1]
            bw_dict[line[0]] = line[2]
    return fa_dict, bw_dict

fa_dict, bw_dict = bw_map(BW_MAP)
BW_FILE = bw_dict[chrom.split('.')[0]]
df = pd.read_csv(
    PRED_TSV,
    sep="\t"
)

bw = pyBigWig.open(f"../{BW_FILE}")

truth = bw.values(
    chrom,
    0,
    int(df["end"].iloc[-1]),
    numpy=True
)

bw.close()

results = evaluate_prediction(
    pred=df["prediction_mean"].values,
    truth=truth_bin
)

print(results)
