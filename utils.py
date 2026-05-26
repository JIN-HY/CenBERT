from Bio import SeqIO


def load_genome(fasta_path):

    genome = SeqIO.to_dict(
        SeqIO.parse(fasta_path, "fasta")
    )

    return genome


def get_chrom_sizes(genome, window_bp):

    chrom_sizes = {}

    for chrom in genome:

        chrom_sizes[chrom] = (
            len(genome[chrom]) // window_bp
        )

    return chrom_sizes

def bw_map(mapfile):
    fa_dict = {}
    bw_dict = {}
    with open(mapfile) as f:
        for l in f:
            line = l.strip().split(' ')
            fa_dict[line[0]] = line[1]
            bw_dict[line[0]] = line[2]
    return fa_dict, bw_dict



import numpy as np

from scipy.stats import (
    pearsonr,
    spearmanr
)

from scipy.signal import correlate

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score
)

# from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def regression_metrics(
    pred,
    truth
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

    return {

        "pearson_r":
            pearsonr(
                pred,
                truth
            )[0],

        "spearman_r":
            spearmanr(
                pred,
                truth
            )[0],

        "mse":
            mean_squared_error(
                truth,
                pred
            ),

        "mae":
            mean_absolute_error(
                truth,
                pred
            )
    }


def peak_metrics(
    pred,
    truth,
    bin_size_bp=512
):

    pred = np.asarray(pred)
    truth = np.asarray(truth)

    pred_peak_idx = np.argmax(pred)
    truth_peak_idx = np.argmax(truth)

    peak_shift_bins = (
        pred_peak_idx
        -
        truth_peak_idx
    )

    peak_shift_bp = (
        peak_shift_bins
        *
        bin_size_bp
    )

    return {

        "pred_peak_idx":
            int(pred_peak_idx),

        "truth_peak_idx":
            int(truth_peak_idx),

        "peak_shift_bins":
            int(peak_shift_bins),

        "peak_shift_bp":
            int(peak_shift_bp)
    }


def alignment_metrics(
    pred,
    truth
):

    pred = np.asarray(pred)
    truth = np.asarray(truth)

    pred_centered = (
        pred - pred.mean()
    )

    truth_centered = (
        truth - truth.mean()
    )

    corr = correlate(
        pred_centered,
        truth_centered,
        mode="full"
    )

    lags = np.arange(
        -len(pred) + 1,
        len(pred)
    )

    best_idx = np.argmax(corr)

    best_lag = lags[best_idx]

    aligned_pred = np.roll(
        pred,
        -best_lag
    )

    aligned_r = pearsonr(
        aligned_pred,
        truth
    )[0]

    return {

        "best_lag_bins":
            int(best_lag),

        "aligned_pearson_r":
            float(aligned_r),

        "aligned_prediction":
            aligned_pred
    }


def overlap_metrics(
    pred,
    truth,
    threshold=1
):

    pred = np.asarray(pred)
    truth = np.asarray(truth)

    pred_mask = pred > threshold
    truth_mask = truth > threshold

    intersection = np.sum(
        pred_mask
        &
        truth_mask
    )

    union = np.sum(
        pred_mask
        |
        truth_mask
    )

    iou = (
        intersection / union
        if union > 0
        else np.nan
    )

    dice = (
        2 * intersection
        /
        (
            pred_mask.sum()
            +
            truth_mask.sum()
        )
        if (
            pred_mask.sum()
            +
            truth_mask.sum()
        ) > 0
        else np.nan
    )

    return {

        "iou":
            float(iou),

        "dice":
            float(dice)
    }


def classification_metrics(
    pred,
    truth,
    threshold=1
):

    pred = np.asarray(pred)
    truth = np.asarray(truth)

    truth_binary = (
        truth > threshold
    ).astype(int)

    return {

        "auroc":
            roc_auc_score(
                truth_binary,
                pred
            ),

        "average_precision":
            average_precision_score(
                truth_binary,
                pred
            )
    }


def dtw_metrics(
    pred,
    truth
):

    pred = np.asarray(pred)
    truth = np.asarray(truth)

    distance, path = fastdtw(
        pred,
        truth,
        dist=euclidean
    )

    return {

        "dtw_distance":
            float(distance),

        "dtw_normalized_distance":
            float(
                distance / len(pred)
            ),

        "dtw_path_length":
            int(len(path))
    }


