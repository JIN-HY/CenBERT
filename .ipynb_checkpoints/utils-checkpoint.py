from Bio import SeqIO
import os

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

from fastdtw import fastdtw


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
        dist=lambda x, y: abs(x - y)
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
    

def load_chromosome_attentions(sample, chrom, layer_id=0, outdir="attention"):
    """
    Load attention matrices for a specific chromosome and layer.
    
    Args:
        sample: Sample name
        chrom: Chromosome name
        layer_id: Which transformer layer (0-indexed)
        outdir: Base output directory
    
    Returns:
        attentions: [num_windows, 256, 256] - attention matrices
        positions: [num_windows] - genomic positions
    """
    filepath = f"{outdir}/{sample}/{chrom}/layer{layer_id}_all.npz"
    
    if os.path.exists(filepath):
        data = np.load(filepath)
        return data['attentions'], data['positions']
    else:
        # Fallback: load individual files
        pattern = f"{outdir}/{sample}/{chrom}/pos*_layer{layer_id}.npy"
        import glob
        files = sorted(glob.glob(pattern))
        
        attentions = []
        positions = []
        for f in files:
            pos = int(f.split('pos')[1].split('_')[0])
            positions.append(pos)
            attentions.append(np.load(f))
        
        # Sort by position
        sorted_idx = np.argsort(positions)
        attentions = np.array(attentions)[sorted_idx]
        positions = np.array(positions)[sorted_idx]
        
        return attentions, positions

# # Example usage
# attentions, positions = load_chromosome_attentions('sample1', 'chr1', layer_id=0)
# print(f"Loaded {len(attentions)} windows from chr1")
# print(f"Positions: {positions[:5]} ... {positions[-5:]}")


def visualize_attention_track(sample, chrom, layer_id=0, start_pos=None, end_pos=None):
    """
    Visualize attention across genomic positions.
    """
    attentions, positions = load_chromosome_attentions(sample, chrom, layer_id)
    
    # Filter by genomic range if specified
    if start_pos is not None and end_pos is not None:
        mask = (positions >= start_pos) & (positions <= end_pos)
        attentions = attentions[mask]
        positions = positions[mask]
    
    # Compute statistics per window
    mean_attn = attentions.mean(axis=(1, 2))  # [num_windows]
    max_attn = attentions.max(axis=(1, 2))    # [num_windows]
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot mean attention
    axes[0].plot(positions, mean_attn)
    axes[0].set_title(f'{sample} {chrom} Layer {layer_id} - Mean Attention')
    axes[0].set_ylabel('Mean Attention')
    
    # Plot max attention
    axes[1].plot(positions, max_attn)
    axes[1].set_title('Max Attention')
    axes[1].set_ylabel('Max Attention')
    
    # Heatmap of attention for a specific window (e.g., middle)
    mid_idx = len(attentions) // 2
    im = axes[2].imshow(attentions[mid_idx], cmap='viridis', aspect='auto')
    axes[2].set_title(f'Attention Matrix at Position {positions[mid_idx]}')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f"{sample}_{chrom}_layer{layer_id}_attention.png", dpi=300)
    plt.show()


def compare_layers(sample, chrom, position, num_layers=4, base_dir="attention"):
    """Compare attention across layers at a specific position."""
    # Find closest position
    positions_all = None
    for layer_id in range(num_layers):
        _, positions_all = load_attentions(sample, chrom, layer_id, base_dir)
        if positions_all is not None:
            break
    
    if positions_all is None:
        return
    
    # Find closest position
    idx = np.argmin(np.abs(positions_all - position))
    pos_actual = positions_all[idx]
    print(f"Showing position {pos_actual} (requested {position})")
    
    fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    
    for layer_id in range(num_layers):
        attentions, _ = load_attentions(sample, chrom, layer_id, base_dir)
        if attentions is not None and idx < len(attentions):
            im = axes[layer_id].imshow(attentions[idx], cmap='viridis', aspect='auto')
            axes[layer_id].set_title(f'Layer {layer_id}')
            axes[layer_id].set_xlabel('Key')
            axes[layer_id].set_ylabel('Query')
            plt.colorbar(im, ax=axes[layer_id])
    
    plt.suptitle(f'{sample} {chrom} Position {pos_actual}')
    plt.tight_layout()
    plt.savefig(f"{sample}_{chrom}_pos{pos_actual}_layers.png", dpi=150)
    plt.show()


from pathlib import Path

def load_attentions(sample, chrom, layer_id=0, base_dir="attention"):
    """Load attention matrices and positions from compressed npz file."""
    filepath = f"{base_dir}/{sample}/{chrom}/layer{layer_id}_all.npz"
    
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        return None, None
    
    data = np.load(filepath)
    return data['attentions'], data['positions']


def plot_heatmap(attn_matrix, save=None):
    """Plot single attention heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Weight')
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


def plot_track(sample, chrom, layer_id=0, base_dir="attention", start=None, end=None):
    """Plot attention statistics across genomic positions."""
    attentions, positions = load_attentions(sample, chrom, layer_id, base_dir)
    if attentions is None:
        return
    
    # Filter by position
    if start is not None and end is not None:
        mask = (positions >= start) & (positions <= end)
        attentions = attentions[mask]
        positions = positions[mask]
    
    if len(attentions) == 0:
        print("No windows in range")
        return
    
    # Compute statistics
    mean_attn = attentions.mean(axis=(1, 2))
    max_attn = attentions.max(axis=(1, 2))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(positions, mean_attn, 'b-', linewidth=1)
    axes[0].set_ylabel('Mean Attention')
    axes[0].set_title(f'{sample} {chrom} Layer {layer_id}')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(positions, max_attn, 'r-', linewidth=1)
    axes[1].set_ylabel('Max Attention')
    axes[1].set_xlabel('Genomic Position')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{sample}_{chrom}_layer{layer_id}_track.png", dpi=150)
    plt.show()


def plot_average_attention(sample, chrom, num_layers=4, base_dir="attention"):
    """Plot average attention pattern across all positions."""
    fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    
    for layer_id in range(num_layers):
        attentions, _ = load_attentions(sample, chrom, layer_id, base_dir)
        if attentions is not None:
            avg_attn = attentions.mean(axis=0)
            im = axes[layer_id].imshow(avg_attn, cmap='viridis', aspect='auto')
            axes[layer_id].set_title(f'Layer {layer_id} (avg)')
            axes[layer_id].set_xlabel('Key')
            axes[layer_id].set_ylabel('Query')
            plt.colorbar(im, ax=axes[layer_id])
    
    plt.suptitle(f'{sample} {chrom} - Average Attention Across All Positions')
    plt.tight_layout()
    plt.savefig(f"{sample}_{chrom}_average_attention.png", dpi=150)
    plt.show()