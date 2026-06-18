import os
import sys
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
import json
from config import *
from utils import *
from dataset import GenomeInferenceDataset
from model import GlobalCentFormerInterpret


CHECKPOINT = sys.argv[1]
OUTDIR = sys.argv[2]

model = GlobalCentFormerInterpret().to(DEVICE)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
state_dict = {}
for key, value in checkpoint.items():
    if key.startswith('encoder.'):
        state_dict[key[8:]] = value  # Remove 'encoder.'
    else:
        state_dict[key] = value

# Load the fixed state dict
missing, unexpected = model.load_state_dict(state_dict, strict=False)

# model.load_state_dict(
#     torch.load(CHECKPOINT, map_location=DEVICE)
# )
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
        
        # Main output directory
        outdir = f"attention/{OUTDIR}/{SAMPLE}/{CHROM}"
        os.makedirs(outdir, exist_ok=True)
        
        # Storage
        positions = []
        layer_attentions = defaultdict(list)
        
        with torch.no_grad():
            for emb, start_token in loader:
                emb = emb.to(DEVICE)
                
                pred, attentions = model(
                    emb,
                    return_attention=True,
                )
                
                pos = start_token.item()
                positions.append(pos)
                
                for layer_id, attn in enumerate(attentions):
                    # attn: [1, 256, 256] -> [256, 256]
                    attn_matrix = attn.squeeze(0).cpu().numpy()
                    layer_attentions[layer_id].append(attn_matrix)
        
        # Sort everything by position
        sorted_idx = np.argsort(positions)
        sorted_positions = np.array(positions)[sorted_idx]
        
        # Save layer-wise data
        for layer_id, attn_list in layer_attentions.items():
            # Stack and sort
            all_attn = np.stack(attn_list)[sorted_idx]  # [num_windows, 256, 256]
            
            # Save as compressed numpy
            np.savez_compressed(
                f"{outdir}/layer{layer_id}_all.npz",
                attentions=all_attn.astype(np.float16),  # Save as float16 to save space
                positions=sorted_positions,
                shape=(256, 256),
                layer_id=layer_id
            )
            
            # # Also save individual files if needed (optional)
            # for i, pos in enumerate(sorted_positions):
            #     np.save(
            #         f"{outdir}/pos{pos:010d}_layer{layer_id}.npy",
            #         all_attn[i]  # [256, 256]
            #     )
        
        # Save metadata
        metadata = { 
            'sample': SAMPLE,        
            'chromosome': CHROM,
            'window_size': 256,
            'stride': 200,
            'num_windows': len(sorted_positions),
            'positions': sorted_positions.tolist(), 
            'position_range': [
                int(sorted_positions.min()),  
                int(sorted_positions.max())   
            ],
            'num_layers': len(layer_attentions),
            'd_model': 768,
        }
        
        with open(f"{outdir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ {SAMPLE} {CHROM}: {len(sorted_positions)} windows saved")