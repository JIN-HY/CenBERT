import os
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, random_split, ConcatDataset

import numpy as np

from config import *
from utils import *

from dataset import GenomeEmbeddingDataset
from model import GlobalCentFormer


def main():

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )
    
    fa_dict, bw_dict = bw_map(BW_MAP)
    samples = fa_dict.keys()

    chroms_all = [f"Chr{i:02d}" for i in range(1, 13)]
    if len(sys.argv) >= 2: 
        val_chroms = sys.argv[1:]
        train_chroms = {chrom for chrom in chroms_all if chrom not in val_chroms}
    else:
        train_chroms = chroms_all

    train_datasets = []
    val_datasets = []
    for SAMPLE in samples:
        genome = load_genome(fa_dict[SAMPLE])
        chrom_sizes = get_chrom_sizes(
            genome,
            WINDOW_BP
        )

        train_chrom_sizes = {f"{SAMPLE}.{CHROM}":chrom_sizes[f"{SAMPLE}.{CHROM}"] for CHROM in train_chroms}
        val_chrom_sizes = {f"{SAMPLE}.{CHROM}":chrom_sizes[f"{SAMPLE}.{CHROM}"] for CHROM in val_chroms}

        train_dt = GenomeEmbeddingDataset(
            chrom_sizes=train_chrom_sizes,
            tokenstep=TOKEN_STEP,
            mmap_dir=EMBEDDING_DIR,
            bw_path=bw_dict[SAMPLE]
        )
        val_dt = GenomeEmbeddingDataset(
            chrom_sizes=val_chrom_sizes,
            tokenstep=TOKEN_STEP,
            mmap_dir=EMBEDDING_DIR,
            bw_path=bw_dict[SAMPLE]
        )

        train_datasets.append(train_dt)
        val_datasets.append(val_dt)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = GlobalCentFormer().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    criterion = nn.SmoothL1Loss()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for emb, label in train_loader:

            emb = emb.to(
                DEVICE,
                non_blocking=True
            )

            label = label.to(
                DEVICE,
                non_blocking=True
            )

            pred = model(emb)

            loss = criterion(
                pred,
                label
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        mean_loss = (
            total_loss / len(train_loader)
        )

        model.eval()
        with torch.no_grad():
            val_loss = 0
            
            for emb, label in val_loader:

                emb = emb.to(
                    DEVICE,
                    non_blocking=True
                )

                label = label.to(
                    DEVICE,
                    non_blocking=True
                )

                pred = model(emb)

                loss = criterion(
                    pred,
                    label
                )

                val_loss += loss.item()

            mean_val_loss = (
                val_loss / len(val_loader)
            
            )
        print(
            f"Epoch {epoch+1} "
            f"| Train Loss: {mean_loss:.4f} | Val Loss: {mean_val_loss:.4f} "
        )

        # torch.save(
        #     model.state_dict(),
        #     f"{CHECKPOINT_DIR}/epoch{epoch+1}.pt"
        # )
    torch.save(
        model.state_dict(),
        f"{CHECKPOINT_DIR}/CenBERT_chrom_0521.pt"
    )

if __name__ == "__main__":
    main()