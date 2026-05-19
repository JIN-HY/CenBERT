import os
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, random_split

import numpy as np

from config import *
from utils import *

from dataset import GenomeEmbeddingDataset
from model import GlobalCentFormer

SAMPLE = sys.argv[1]

def main():

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )
    
    fa_dict, bw_dict = bw_map(BW_MAP)

    genome = load_genome(fa_dict[SAMPLE])

    chrom_sizes = get_chrom_sizes(
        genome,
        WINDOW_BP
    )

    dataset = GenomeEmbeddingDataset(
        chrom_sizes=chrom_sizes,
        tokenstep=TOKEN_STEP,
        mmap_dir=EMBEDDING_DIR,
        bw_path=bw_dict[SAMPLE]
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

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

        for emb, label in loader:

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
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch+1} "
            f"| Loss: {mean_loss:.5f}"
        )

        # torch.save(
        #     model.state_dict(),
        #     f"{CHECKPOINT_DIR}/epoch{epoch+1}.pt"
        # )
    torch.save(
        model.state_dict(),
        f"{CHECKPOINT_DIR}/{SAMPLE}.pt"
    )

if __name__ == "__main__":
    main()