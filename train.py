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

    if len(sys.argv) == 2: 
        samples = [sys.argv[1]]
    else:
        samples = fa_dict.keys()

    datasets = []
    for SAMPLE in samples:
        genome = load_genome(fa_dict[SAMPLE])
        chrom_sizes = get_chrom_sizes(
            genome,
            WINDOW_BP
        )

        dt = GenomeEmbeddingDataset(
            chrom_sizes=chrom_sizes,
            tokenstep=TOKEN_STEP,
            mmap_dir=EMBEDDING_DIR,
            bw_path=bw_dict[SAMPLE]
        )

        datasets.append(dt)

    dataset = ConcatDataset(datasets)

    train_size = int(0.8 * len(dataset))
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
        f"{CHECKPOINT_DIR}/CenBERT_all_0519.pt"
    )

if __name__ == "__main__":
    main()