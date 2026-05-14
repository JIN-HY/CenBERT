import os
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel

from config import *
from utils import *


DEVICE = DEVICE

tokenizer = AutoTokenizer.from_pretrained(
    DNABERT2_NAME,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    DNABERT2_NAME,
    trust_remote_code=True
).to(DEVICE)

model.eval()


def chunk_sequence(seq):

    chunks = []

    for i in range(0, len(seq), WINDOW_BP):

        chunk = seq[i:i+WINDOW_BP]

        if len(chunk) == WINDOW_BP:
            chunks.append(chunk)

    return chunks


def encode_chunks(chunks, batch_size=64):

    all_embeddings = []

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i+batch_size]

        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        tokens = {
            k: v.to(DEVICE)
            for k, v in tokens.items()
        }

        with torch.no_grad():

            outputs = model(**tokens)

            emb = outputs.last_hidden_state[:, 0]

        emb = emb.cpu().numpy().astype(np.float16)

        all_embeddings.append(emb)

    return np.concatenate(all_embeddings, axis=0)


def main():

    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    genome = load_genome(GENOME_PATH)

    for chrom in genome:

        print(f"Embedding {chrom}")

        seq = str(genome[chrom].seq)

        chunks = chunk_sequence(seq)

        embeddings = encode_chunks(chunks)

        mmap = np.memmap(
            f"{EMBEDDING_DIR}/{chrom}.fp16.mmap",
            mode="w+",
            dtype=np.float16,
            shape=embeddings.shape
        )

        mmap[:] = embeddings[:]

        mmap.flush()

        print(embeddings.shape)


if __name__ == "__main__":
    main()