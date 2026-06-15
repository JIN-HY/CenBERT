# CenBERT

CenBERT is a deep learning framework for predicting CENH3 enrichment directly from genome sequence. The model combines DNABERT2 sequence embeddings with a global Transformer architecture to learn long-range genomic context and identify centromeric regions from assembled genomes.

## 1. Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv CenBERT_env
source CenBERT_env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Generate DNABERT2 Embeddings

Before training or inference, generate DNABERT2 embeddings for your genome assembly:

```bash
python embed_genome.py
```

CenBERT uses DNABERT2 embeddings as input features. Embeddings are generated for all genomes listed in `sample-bw.txt`.

Prepare a tab-delimited `sample-bw.txt` file with the following format:

```text
sample_name    genome_path    bigwig_path
```

Example:

```text
S001    genome/S001.fa    chipseq/S001.bw
S002    genome/S002.fa    chipseq/S002.bw
```

If a BigWig file is unavailable, the third column may contain any placeholder value.

---

## 3. Training

After generating genome embeddings, train a CenBERT model using genome assemblies and corresponding CENH3 ChIP-seq data:

```bash
python train.py
```

Model checkpoints will be saved automatically according to the configuration settings.

---

## 4. Inference

You may use either a pretrained CenBERT model or a model trained on your own data.

After embedding generation is complete, run:

```bash
python inference.py model.pt outdir
```

where:

* `model.pt` is the trained model checkpoint.
* `outdir` is the output directory for predictions.

Predictions will be written to `outdir`.

---

## 5. Evaluation

If ground-truth CENH3 ChIP-seq BigWig files are available, predictions can be compared against the experimental signal.

```bash
python evaluation.py sample chrom outdir
```

Example:

```bash
python evaluation.py S001 Chr01 outdir
```

Parameters:

* `sample` — sample name.
* `chrom` — chromosome name.
* `outdir` — the same output directory used during inference.

Note that `outdir` should be specified exactly as used in the inference step and should not include additional parent directory paths.
