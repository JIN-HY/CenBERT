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