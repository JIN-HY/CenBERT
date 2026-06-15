import numpy as np
import pyBigWig
from sklearn.datasets import dump_svmlight_file
import xgboost as xgb
from xgboost import XGBRegressor
from config import *
from utils import *
from scipy.stats import pearsonr, spearmanr

SAMPLE = "S001"
EMBEDDING_DIR = "/mnt/e/CENH3/embeddings"

def load_targets(
    bw_path,
    chrom_sizes, 
    window_bp
):

    targets = []

    bw = pyBigWig.open(bw_path)

    for chrom, length in chrom_sizes.items():

        for start in range(
            0,
            length*window_bp,
            window_bp
        ):

            end = min(
                start + window_bp,
                length*window_bp
            )

            vals = bw.values(
                chrom,
                start,
                end,
                numpy=True
            )

            targets.append(
                np.nanmean(vals)
            )

    bw.close()

    return np.array(
        targets,
        dtype=np.float16
    )



def main():

    fa_dict, bw_dict = bw_map(BW_MAP)

    SAMPLE = "S001"

    genome = load_genome(
        fa_dict[SAMPLE]
    )

    chrom_sizes = get_chrom_sizes(
        genome,
        WINDOW_BP
    )

    all_X = []

    for chrom, n_tokens in chrom_sizes.items():

        x = np.memmap(
            f"{EMBEDDING_DIR}/{chrom}.fp16.mmap",
            mode="r",
            dtype=np.float16,
            shape=(n_tokens, D_MODEL)
        )

        all_X.append(
            np.asarray(x, dtype=np.float32)
        )
    print(len(all_X))

    X = np.concatenate(
        all_X,
        axis=0
    )

    y = load_targets(
        bw_path=bw_dict[SAMPLE],
        chrom_sizes=chrom_sizes,
        window_bp=WINDOW_BP
    )

    # print("X shape:", X.shape)
    # print("y shape:", y.shape)

    # assert X.shape[0] == y.shape[0]

    # dump_svmlight_file(
    #     X,
    #     y,
    #     "/mnt/e/CENH3/train.svm",
    #     zero_based=True
    # )

    # del X
    # del all_X

    # dtrain = xgb.DMatrix(
    #     "/mnt/e/CENH3/train.svm#train.cache"
    # )

    # params = {
    #     "objective": "reg:squarederror",
    #     "tree_method": "hist",
    #     "max_depth": 8,
    #     "eta": 0.05,
    #     "subsample": 0.8,
    #     "colsample_bytree": 0.8,
    #     "seed": 42,
    # }

    # model = xgb.train(
    #     params,
    #     dtrain,
    #     num_boost_round=500
    # )
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X, y)

    model.save_model(
        "xgboost_s001.json"
    )

    pred = model.predict(X)

    pearson = pearsonr(pred, y)[0]
    print(pearson)

if __name__ == "__main__":
    main()