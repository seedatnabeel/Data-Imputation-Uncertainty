import argparse
import numpy as np
import pandas as pd
import time
import os
import json

# import initpath_alg
# initpath_alg.init_sys_path()
import utilmlab
import data_loader_mlab


def normalize_array(a):
    Dim = a.shape[1]
    Min_Val = np.zeros(Dim)
    Max_Val = np.zeros(Dim)

    for i in range(Dim):
        Min_Val[i] = np.min(a[:, i])
        a[:, i] = a[:, i] - np.min(a[:, i])
        Max_Val[i] = np.max(a[:, i])
        a[:, i] = a[:, i] / (np.max(a[:, i]) + 1e-6)
    return a


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="spam")
    parser.add_argument("--pmiss", default=0.2, type=float, help="missing rate")
    parser.add_argument("--oref", default="x.csv")
    parser.add_argument("-o", default="xmissing.csv")
    parser.add_argument("--properties")
    parser.add_argument("--normalize01", default=1, type=int)
    parser.add_argument("-n", default=0, type=int)
    parser.add_argument(
        "--istarget", help="include target if not zero", default=0, type=int
    )
    parser.add_argument(
        "--uniform_miss",
        help="uniform missing vs non-uniform. Set 1 for uniform, set 0 for Type II (see project), set 2 for Type III. Set 4 for generating line-missing data in mnist. Use option 3 for manually setting the amount of correlation.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--miss_corr",
        help="correlation of missing masks. Only used when uniform_miss=3",
        default=0.0,
        type=float,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()
    dataset = args.dataset

    # If folder does not exist to write data, then create it
    # Else we know the path exists
    import os

    folder_path = f"./data/raw/{dataset}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    p_miss = args.pmiss
    odir = os.path.dirname(args.o)
    nsample = args.n
    logger = utilmlab.init_logger(odir if len(odir) else ".")
    islabel = args.istarget
    fn_missing_csv = folder_path + args.o
    fn_csv = folder_path + args.oref
    fn_json = args.properties
    uniform_miss = args.uniform_miss
    miss_corr = args.miss_corr

    is_normalize_0_1 = args.normalize01

    rval, dset = data_loader_mlab.get_dataset(dataset, nsample)
    assert rval == 0
    data_loader_mlab.dataset_log_properties(logger, dset)

    df = dset["df"]
    features = dset["features"]
    labels = dset["targets"]
    features_drop = []
    for el in features:
        # drop columns with missing data as we cannot then calculate the rmse
        if sum(dset["df"][el].isnull()):
            features_drop.append(el)
    if len(features_drop):
        logger.info("dropping features {}".format(features_drop))
        time.sleep(2)

    features = [el for el in features if el not in features_drop]

    data = dset["df"][features].values.astype(np.float)

    if is_normalize_0_1:
        data = normalize_array(data)

    X_missing = utilmlab.introduce_missing(data, p_miss, uniform_miss, miss_corr)
    df_missing = pd.DataFrame(X_missing, columns=features)

    df_missing = pd.DataFrame(X_missing, columns=features)

    if islabel:
        df_missing[labels] = df[labels]

    df_missing.to_csv(fn_missing_csv, index=False)

    df_data = pd.DataFrame(data, columns=features)

    if islabel:
        df_data[labels] = df[labels]

    df_data.to_csv(fn_csv, index=False)

    if fn_json is not None:
        with open(fn_json, "w") as f:
            dset_prop = {"features": dset["features"], "targets": dset["targets"]}
            json.dump(dset_prop, f)
