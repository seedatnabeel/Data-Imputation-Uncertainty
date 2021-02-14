"""
Written by Jinsung Yoon
Date: Jan 29th 2019
Generative Adversarial Imputation Networks (GAIN) Implementation on Spam Dataset
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN.pdf
Appendix Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN_Supp.pdf
Contact: jsyoon0823@g.ucla.edu

Modified by: Nabeel Seedat
Date: Feb 2021
"""

#%% Packages
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import initpath_alg

initpath_alg.init_sys_path()
import utilmlab
import data_loader_mlab
import matplotlib.pyplot as plt


def generate_mask(a, pmiss):
    Dim = a.shape[1]
    No = a.shape[0]
    p_miss_vec = p_miss * np.ones((Dim, 1))

    Missing = np.zeros((No, Dim))
    for i in range(Dim):
        A = np.random.uniform(
            0.0,
            1.0,
            size=[
                len(Data),
            ],
        )
        B = A > p_miss_vec[i]
        Missing[:, i] = 1.0 * B
    return Missing


def serialize_scaler(scaler, folder_path):
    import pickle

    PIK = folder_path + "scaler.p"
    with open(PIK, "wb") as f:
        pickle.dump(scaler, f)


def write_tmp_data(df, columns, file_name, folder_path):
    df_Data = pd.DataFrame(df, columns=columns)

    df_Data.to_csv(folder_path + file_name, index=False)


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", default="imputed.csv", help="output (csv) file")
    parser.add_argument("--it", default=5000, type=int, help="iterations")
    parser.add_argument(
        "--dataset",
        help="load one of the available/buildin datasets"
        " [spam, spambase, letter, ...] use show to see a list",
    )
    parser.add_argument(
        "-i",
        help="load data as a csv file, requires the name of the label"
        " (reponsevar) to be specified as well (if applicable), this column"
        "will not be processed",
    )
    parser.add_argument(
        "--target",
        help="specifies the column with the response var "
        "if applicable when loading a csv file, this column will"
        " not be processed",
    )
    parser.add_argument("--testall", type=int, default=1)
    parser.add_argument("--ref")
    parser.add_argument("--bs", default=128, type=int, help="batch size")
    parser.add_argument("--pmiss", default=0.2, type=float, help="missing rate")
    parser.add_argument("--phint", default=0.5, type=float, help="hint rate")
    parser.add_argument("--alpha", default=10, type=float, help="")
    parser.add_argument("--autocategorical", default=1, type=int, help="")
    parser.add_argument("--verbose", default=0, type=int, help="")
    parser.add_argument("--trainratio", default=0.8, type=float, help="")
    parser.add_argument(
        "--mnist_visualise",
        default=False,
        type=bool,
        help="Saves some intermediate resulting imputations for MNIST",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()

    dataset = args.dataset

    fn_icsv = f"./data/raw/{dataset}/" + args.i
    fn_ref_csv = f"./data/raw/{dataset}/" + args.ref

    fn_ocsv = f"./data/raw/{dataset}/" + args.o
    odir = os.path.dirname(fn_ocsv)
    odir = odir if len(odir) else "."
    logger = utilmlab.init_logger(odir)

    mb_size = args.bs
    p_miss = args.pmiss
    p_hint = args.phint
    alpha = args.alpha
    train_rate = args.trainratio
    dataset = args.dataset
    niter = args.it
    test_all = args.testall
    label = args.target
    is_auto_categorical = args.autocategorical
    is_cat_one_hot = args.autocategorical == 2
    is_verbose = args.verbose
    mnist_visualise = args.mnist_visualise
    logger.info(
        "gain data:{} # it:{} testall:{} odir:{} "
        "autocat:{} is_cat_one_hot:{}".format(
            dataset if dataset is not None else fn_ocsv,
            niter,
            test_all,
            odir,
            is_auto_categorical,
            is_cat_one_hot,
        )
    )

    logger.info("")
    logger.info("{}".format(args))
    logger.info("")

    if fn_icsv is not None:
        if is_verbose:
            logger.info("loading csv {}".format(fn_icsv))

        df = pd.read_csv(fn_icsv)
        features = list(df.columns)
        if label is not None:
            target_vals = df[label].values
            if label not in features:
                assert label in features
                features.remove(label)
        if is_auto_categorical:
            df_tmp, prop_df_one_hot = utilmlab.df_cat_to_one_hot(
                df[features], is_verbose=is_verbose, is_cat_one_hot=is_cat_one_hot
            )
            Data = df_tmp.values
        else:
            Data = df[features].values
        Missing = np.where(np.isnan(Data), 0.0, 1.0)
        Data = np.where(Missing, Data, 0)
        if fn_ref_csv is not None:
            df_ref = pd.read_csv(fn_ref_csv)
        logger.info("features: #{} {} label:{}".format(len(features), features, label))
    else:
        logger.info("loading {} using dataloader".format(dataset))
        rval, dset = data_loader_mlab.get_dataset(dataset)
        assert rval == 0
        data_loader_mlab.dataset_log_properties(logger, dset)
        features = dset["features"]
        Data = dset["df"][dset["features"]].values.astype(np.float)

    # If folder does not exist to write data, then create it
    # Else we know the path exists
    import os

    folder_path = f"./data/interim/{dataset}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Parameters
    No = len(Data)
    Dim = len(Data[0, :])

    # Hidden state dimensions
    H_Dim1 = Dim
    H_Dim2 = Dim

    if True:
        if fn_icsv is not None:
            pass
        else:
            Missing = generate_mask(Data, p_miss)

    idx = np.random.permutation(No)

    Train_No = int(No * train_rate)
    Test_No = No - Train_No

    # Shuffle data sets unless we want to keep track of mnist results for visualisation
    if mnist_visualise and data_set == "mnist":
        trainX = Data[:Train_No]
        testX = Data[Train_No:]
        trainM = Missing[:Train_No]
        testM = Missing[Train_No:]
    else:
        trainX = Data[idx[:Train_No], :]
        testX = Data[idx[Train_No:], :]
        trainM = Missing[idx[:Train_No], :]
        testM = Missing[idx[Train_No:], :]
        trainY = target_vals[idx[:Train_No]]
        testY = target_vals[idx[Train_No:]]

    write_tmp_data(
        df=idx[:Train_No],
        columns=["idx"],
        file_name="trainIdx.csv",
        folder_path=folder_path,
    )
    write_tmp_data(
        df=idx[Train_No:],
        columns=["idx"],
        file_name="testIdx.csv",
        folder_path=folder_path,
    )

    # scale/normalize dataset
    range_scaler = (0, 1)
    scaler = MinMaxScaler(feature_range=range_scaler)
    scaler.fit(trainX)

    trainX = scaler.transform(trainX)

    if fn_ref_csv:
        testX = df_ref[features].values[idx[Train_No:], :]

    testX = scaler.transform(testX)
    Data = scaler.transform(Data)

    # Write the datasets to folder - so that it's reusable & reproducible for the other modelling approaches
    write_tmp_data(
        df=testY, columns=["target"], file_name="testY.csv", folder_path=folder_path
    )

    write_tmp_data(
        df=trainY, columns=["target"], file_name="trainY.csv", folder_path=folder_path
    )

    write_tmp_data(
        df=testM, columns=features, file_name="testM.csv", folder_path=folder_path
    )

    write_tmp_data(
        df=trainM, columns=features, file_name="trainM.csv", folder_path=folder_path
    )

    write_tmp_data(
        df=testX, columns=features, file_name="testX.csv", folder_path=folder_path
    )

    write_tmp_data(
        df=trainX, columns=features, file_name="trainX.csv", folder_path=folder_path
    )

    write_tmp_data(
        df=Missing, columns=features, file_name="Missingdf.csv", folder_path=folder_path
    )

    write_tmp_data(
        df=Data, columns=features, file_name="Datadf.csv", folder_path=folder_path
    )

    write_tmp_data(
        df=idx[:Train_No],
        columns=["idx"],
        file_name="trainIdx.csv",
        folder_path=folder_path,
    )

    write_tmp_data(
        df=idx[Train_No:],
        columns=["idx"],
        file_name="testIdx.csv",
        folder_path=folder_path,
    )

    # Serialize the data scaler to use on the other models
    serialize_scaler(scaler, folder_path)
