from vae_imputer import MC_VAE_Imputer
import argparse
import numpy as np
import pandas as pd
import logging
import os
import pickle

from utils import prior

logging.getLogger().setLevel(logging.INFO)


def load_dataset(dataset, missing_prior="mean"):
    """
    Loads generated dataset for reproducibility & applies prior

    Args:
    dataset (str): dataset to load
    missing_prior (str): prior to apply prior to

    Returns trainX, trainM, testX, testM, testX_ori

    """

    logging.info(f"Loading dataset: {dataset}...")

    assert missing_prior in [
        "mean",
        "zero",
        "uniform",
        "epsilon",
        "std",
    ], "Not in ['mean', 'zero', 'uniform', 'epsilon', 'std']"

    interim_folder = f"./data/interim/{dataset}/"

    trainX = np.array(pd.read_csv(interim_folder + "trainX.csv"))
    trainM = np.array(pd.read_csv(interim_folder + "trainM.csv"))
    testX = np.array(pd.read_csv(interim_folder + "testX.csv"))
    testM = np.array(pd.read_csv(interim_folder + "testM.csv"))

    testX_ori = prior(testX, testM, prior_type=missing_prior)

    trainX = prior(trainX, trainM, prior_type=missing_prior)

    return trainX, trainM, testX, testM, testX_ori


def create_dataset(trainX, testX_ori):

    """
    Creates dataset for training

    i.e. the data + missing masks

    Args:
    trainX (df): training data
    testX_ori (df): test data

    Returns:
    x_train (df): train dataset
    x_test (df): test dataset
    missing_mask (np.array): train missing mask
    x_test_missing_mask (np.array): test missing mask
    """

    def create_vae_data(data):
        if data.dtype != "f" and data.dtype != "d":
            data = data.astype(float)
        data[np.isnan(data)] = -1
        return data

    logging.info("Creating dataset for training...")
    x_train = trainX
    x_test = testX_ori

    x_train = create_vae_data(trainX)
    x_test = create_vae_data(testX_ori)

    missing_mask = ~trainM.astype(bool)
    x_test_missing_mask = ~testM.astype(bool)

    return x_train, x_test, missing_mask, x_test_missing_mask


def train_vae(x_train, x_test, missing_mask, x_test_missing_mask, epochs, mc_samples):
    """
    Training function for the VAE

    """

    logging.info("Training")

    # Instantiate object of the VAE and build the model
    vaedropout = MC_VAE_Imputer(n_dims=trainX.shape[1], mc_samples=mc_samples)

    # Output built model
    vaedropout.model.summary()

    # Training loop
    output = []
    for i in range(1):
        vae_complete = vaedropout.fit(
            x_train,
            x_test,
            missing_mask=missing_mask,
            x_test_missing_mask=x_test_missing_mask,
            epochs=epochs,
            batch_size=1024,
        )

        output.append(vae_complete)

    return output


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", default="imputed.csv", help="output (csv) file")
    parser.add_argument("--it", default=2, type=int, help="iterations")
    parser.add_argument(
        "--dataset",
        help="load one of the available/buildin datasets"
        " [spam, spambase, letter, ...] use show to see a list",
    )
    parser.add_argument("--bs", default=128, type=int, help="batch size")
    parser.add_argument("--verbose", default=0, type=int, help="")
    parser.add_argument("--prior", default="mean", type=str, help="")
    parser.add_argument("--mcsamples", default=50, type=int, help="")

    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()

    dataset = args.dataset

    # If folder does not exist to write data, then create it
    # Else we know the path exists
    folder_path = f"./data/imputed/{dataset}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fn_ocsv = f"./data/imputed/{dataset}/" + args.o
    odir = os.path.dirname(fn_ocsv)
    odir = odir if len(odir) else "."

    chosen_prior = args.prior

    mb_size = args.bs

    n_iters = args.it

    mc_samples = args.mcsamples

    is_verbose = args.verbose

    # Load dataset
    trainX, trainM, testX, testM, testX_ori = load_dataset(
        dataset, missing_prior=chosen_prior
    )

    # Create tuple dataset for training (data, missing)
    x_train, x_test, missing_mask, x_test_missing_mask = create_dataset(
        trainX, testX_ori
    )

    # Start training
    testX_imputed_multiple = train_vae(
        x_train,
        x_test,
        missing_mask,
        x_test_missing_mask,
        epochs=n_iters,
        mc_samples=mc_samples,
    )

    PIK = f"./data/imputed/{dataset}/" + args.o
    with open(PIK, "wb") as f:
        pickle.dump(testX_imputed_multiple, f)
