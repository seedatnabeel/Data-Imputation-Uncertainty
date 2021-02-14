from analysis_utils import *
from analysis_diagrams import *
import numpy as np
import pandas as pd
import argparse
from utils import (
    prior,
    get_uncert_dict,
    compute_uncertainty_per_row,
    get_missing_indices,
    normdata,
    myrmse,
)
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import random
import pickle
import os


def performance_vs_confidence(
    analysis_scores,
    original_data,
    imp_data,
    missing_data,
    testY,
    test_idx,
    total_uncertainty,
    coeff_variation,
    clf=None,
):
    """
    Computes the performance vs confidence (i.e exclusions)

    Args:
    analysis_scores (dict): dict of different analysis scores

    """

    df_mis = missing_data
    testX = original_data

    percents = np.linspace(0.01, 0.9, 10)
    amounts = percents * testX.shape[0]

    # sort based on variance
    uncert = np.argsort(total_uncertainty)
    # sort based on CV
    cv_uncert = np.argsort(coeff_variation)[::-1]

    uncert_rmses_retention = []
    cv_rmses_retention = []
    random_rmses_retention = []

    y_score_retention = []
    auc_retention = []

    rmse_oracle = []

    gt_y = []

    acc_scores = []

    # apply mask
    true = testX[~missing_data.astype(bool)]
    preds = imp_data[~missing_data.astype(bool)]

    # oracle error
    errors = np.abs(preds - true)

    # sort based on error - oracle
    uncert_oracle = np.argsort(errors)

    for count, amount in enumerate(amounts):
        idx = int(amount)

        # Calculations and exclusions based on variance
        excl = uncert[:-idx]
        ori_data = testX[excl, :]
        imputed_data = imp_data[excl, :]
        data_m = df_mis[excl, :]
        rmse = myrmse(
            actual=ori_data, predicted=imputed_data, mask=~data_m.astype(bool)
        )
        print(rmse)
        uncert_rmses_retention.append(rmse)

        # Calculations for oracle
        if count > 0:
            excl_oracle = uncert_oracle[: -int(amount)]
            rmseval = mean_squared_error(true[excl_oracle], preds[excl_oracle])
            rmse_oracle.append(rmseval)
        else:
            rmse_oracle.append(rmse)
            excl_oracle = uncert_oracle[: -int(amount)]
            rmseval = mean_squared_error(true[excl_oracle], preds[excl_oracle])
            rmse_oracle.append(rmseval)

        # if a classifier is specified apply the sortings for diff acc and auc
        if clf:
            y_preds = clf.predict(imputed_data[:, 0:-1])

            y_scores = clf.predict_proba(imputed_data[:, 0:-1])[:, 1]

            if len(np.unique(testY)) == 2:
                auc_retention.append(
                    roc_auc_score(testY[excl], y_scores, multi_class="ovr")
                )

            y_score_retention.append(y_scores)

            gt_y.append(testY[excl])

            acc_scores.append(accuracy_score(testY[excl], y_preds))

        # Calculations and exclusions based on CV
        excl = cv_uncert[:-idx]
        ori_data = testX[excl, :]
        imputed_data = imp_data[excl, :]
        data_m = df_mis[excl, :]
        rmse = myrmse(
            actual=ori_data, predicted=imputed_data, mask=~data_m.astype(bool)
        )
        cv_rmses_retention.append(rmse)

        # Calculations and exclusions based on random
        rand_excl = random.sample(range(len(uncert)), idx)
        ori_data = testX[rand_excl, :]
        imputed_data = imp_data[rand_excl, :]
        data_m = df_mis[rand_excl, :]
        rmse = myrmse(
            actual=ori_data, predicted=imputed_data, mask=~data_m.astype(bool)
        )
        random_rmses_retention.append(rmse)

    # Create dictionary of results for easier indexing
    analysis_scores["uncert_rmses_retention"].append(uncert_rmses_retention)
    analysis_scores["cv_rmses_retention"].append(cv_rmses_retention)
    analysis_scores["random_rmses_retention"].append(random_rmses_retention)
    analysis_scores["y_score_retention"].append(y_score_retention)
    analysis_scores["auc_retention"].append(auc_retention)
    analysis_scores["gt_y"].append(gt_y)
    analysis_scores["acc_scores"].append(acc_scores)
    analysis_scores["rmse_oracle"].append(rmse_oracle)

    return analysis_scores


def load_results(dataset, pickle_name):
    """
    Loads a dataset and results

    Args:
    dataset : dataset components to load name to load
    pickle_name (str): name of  pickle file

    """
    interim_folder = f"./data/interim/{dataset}/"

    trainY = np.array(pd.read_csv(interim_folder + "trainY.csv").target.values)
    testY = np.array(pd.read_csv(interim_folder + "testY.csv").target.values)

    test_idx = pd.read_csv(interim_folder + "testIdx.csv").idx.values
    train_idx = pd.read_csv(interim_folder + "trainIdx.csv").idx.values

    trainX = np.array(pd.read_csv(interim_folder + "trainX.csv"))
    trainM = np.array(pd.read_csv(interim_folder + "trainM.csv"))
    testX = np.array(pd.read_csv(interim_folder + "testX.csv"))
    testM = np.array(pd.read_csv(interim_folder + "testM.csv"))

    testX_ori = prior(testX, testM, prior_type="zero")

    trainX = prior(trainX, trainM, prior_type="zero")

    PIK = f"./data/imputed/{dataset}/" + pickle_name
    with open(PIK, "rb") as f:
        output = pickle.load(f)

    return (
        trainY,
        testY,
        test_idx,
        train_idx,
        trainX,
        trainM,
        testX,
        testM,
        testX_ori,
        trainX,
        output,
    )


def serialize_analysis_scores(analysis_scores, name):
    """
    Serializes a set of scores

    Args:
    analysis_scores (dict): dict scores returned from the analysis
    name (str): name of file

    """
    PIK = name
    with open(PIK, "wb") as f:
        pickle.dump(analysis_scores, f)


def unserialize_analysis_scores(name):
    """
    Deserializes a set of scores

    Args:
    name (str): name of file

    Returns:
    output (dict)
    """
    PIK = name
    with open(PIK, "rb") as f:
        output = pickle.load(f)
    return output


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="spam")

    parser.add_argument("--serialized", default="testX_imputed_multiple.p")

    parser.add_argument("--new", default=0)

    parser.add_argument("--model", default="VAE")
    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()

    dataset = args.dataset

    pickle_name = args.serialized

    model = args.model

    new = args.new

    (
        trainY,
        testY,
        test_idx,
        train_idx,
        trainX,
        trainM,
        testX,
        testM,
        testX_ori,
        trainX,
        output,
    ) = load_results(dataset, pickle_name)

    # get missing indices
    miss_ind = get_missing_indices(testM)

    uncertain_dict = get_uncert_dict(miss_ind)

    if model == "GAIN":
        samples = np.array(output)
        imputed_data = normdata(np.mean(np.array(output), axis=0))[test_idx, :]
        testM = pd.read_csv(f"./data/raw/{dataset}/missing.csv")
        testMissing = np.array(testM != testM)[test_idx, :]
        testX = normdata(np.array(pd.read_csv(f"./data/raw/{dataset}/ref.csv")))[
            test_idx, :
        ]
    else:
        samples = output[0][2]
        imputed_data = output[0][1]
        testMissing = np.array(testM != testM)

    (
        total_uncertainty,
        uncertainty_list,
        deviation_uncertainty,
        mean_uncertainty,
    ) = compute_uncertainty_per_row(samples, uncertain_dict)

    mses = compute_mses(uncertain_dict, testX, samples)

    rmses = compute_rmses(uncertain_dict, testX, samples)

    df_mis = pd.read_csv(f"./data/interim/{dataset}/testM.csv")

    coeff_variation = np.array(deviation_uncertainty) / np.array(mean_uncertainty)

    folder_path = f"./data/results/{dataset}/"

    # If folder does not exist to write data, then create it
    # Else we know the path exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = f"./data/results/{dataset}/analysis_scores_{dataset}.p"
    if new == "true" and os.path.exists(file_path):
        os.remove(file_path)

    if not os.path.exists(file_path):
        from collections import defaultdict

        analysis_scores = defaultdict(list)
        serialize_analysis_scores(analysis_scores, file_path)

    analysis_scores = unserialize_analysis_scores(file_path)

    # compute analysis scores
    analysis_scores = performance_vs_confidence(
        analysis_scores,
        testX,
        imputed_data,
        testMissing,
        testY,
        test_idx,
        total_uncertainty,
        coeff_variation,
    )

    pred_hist = np.linspace(0, 1, 15)
    analysis_scores["ECE"].append(
        ece_calculation(acc=rmses, conf=uncertainty_list, bin_sizes=pred_hist)
    )

    # serialize for later graphing
    serialize_analysis_scores(analysis_scores, file_path)

    plot_reliability_diagram(uncertainty_list, rmses, dataset, pickle_name)
    plot_reliability_diagram(
        np.sqrt(uncertainty_list), rmses, dataset, pickle_name + "_root_mean"
    )
