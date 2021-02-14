from analysis_utils import *
from analysis_diagrams import *
import numpy as np
import pandas as pd
from utils import prior
import argparse
import pickle


def unserialize_analysis_scores(filename):
    """Function to unserialize_analysis_scores"""

    PIK = filename
    print(PIK)
    with open(PIK, "rb") as f:
        output = pickle.load(f)
    return output


def conf_gradient(analysis_scores):
    """Computes the Confidence-Exclusion gradient """
    error_drop_uncert_mean = np.mean(
        [
            compute_grad(analysis_scores["uncert_rmses_retention"][i])
            for i in range(len(analysis_scores["uncert_rmses_retention"]))
        ]
    )
    error_drop_uncert_std = np.std(
        [
            compute_grad(analysis_scores["uncert_rmses_retention"][i])
            for i in range(len(analysis_scores["uncert_rmses_retention"]))
        ]
    )

    error_drop_uncert_rand_mean = np.mean(
        [
            compute_grad(analysis_scores["random_rmses_retention"][i])
            for i in range(len(analysis_scores["random_rmses_retention"]))
        ]
    )
    error_drop_uncert_rand_std = np.std(
        [
            compute_grad(analysis_scores["random_rmses_retention"][i])
            for i in range(len(analysis_scores["random_rmses_retention"]))
        ]
    )

    error_drop_uncert_cv_mean = np.mean(
        [
            compute_grad(analysis_scores["cv_rmses_retention"][i])
            for i in range(len(analysis_scores["cv_rmses_retention"]))
        ]
    )
    error_drop_uncert_cv_std = np.std(
        [
            compute_grad(analysis_scores["cv_rmses_retention"][i])
            for i in range(len(analysis_scores["cv_rmses_retention"]))
        ]
    )

    print(
        f"Uncert = {round(error_drop_uncert_mean,2)}+-{round(error_drop_uncert_std,2)}"
    )
    print(
        f"Rand = {round(error_drop_uncert_rand_mean,2)}+-{round(error_drop_uncert_rand_std,2)}"
    )
    print(
        f"CV = {round(error_drop_uncert_cv_mean,2)}+-{round(error_drop_uncert_cv_std,2)}"
    )


def auc_curve_and_values(analysis_scores, dataset, filename):
    """Helper function to compute the Sparsification AUSE"""

    rmse_oracle = np.mean(analysis_scores["rmse_oracle"], axis=0)

    aoc_uncert_lists, auc_uncerts = [], []
    aoc_uncert_rand_lists, auc_rand_uncerts = [], []
    aoc_uncert_cv_lists, auc_cv_uncerts = [], []

    for i in range(len(analysis_scores["cv_rmses_retention"])):
        aoc_list_uncert, auc_uncert = compute_aoc_curve_and_value(
            analysis_scores["uncert_rmses_retention"][i], rmse_oracle[0:-1]
        )
        aoc_list_rand, auc_rand = compute_aoc_curve_and_value(
            analysis_scores["random_rmses_retention"][i], rmse_oracle[0:-1]
        )
        aoc_list_cv, auc_cv = compute_aoc_curve_and_value(
            analysis_scores["cv_rmses_retention"][i], rmse_oracle[0:-1]
        )

        aoc_uncert_lists.append(aoc_list_uncert)
        auc_uncerts.append(auc_uncert)
        aoc_uncert_rand_lists.append(aoc_list_rand)
        auc_rand_uncerts.append(auc_rand)
        aoc_uncert_cv_lists.append(aoc_list_cv)
        auc_cv_uncerts.append(auc_cv)

    plot_auc_sparsification(
        aoc_uncert_lists,
        aoc_uncert_rand_lists,
        aoc_uncert_cv_lists,
        auc_uncerts,
        auc_rand_uncerts,
        auc_cv_uncerts,
        dataset,
        filename,
    )


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="spam")

    parser.add_argument("--filename", default="analysis_scores_bc.p")

    parser.add_argument("--new", default=0)

    parser.add_argument("--model", default="x")

    parser.add_argument("--prior", default="x")
    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()

    dataset = args.dataset

    name = args.filename

    new = args.new

    model = args.model

    prior = args.prior

    import os

    folder_path = f"./data/results/{dataset}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    analysis_scores = unserialize_analysis_scores(folder_path + name)

    print(f"Results for {name}, model: {model} and prior: {prior}:")

    print(
        f"ECE = {round(np.mean(analysis_scores['ECE']),4)}+-{round(np.std(analysis_scores['ECE']),4)}"
    )

    plot_rmse_conf_curve(
        analysis_scores, dataset, filename=f"retention_curve_{name}_{prior}_{model}"
    )

    conf_gradient(analysis_scores)

    auc_curve_and_values(
        analysis_scores, dataset, filename=f"auc_curve_{name}_{prior}_{model}"
    )
