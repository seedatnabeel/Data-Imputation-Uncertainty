from utils import normdata, myrmse
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    roc_auc_score,
    mean_squared_error,
)
import numpy as np
import random
import matplotlib.pyplot as plt


def performance_vs_confidence(
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

    gt_y = []

    acc_scores = []

    # apply mask
    true = testX[~missing_data.astype(bool)]
    preds = imp_data[~missing_data.astype(bool)]

    # oracle error
    errors = np.abs(preds - true)

    # sort based on error - oracle
    uncert_oracle = np.argsort(errors)

    rmse_oracle = []

    for count, amount in enumerate(amounts):
        idx = int(amount)

        # Calculations and exclusions based on variance
        excl = uncert[:-idx]
        ori_data = testX[excl, :]
        imputed_data = imp_data[excl, :]
        data_m = np.array(df_mis != df_mis)[excl, :]
        rmse = myrmse(
            actual=ori_data, predicted=imputed_data, mask=~data_m.astype(bool)
        )
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
        data_m = np.array(df_mis != df_mis)[excl, :]
        rmse = myrmse(
            actual=ori_data, predicted=imputed_data, mask=~data_m.astype(bool)
        )
        cv_rmses_retention.append(rmse)

        # Calculations and exclusions based on random
        rand_excl = random.sample(range(len(uncert)), idx)
        ori_data = testX[rand_excl, :]
        imputed_data = imp_data[rand_excl, :]
        data_m = np.array(df_mis != df_mis)[rand_excl, :]
        rmse = myrmse(
            actual=ori_data, predicted=imputed_data, mask=~data_m.astype(bool)
        )
        random_rmses_retention.append(rmse)

    return (
        uncert_rmses_retention,
        cv_rmses_retention,
        random_rmses_retention,
        y_score_retention,
        auc_retention,
        gt_y,
        acc_scores,
        rmse_oracle[:-1],
    )


def plot_rmse_conf_curve(analysis_scores, dataset, filename):
    """
    Plots the RMSE Confidence-Exclusion curve
    """

    plt.style.reload_library()
    plt.style.use(["science", "ieee", "no-latex", "notebook", "grid", "vibrant"])

    mean_uncert = np.mean(analysis_scores["uncert_rmses_retention"], axis=0)
    std_uncert = np.std(analysis_scores["uncert_rmses_retention"], axis=0)
    plt.plot(np.linspace(0, 1, 10), mean_uncert, label="Variance", marker="o")
    plt.fill_between(
        np.linspace(0, 1, 10),
        mean_uncert - std_uncert,
        mean_uncert + std_uncert,
        alpha=0.25,
    )

    mean_uncert = np.mean(analysis_scores["random_rmses_retention"], axis=0)
    std_uncert = np.std(analysis_scores["random_rmses_retention"], axis=0)
    plt.plot(np.linspace(0, 1, 10), mean_uncert, label="Random", marker="o")
    plt.fill_between(
        np.linspace(0, 1, 10),
        mean_uncert - std_uncert,
        mean_uncert + std_uncert,
        alpha=0.25,
    )

    mean_uncert = np.mean(analysis_scores["cv_rmses_retention"], axis=0)
    std_uncert = np.std(analysis_scores["cv_rmses_retention"], axis=0)
    plt.plot(np.linspace(0, 1, 10), mean_uncert, label="CV", marker="o")
    plt.fill_between(
        np.linspace(0, 1, 10),
        mean_uncert - std_uncert,
        mean_uncert + std_uncert,
        alpha=0.25,
    )

    mean_uncert = np.mean(analysis_scores["rmse_oracle"], axis=0)
    std_uncert = np.std(analysis_scores["rmse_oracle"], axis=0)
    plt.plot(np.linspace(0, 1, 11), mean_uncert, label="Oracle", marker="o")
    plt.fill_between(
        np.linspace(0, 1, 11),
        mean_uncert - std_uncert,
        mean_uncert + std_uncert,
        alpha=0.25,
    )

    plt.xlabel("Proportion Data Excluded")
    plt.ylabel("RMSE")
    plt.legend()

    plt.savefig(f"data/results/{dataset}/{filename}.png")


def plot_reliability_diagram(uncertainty_list, rmses, dataset, filename):
    """
    Plots the Reliability diagram
    """

    f, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(uncertainty_list, rmses, c=".3")

    (diag_line,) = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

    plt.xlabel("Uncertainty")
    plt.ylabel("RMSE")

    plt.savefig(f"data/results/{dataset}/reliability_{filename}.png")


def plot_auc_sparsification(
    aoc_uncert_lists,
    aoc_uncert_rand_lists,
    aoc_uncert_cv_lists,
    auc_uncerts,
    auc_rands,
    auc_cvs,
    dataset,
    filename,
):
    """
    Plots the Sparsification curve
    """

    plt.figure()
    xx = np.linspace(0, 1, 10)

    mean_uncert = np.mean(aoc_uncert_lists, axis=0)
    std_uncert = np.std(aoc_uncert_lists, axis=0)
    auc_label = f"Variances Scores AUC: {round(np.mean(auc_uncerts),2)}+-{round(np.std(auc_uncerts),2)}"
    plt.plot(np.linspace(0, 1, 10), mean_uncert, marker="o", label=auc_label)
    plt.fill_between(
        np.linspace(0, 1, 10),
        mean_uncert - std_uncert,
        mean_uncert + std_uncert,
        alpha=0.25,
    )

    mean_uncert = np.mean(aoc_uncert_rand_lists, axis=0)
    std_uncert = np.std(aoc_uncert_rand_lists, axis=0)
    auc_label = f"Random Scores AUC: {round(np.mean(auc_rands),2)}+-{round(np.std(auc_rands),2)}"
    plt.plot(np.linspace(0, 1, 10), mean_uncert, marker="o", label=auc_label)
    plt.fill_between(
        np.linspace(0, 1, 10),
        mean_uncert - std_uncert,
        mean_uncert + std_uncert,
        alpha=0.25,
    )

    mean_uncert = np.mean(aoc_uncert_cv_lists, axis=0)
    std_uncert = np.std(aoc_uncert_cv_lists, axis=0)
    auc_label = (
        f"Coeff Variation AUC: {round(np.mean(auc_cvs),2)}+-{round(np.std(auc_cvs),2)}"
    )
    plt.plot(np.linspace(0, 1, 10), mean_uncert, marker="o", label=auc_label)
    plt.fill_between(
        np.linspace(0, 1, 10),
        mean_uncert - std_uncert,
        mean_uncert + std_uncert,
        alpha=0.25,
    )

    plt.legend()
    plt.title("Area under the sparsification curve")

    plt.savefig(f"data/results/{dataset}/{filename}.png")
