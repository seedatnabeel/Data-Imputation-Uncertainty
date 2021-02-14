from tqdm import tqdm
import numpy as np


def compute_mses(uncertain_dict, testX, samples):
    """
    Computes the MSE for multiple samples on the uncertain/imputed data

    Args:
    uncertain_dict (dict): dictionary of uncertainty/imputed indices
    testX (np.array): array of data with imputed data
    samples (np.array): multiple samples of the data

    Returns: mses(list)
    """

    mses = []
    for row in tqdm(uncertain_dict.keys()):
        preds = []
        targets = []
        for col in uncertain_dict[row]:
            true_target = testX[row][col]
            for s in range(samples.shape[0]):
                preds.append(samples[s][row][col])
                targets.append(true_target)

            mses.append(mse(np.array(preds), np.array(targets)))

    return mses


def compute_rmses(uncertain_dict, testX, samples):
    """
    Computes the RMSE for multiple samples on the uncertain/imputed data

    Args:
    uncertain_dict (dict): dictionary of uncertainty/imputed indices
    testX (np.array): array of data with imputed data
    samples (np.array): multiple samples of the data

    Returns: rmses(list)
    """
    rmses = []
    for row in tqdm(uncertain_dict.keys()):
        preds = []
        targets = []
        for col in uncertain_dict[row]:
            true_target = testX[row][col]
            for s in range(samples.shape[0]):
                preds.append(samples[s][row][col])
                targets.append(true_target)

            rmses.append(rmse(np.array(preds), np.array(targets)))

    return rmses


def oracle_rmse(testX, imputedX, testM):
    """
    Computes the RMSE for an oracle
    (i.e. knows which are the most inaccurate samples)

    Args:
    testX (np.array): ground truth array
    imputedX (np.array): imputed array
    testM (np.array): missing mask array

    Returns: rmse_oracle (list)
    """
    from sklearn.metrics import mean_squared_error

    percents = np.linspace(0.01, 0.99, 10)
    amounts = percents * testX.shape[0]

    # apply the missing mask
    true = testX[~testM.astype(bool)]
    preds = imputedX[~testM.astype(bool)]

    # compute ground truth errors
    errors = preds - true

    uncert = np.argsort(errors)

    rmseval = mean_squared_error(true, preds)

    rmse_oracle = []

    for amount in amounts:
        excl = uncert[: -int(amount)]

        rmseval = mean_squared_error(true[excl], preds[excl])

        rmse_oracle.append(rmseval)

    return rmse_oracle


def compute_grad(data):
    """
    Computes the Error Gradient Drop for any arbitrary error vector

    Args:
    data (list): error vector

    Returns: error_drop (float)
    """

    error_grad_drop = data[0] / data[-1]
    return error_grad_drop


def compute_aoc_curve_and_value(rmses, rmse_oracle):
    """
    Computes AOC curve and the AUSE

    Args:
    rmses (list): rmses for a specific model
    rmse_oracle (list): rmse for the oracle

    Returns:
    aoc_list (list): sparsification curve
    auc (float): AUSE value
    """
    from sklearn.metrics import auc

    assert len(rmses) == len(rmse_oracle)

    aoc_list = np.array(rmses) - np.array(rmse_oracle)

    xx = np.arange(1, 100, len(rmses))
    yy = aoc_list

    return aoc_list, auc(xx, yy)


def ece_calculation(acc, conf, bin_sizes):
    """
    Computes Expected Calibration Error

    Args:
    acc (list): accuracy vector
    conf (list): confidence vector
    bin_size (float): size of bin to divide vector into

    Returns: ece (float)
    """
    ece = 0
    for m in np.arange(len(bin_sizes)):
        ece = ece + (bin_sizes[m] / sum(bin_sizes)) * np.abs(acc[m] - conf[m])
    return ece


def rmse(preds, targets):
    """
    Computes RMSE between predictions & targets

    Args:
    preds (list): predictions vector
    targets (list): targets vector

    Returns: rmse (float)
    """
    preds = np.array(preds)
    targets = np.array(targets)

    return np.sqrt(np.mean((preds - targets) ** 2))


def mse(preds, targets):
    """
    Computes MSE between predictions & targets

    Args:
    preds (list): predictions vector
    targets (list): targets vector

    Returns: mse (float)
    """

    preds = np.array(preds)
    targets = np.array(targets)
    return np.mean((preds - targets) ** 2)
