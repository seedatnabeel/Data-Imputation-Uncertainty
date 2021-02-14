import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math
import pandas as pd
from tqdm import tqdm


def prior(arr, mask, prior_type):
    """
    Applies the chosen prior column wise

    Args:
    arr (np.array): array of values (incl missing)
    mask (np.array): missing mask
    prior_type (str): specified prior type to apply

    Return arr_prior (array with priors applied)
    """

    if prior_type == "mean":
        col_mean = np.nanmedian(arr, axis=0)
        inds = np.where(mask < 1)
        arr_prior = arr.copy()
        arr_prior[inds] = np.take(col_mean, inds[1])
        return arr_prior

    if prior_type == "std":
        col_mean = np.nanmean(arr, axis=0)
        col_std = np.nanstd(arr, axis=0)
        inds = np.where(mask < 1)
        arr_prior = arr.copy()
        arr_prior[inds] = np.take(
            np.random.uniform(col_mean - 1 * col_std, col_mean + 1 * col_std), inds[1]
        )

        return arr_prior

    elif prior_type == "zero":
        return np.where(mask < 1, 0, arr)

    elif prior_type == "epsilon":
        return np.where(mask < 1, np.random.uniform(0, 0.01), arr)

    elif prior_type == "uniform":
        return np.where(mask < 1, np.random.uniform(0, 1), arr)


def myrmse(actual, predicted, mask):
    """
    Computes the RMSE

    Args:
    actual: actual values
    predicted: predicted imputed values
    mask (np.array): boolean mask of missingness (i.e. which samples)

    Return RMSE
    """

    mse = mean_squared_error(actual[mask], predicted[mask])

    rmse = math.sqrt(mse)
    return rmse


def normdata(data):
    """
    Normalizes data between 0-1 and returns the norm data

    Args:
    data_scaled (np.array): scaled data (0-1)

    Returns: mses(list)
    """
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled


def get_uncert_dict(res):
    """
    Gets the row and column of missing values as a dict

    Args:
    res(np.array): missing mask


    Returns:   uncertain_dict (dict): dictionary with row and col of missingness
    """
    uncertain_dict = {}

    for mytuple in res:
        row = mytuple[0]
        col = mytuple[1]

        if uncertain_dict.get(row):
            uncertain_dict[row].append(col)
        else:
            uncertain_dict[row] = [col]
    return uncertain_dict


def compute_uncertainty_per_row(data_list, uncertain_dict):
    """
    Computes the uncertainty per row of the dataset

    Args:
    data_list (np.array): np.array of diff data samples
    uncertain_dict (dict): dictionary with row and col of missingness

    Returns: total_uncertainty, uncertainty_list,  deviation_uncertainty, mean_uncertainty
    """
    df_list = [
        pd.DataFrame(np_array, columns=list(range(data_list.shape[2])))
        for np_array in data_list
    ]
    grouped_df = pd.concat(df_list).groupby(level=0).std()

    uncert = []

    for row in tqdm(uncertain_dict.keys()):
        uncert.append(grouped_df.iloc[row, uncertain_dict[row]].values)

    total_uncertainty = []
    deviation_uncertainty = []
    mean_uncertainty = []
    uncertainty_list = []

    for val in np.array(uncert):
        for num in val:
            uncertainty_list.append(num)
        total_uncertainty.append(np.sum(val))
        deviation_uncertainty.append(np.std(val))
        mean_uncertainty.append(np.mean(val))

    return total_uncertainty, uncertainty_list, deviation_uncertainty, mean_uncertainty


def get_missing_indices(mask):
    """
    Gets indices where data is missing, e.g ==0

    Args:
    mask (np.array): missing masks (0,1), 0=missing

    Returns: indices_missing (np.array)
    """
    indices_missing = np.argwhere(mask == 0)
    return indices_missing
