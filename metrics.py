import numpy as np


def mean_bias_error(preds: np.array, targets: np.array) -> float:
    """
    Compute the mean bias error (MBE).

    Parameters
    ----------
    preds: np.array
        A two-dimensional array of predictions over the data points.
    targets: np.array
        A two-dimensional array of target variables.

    Returns
    -------
    float
        The computed MBE, which indicates systematic over-prediction (positive)
        or under-prediction (negative).
    """
    return np.mean(preds - targets, axis=1)


def mbe(preds: np.array, targets: np.array) -> float:
    """See :func:`.mean_bias_error`."""
    return mean_bias_error(preds, targets)


def absolute_bias(preds: np.array, targets: np.array) -> float:
    """
    Compute the absolute bias.

    Parameters
    ----------
    preds: np.array
        A two-dimensional array of predictions over the data points.
    targets: np.array
        A two-dimensional array of target variables.

    Returns
    -------
    float
        The absolute value of the mean bias error.
    """
    return np.abs(mean_bias_error(preds, targets))


def percent_bias(preds: np.array, targets: np.array) -> float:
    """
    Compute the percent bias (PBIAS).

    Parameters
    ----------
    preds: np.array
        A two-dimensional array of predictions over the data points.
    targets: np.array
        A two-dimensional array of target variables.

    Returns
    -------
    float
        The percent bias, indicating the tendency of predicted values
        to be larger or smaller than observed values.
        Positive values indicate overestimation, negative values indicate underestimation.
    """
    return 100 * np.sum(preds - targets, axis=1) / np.sum(targets, axis=1)


def normalized_bias(preds: np.array, targets: np.array) -> float:
    """
    Compute the normalized bias.

    Parameters
    ----------
    preds: np.array
        A two-dimensional array of predictions over the data points.
    targets: np.array
        A two-dimensional array of target variables.

    Returns
    -------
    float
        The bias normalized by the standard deviation of the targets.
    """
    bias = mean_bias_error(preds, targets)
    target_std = np.std(targets, axis=1)
    return bias / target_std


def root_mean_squared_error(preds: np.array, targets: np.array) -> float:
    """
    Compute the root-mean-squared error (RMSE).

    Parameters
    ----------
    preds: np.array
        A two-dimensional array of predictions over the data points.
    targets: np.array
        A two-dimensional array of target variables.

    Returns
    -------
    float
        The computed RMSE.
    """
    return np.sqrt(np.mean((preds - targets) ** 2, axis=1))


def rmse(preds: np.array, targets: np.array) -> float:
    """See :func:`.root_mean_squared_error`."""
    return root_mean_squared_error(preds, targets)


def mean_squared_error(preds: np.array, targets: np.array) -> np.array:
    """
    Compute the mean-squared error (MSE).

    Parameters
    ----------
    preds: np.array
        A two-dimensional array of predictions over the data points.
    targets: np.array
        A two-dimensional array of target variables.

    Returns
    -------
    np.array
        The computed MSE.
    """
    return np.mean((preds - targets) ** 2, axis=1)


def mse(preds: np.array, targets: np.array) -> float:
    """See :func:`.mean_squared_error`."""
    return mean_squared_error(preds, targets)


def root_mean_absolute_error(preds: np.array, targets: np.array) -> float:
    """
    Compute the root-mean-absolute error (RMAE).

    Parameters
    ----------
    preds: np.array
        A two-dimensional array of predictions over the data points.
    targets: np.array
        A two-dimensional array of target variables.

    Returns
    -------
    float
        The computed RMAE.
    """
    return np.sqrt(np.mean(np.abs(preds - targets), axis=1))


def rmae(preds: np.array, targets: np.array) -> float:
    """See :func:`.root_mean_absolute_error`."""
    return root_mean_absolute_error(preds, targets)


def mean_absolute_error(preds: np.array, targets: np.array) -> np.array:
    """
    Compute the mean-absolute error (MAE).

    Parameters
    ----------
    preds: np.array
        A two-dimensional array of predictions over the data points.
    targets: np.array
        A two-dimensional array of target variables.

    Returns
    -------
    np.array
        The computed MAE.
    """
    return np.mean(np.abs(preds - targets), axis=1)


def mae(preds: np.array, targets: np.array) -> float:
    """See :func:`.mean_absolute_error`."""
    return mean_absolute_error(preds, targets)


def relative_error(preds: np.array, targets: np.array) -> np.array:
    """
    Compute the relative error.

    Parameters
    ----------
    preds: np.array
        A two-dimensional array of predictions over the data points.
    targets: np.array
        A two-dimensional array of target variables.

    Returns
    -------
    np.array
        The computed relative error.
    """
    return np.linalg.norm((preds - targets), axis=1) / np.linalg.norm(targets, axis=1)


def rel_error(preds: np.array, targets: np.array) -> np.array:
    """See :func:`.relative_error`."""
    return relative_error(preds, targets)


def data_misfit(preds: np.array, targets: np.array, H: np.array) -> np.array:
    """
    Compute the data misfit.

    Parameters
    ----------
    preds: np.array
        A two-dimensional array of predictions over the data points.
    targets: np.array
        A two-dimensional array of target variables.

    Returns
    -------
    np.array
        The computed data misfit.
    """
    return np.linalg.norm((H @ preds.T) - targets.T, axis=1)


def misfit(preds: np.array, targets: np.array, params: dict) -> np.array:
    """See :func:`.data_misfit`."""
    return data_misfit(preds, targets, params)
