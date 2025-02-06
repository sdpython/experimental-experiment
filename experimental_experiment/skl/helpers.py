import torch


def flatnonzero(x):
    "Similar to :func:`numpy.flatnonzero`"
    return torch.nonzero(torch.reshape(x, (-1,)), as_tuple=True)[0]


def check_non_negative(array, whom):
    assert array.min() >= 0, f"{whom} has passed a negative value."


def _num_samples(x):
    return len(x)


def _get_weights(dist, weights):
    """Get the weights from an array of distances and a parameter ``weights``.

    Assume weights have already been validated.

    Parameters
    ----------
    dist : ndarray
        The input distances.

    weights : {'uniform', 'distance'}, callable or None
        The kind of weighting used.

    Returns
    -------
    weights_arr : array of the same shape as ``dist``
        If ``weights == 'uniform'``, then returns None.
    """
    if weights in (None, "uniform"):
        return None

    if weights == "distance":
        # if user attempts to classify a point that was zero distance from one
        # or more training points, those training points are weighted as 1.0
        # and the other points as 0.0
        dist = 1.0 / dist
        inf_mask = torch.isinf(dist)
        inf_row = torch.any(inf_mask, axis=1)
        dist[inf_row] = inf_mask[inf_row]
        return dist

    if callable(weights):
        return weights(dist)


def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    Y_dtype = X.dtype if Y is None else Y.dtype
    dtype = X.dtype if X.dtype == Y_dtype == torch.float32 else torch.float64
    X = X.to(dtype)
    Y = Y.to(dtype)
    return X, Y, dtype


def _check_estimator_name(estimator):
    if estimator is not None:
        if isinstance(estimator, str):
            return estimator
        else:
            return estimator.__class__.__name__
    return None
