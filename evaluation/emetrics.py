import numpy as np

def r_squared_error(y_obs, y_pred):
    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)

    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)

    num = np.sum((y_obs - y_obs_mean) *
                 (y_pred - y_pred_mean))
    num = num ** 2

    den = (np.sum((y_obs - y_obs_mean) ** 2) *
           np.sum((y_pred - y_pred_mean) ** 2))

    return num / den

def get_k(y_obs, y_pred):
    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)

    return np.sum(y_obs * y_pred) / np.sum(y_pred ** 2)


def squared_error_zero(y_obs, y_pred):
    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)

    k = get_k(y_obs, y_pred)

    numerator = np.sum((y_obs - k * y_pred) ** 2)
    denominator = np.sum((y_obs - np.mean(y_obs)) ** 2)

    return 1 - numerator / denominator

def get_rm2(y_obs, y_pred):
    r2 = r_squared_error(y_obs, y_pred)
    r02 = squared_error_zero(y_obs, y_pred)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


