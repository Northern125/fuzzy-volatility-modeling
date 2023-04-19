from numpy import array, reshape


def update_parameters(params_prev: array,
                      cov_prev: array,
                      y_new: float,
                      coeffs_prev: array
                      ) -> tuple:
    """
    The equation is as follows: `y_{t + 1} = psi_t^T * theta_t`. At time point `t` we have `y_t`, `psi_{t-1}`,
    `theta_{t-1}`, `C_{t-1}` and we estimate `theta_t`.
    `psi_t` is a coefficients column vector
    `theta_t` is an unknown parameters column vector
    `C_t` is a covariance matrix
    :param params_prev: 1D numpy.array. `theta_{t-1}`
    :param cov_prev: 2D numpy.array. `C{t-1}`
    :param y_new: float. `y_t`
    :param coeffs_prev: 1D numpy.array. `psi{t-1}`
    :return: tuple (`C_t`, `theta_t`)
    """
    params_prev = reshape(params_prev, (-1, 1)).copy()
    coeffs_prev = reshape(coeffs_prev, (-1, 1)).copy()

    cov_new = \
        cov_prev - (cov_prev @ params_prev @ params_prev.T @ cov_prev) / (1 + params_prev.T @ cov_prev @ params_prev)

    params_new = params_prev + cov_new @ coeffs_prev * (y_new - coeffs_prev.T @ params_prev)
    params_new = params_new.flatten().copy()

    return cov_new, params_new
