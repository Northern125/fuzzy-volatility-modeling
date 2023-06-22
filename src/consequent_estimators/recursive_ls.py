from numpy import array, reshape, dot, concatenate, eye, block, zeros


def re_estimate_params_plain_rls(params_prev: array,
                                 cov_prev: array,
                                 y_new: float,
                                 coeffs_prev: array
                                 ) -> tuple[array, array]:
    """
    Re-estimate the parameters of a linear model at time instant `t`.
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
        cov_prev - (cov_prev @ coeffs_prev @ coeffs_prev.T @ cov_prev) / (1 + coeffs_prev.T @ cov_prev @ coeffs_prev)

    params_new = params_prev + cov_new @ coeffs_prev * (y_new - coeffs_prev.T @ params_prev)
    params_new = params_new.flatten().copy()

    return cov_new, params_new


def ets_new_cluster_re_estimate_parameters(params_prev: array,
                                           cov_prev: array,
                                           weights: array,
                                           n_params_in_a_rule: int,
                                           omega: float
                                           ) -> tuple[array, array]:
    """
    Re-estimate parameters of a model (same model as in `update_parameters()`) given that a new cluster is added.
    This procedure is synced with the eTS model (the algorithm was proposed in Angelov, Filev (2004)).
    For more details on the model and notations look `update_parameters()` docstring
    :param params_prev: 1D numpy.array. `theta_{t-1}`
    :param cov_prev: 2D numpy.array. `C{t-1}`
    :param weights: 1D numpy.array. The weights of the fuzzy system at the previous time step `t - 1`
    :param n_params_in_a_rule: int. The # of parameters in each rule
    :param omega: float. `Omega`, a big number via which the new part of the covariance matrix is initialized
    :return: tuple (`C_t`, `theta_t`)
    """
    weights = (weights / weights.sum()).copy()

    if len(params_prev) % n_params_in_a_rule != 0:
        raise ValueError('The length of `params_prev` should be a multiple of `n_params_in_a_rule`; got '
                         f'`params_prev = {params_prev}`, `n_params_in_a_rule = {n_params_in_a_rule}`')

    n_clusters = int(len(params_prev) / n_params_in_a_rule)
    params_by_clusters = array([params_prev[(i - 1) * n_params_in_a_rule:i * n_params_in_a_rule]
                                for i in range(1, n_clusters + 1)]).copy()

    # parameters
    new_cluster_params = dot(weights, params_by_clusters).copy()
    params_new = concatenate((params_prev, new_cluster_params)).copy()

    # covariance matrix
    new_cluster_cov = (eye(n_params_in_a_rule) * omega).copy()
    cov_new = block([[cov_prev, zeros((cov_prev.shape[0], new_cluster_cov.shape[1]))],
                     [zeros((new_cluster_cov.shape[0], cov_prev.shape[1])), new_cluster_cov]
                     ]).copy()

    return cov_new, params_new


def initialize_rls_params(n_clusters: int,
                          n_params_in_a_rule: int,
                          omega: float
                          ) -> tuple[array, array]:
    cov: array = eye(n_clusters * n_params_in_a_rule) * omega
    params: array = array([0 for _ in range(n_clusters * n_params_in_a_rule)])

    return cov, params
