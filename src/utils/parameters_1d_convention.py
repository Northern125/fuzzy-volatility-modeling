from numpy import array, concatenate


def unpack_1d_parameters(parameters, p, q, n_clusters):
    alpha_arr_right_end = n_clusters + (n_clusters * q)

    alpha_0 = parameters[:n_clusters]
    alpha = parameters[n_clusters:alpha_arr_right_end].reshape(q, n_clusters)
    beta = parameters[alpha_arr_right_end:].reshape(p, n_clusters)

    return alpha_0, alpha, beta


def pack_1d_parameters(alpha_0, alpha, beta):
    parameters = list(alpha_0) + list(alpha.flatten()) + list(beta.flatten())
    parameters = array(parameters)

    return parameters


def unpack_1d_params_cbc(params: array,
                         p: int,
                         q: int,
                         n_clusters: int
                         ) -> tuple[array, array, array]:
    """
    'cbc' stands for 'cluster by cluster'
    :param params: 1D numpy.array. Cluster-by-cluster stacked parameters
    :param p: int
    :param q: int
    :param n_clusters: int
    :return: tuple(alpha_0, alpha, beta)
    """
    n_params_in_a_rule = p + q + 1

    params_by_clusters = array([params[(i - 1) * n_params_in_a_rule:i * n_params_in_a_rule]
                                for i in range(1, n_clusters + 1)]).copy()

    alpha_0 = array([_params[0] for _params in params_by_clusters]).copy()
    alpha = array([_params[1:q + 1] for _params in params_by_clusters]).T.copy()
    beta = array([_params[q + 1:] for _params in params_by_clusters]).T.copy()

    return alpha_0, alpha, beta


def pack_1d_params_cbc(alpha_0, alpha, beta):
    """

    :param alpha_0:
    :param alpha:
    :param beta:
    :return:
    """
    n_clusters: int = len(alpha_0)

    params = array([concatenate(([alpha_0[i]], alpha[:, i], beta[:, i])) for i in range(n_clusters)]).flatten()

    return params
