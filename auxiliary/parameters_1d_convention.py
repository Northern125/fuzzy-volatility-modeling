from numpy import array


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
