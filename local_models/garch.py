from rules_related import combine_rules_outputs
from numpy import array
import logging


def calc_ht(alpha_0, alpha, beta, y_squared, h):
    res = alpha_0 + (alpha * y_squared).sum() + (beta * h).sum()
    return res


def calc_fuzzy_ht(alpha_0, alpha, beta, y_squared, h, weights):
    n_clusters = weights.shape[0]

    outputs = [
        calc_ht(alpha_0[j], alpha[:, j], beta[:, j], y_squared, h)
        for j in range(n_clusters)
    ]
    outputs = array(outputs)

    result = combine_rules_outputs(outputs, weights)

    return result


def _calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                   calc_ht_function: callable = calc_ht, **kwargs) -> array:
    logger = logging.getLogger('calc_cond_var_fuzzy')

    q = len(alpha)
    p = len(beta)

    starting_index = max(p, q)
    logger.debug(f'starting_index = {starting_index}')

    if len(first_h) < starting_index:
        raise Exception(f'Not enough first elements of h are given. p = {p}, q = {q}, max(p, q) = {starting_index}. '
                        f'Therefore, we need first {starting_index} elements of h known. '
                        f'However only {len(first_h)} are given: given first_h = {first_h}')

    h = list(first_h)

    y_len = len(y_squared)
    logger.debug(f'y_len = {y_len}')

    for i in range(starting_index, y_len + 1):
        y_slc = slice(i - q, i)
        h_slc = slice(i - p, i)
        h_t = calc_ht_function(alpha_0, alpha, beta, y_squared[y_slc], h[h_slc], **kwargs)
        h.append(h_t)
        logger.debug(f'New iteration; i = {i}: h_t = {h_t}, y_slc = {y_slc}, h_slc = {h_slc}, '
                     f'h[h_slc] = {h[h_slc]}, y_squared[y_slc] =\n{y_squared[y_slc]}')

    h = array(h)

    return h


def calc_cond_var_vanilla(alpha_0, alpha, beta, y_squared, first_h) -> array:
    return _calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                          calc_ht_function=calc_ht)


def calc_cond_var_fuzzy(alpha_0, alpha, beta, y_squared, first_h, weights) -> array:
    return _calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                          weights=weights,
                          calc_ht_function=calc_fuzzy_ht)
