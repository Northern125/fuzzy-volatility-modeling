from rules_related import combine_rules_outputs
from numpy import array
import logging


def calc_ht(alpha_0, alpha, beta, y_squared, h):
    res = alpha_0 + (alpha * y_squared).sum() + (beta * h).sum()
    return res


def calc_fuzzy_ht(alpha_0, alpha, beta, y_squared, h, weights):
    outputs = []
    n_clusters = weights.shape[0]

    for j in range(n_clusters):
        local_output = calc_ht(alpha_0[j], alpha[:, j], beta[:, j], y_squared, h[:, j])
        outputs.append(local_output)

    outputs = array(outputs)

    result = combine_rules_outputs(outputs, weights)

    return result


def calc_cond_var(alpha_0, alpha, beta, y, first_h):
    logger = logging.getLogger('calc_cond_var')

    q = len(alpha)
    p = len(beta)

    starting_index = max(p, q)

    if len(first_h) < starting_index:
        logger.error(f'Not enough first elements of h are set. p = {p}, q = {q}, max(p, q) = {starting_index}. '
                     f'Therefore, we need first {starting_index} elements of h known. '
                     f'However only {len(first_h)} are given: given first_h = {first_h}. '
                     f'Returning None')
        return

    h = list(first_h)

    y_len = len(y)
    y_squared = y ** 2

    for i in range(starting_index, y_len):
        y_slc = slice(i - q, i)
        h_slc = slice(i - p, i)
        h_t = calc_ht(alpha_0, alpha, beta, y_squared[y_slc], h[h_slc])
        h.append(h_t)

    return h
