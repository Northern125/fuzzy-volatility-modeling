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


def calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                  fuzzy=False, weights=None):
    logger = logging.getLogger('calc_cond_var')

    if fuzzy and weights is None:
        raise Exception('fuzzy is True, but weights not provided')

    def _calc_ht(_alpha_0, _alpha, _beta, _y_squared, _h):
        if fuzzy:
            fun = calc_fuzzy_ht(_alpha_0, _alpha, _beta, _y_squared, _h, weights)
        else:
            fun = calc_ht(_alpha_0, _alpha, _beta, _y_squared, _h)

        return fun

    q = len(alpha)
    p = len(beta)

    starting_index = max(p, q)

    if len(first_h) < starting_index:
        raise Exception(f'Not enough first elements of h are given. p = {p}, q = {q}, max(p, q) = {starting_index}. '
                        f'Therefore, we need first {starting_index} elements of h known. '
                        f'However only {len(first_h)} are given: given first_h = {first_h}')

    h = list(first_h)

    y_len = len(y_squared)

    for i in range(starting_index, y_len):
        y_slc = slice(i - q, i)
        h_slc = slice(i - p, i)
        h_t = _calc_ht(alpha_0, alpha, beta, y_squared[y_slc], h[h_slc])
        h.append(h_t)

    return h
