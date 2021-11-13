def calc_ht(alpha_0, alpha, beta, epsilon, h):
    epsilon_squared = (epsilon ** 2).copy()
    res = alpha_0 + (alpha * epsilon_squared).sum() + (beta * h).sum()
    return res
