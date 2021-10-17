from scipy.stats._multivariate import multivariate_normal_frozen
import numpy as np
import logging


class LongMultivariateNormal(multivariate_normal_frozen):
    def pdf(self, x, mean=None, cov=1, allow_singular=False):
        logger = logging.getLogger('pdf')

        log_pdf = self.logpdf(x)
        logger.debug(f'log_pdf = {log_pdf}, type(log_pdf) = {type(log_pdf)}')

        long_log_pdf = np.longdouble(log_pdf)
        logger.debug(f'long_log_pdf = {long_log_pdf}, type(long_log_pdf) = {type(long_log_pdf)}')

        result = np.exp(long_log_pdf)

        return result
