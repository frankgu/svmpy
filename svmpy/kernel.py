import numpy as np
import numpy.linalg as la


class Kernel(object):

    type = "linear"
    sigma = 0.0

    @staticmethod
    def linear():
        Kernel.type = "linear"
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(sigma):
        Kernel.type = "gaussian"
        Kernel.sigma = sigma
        return lambda x, y: \
            np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))