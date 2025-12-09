import numpy as np


def flat_bottom(depth: float):
    return lambda x, t: depth * np.ones_like(x)


def single_well(min_depth: float, max_depth: float, position, width):
    return lambda x, t: min_depth + (max_depth - min_depth) * np.logical_and(x - position < width / 2, x - position > -width / 2)


def gaussian_well(min_depth: float, max_depth: float, position, sigma):
    return lambda x, t: min_depth + (max_depth - min_depth) * np.exp(- (x - position) ** 2 / (2 * sigma ** 2))


def moving_gaussian_well(min_depth: float, max_depth: float, position, sigma, velocity):
    def ret(x, t):
        return min_depth + (max_depth - min_depth) * np.exp(- (x - velocity * t - position) ** 2 / (2 * sigma ** 2))

    return ret


def add_initial_condition(bathymetry, initial_condition):
    return lambda x: bathymetry(x, 0) + initial_condition(x)


def gaussian(mu, sigma):
    def ret(x):
        return 1 / np.sqrt(2 * np.pi * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    return ret
