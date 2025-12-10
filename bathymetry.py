import numpy as np


def moving(f):
    def ret(*args, **kwargs):
        v = kwargs.pop('velocity', 0)
        b = f(*args, **kwargs)
        return lambda x, t: b(x - v * t, t)

    return ret


def flat_bottom(depth: float):
    return lambda x, t: depth * np.ones_like(x)


def single_well(min_depth: float, max_depth: float, position, width):
    return lambda x, t: min_depth + (max_depth - min_depth) * np.logical_and(x - position < width / 2, x - position > -width / 2)


@moving
def gaussian_well(min_depth: float, max_depth: float, position, sigma):
    return lambda x, t: min_depth + (max_depth - min_depth) * np.exp(- (x - position) ** 2 / (2 * sigma ** 2))


def two_wells(min_depth: float, max_depth: float, position1, position2, sigma, velocity):
    def ret(x, t):
        return moving_gaussian_well(min_depth, max_depth, position1, sigma, velocity)(x, t) + moving_gaussian_well(min_depth, max_depth, position2, sigma, velocity)(x, t) - max_depth

    return ret


def moving_gaussian_well(min_depth: float, max_depth: float, position, sigma, velocity):
    def ret(x, t):
        return min_depth + (max_depth - min_depth) * np.exp(- (x - velocity * t - position) ** 2 / (2 * sigma ** 2))

    return ret


def moving_sine_wave(A, offset, L, k, velocity):
    def ret(x, t):
        return A * np.sin(2 * np.pi * k * (x - velocity * t) / L) + offset

    return ret


def add_initial_condition(bathymetry, initial_condition):
    return lambda x: bathymetry(x, 0) + initial_condition(x)


def gaussian(mu, sigma):
    def ret(x):
        return 1 / np.sqrt(2 * np.pi * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    return ret
