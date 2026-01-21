import numpy as np


def moving(f):
    def ret(*args, **kwargs):
        v = kwargs.pop('velocity', 0)
        b = f(*args, **kwargs)
        return lambda x, t: b(x - v * t, t)

    return ret


def accelerating(f):
    def ret(*args, **kwargs):
        v_0 = kwargs.pop('v_0',0)
        a = kwargs.pop('acceleration', 0)
        b = f(*args, **kwargs)
        return lambda x, t: b(x - v_0 * t - a / 2 * t**2, t)

    return ret


def flat_bottom(depth: float):
    return lambda x, t: depth * np.ones_like(x)


@moving
def single_well(min_depth: float, max_depth: float, position, width):
    return lambda x, t: min_depth + (max_depth - min_depth) * np.logical_and(x - position < width / 2, x - position > -width / 2)


@moving
def gaussian_well(min_depth: float, max_depth: float, position, sigma):
    return lambda x, t: min_depth + (max_depth - min_depth) * np.exp(- (x - position) ** 2 / (2 * sigma ** 2))


@moving
def two_wells(min_depth: float, max_depth: float, position1, position2, sigma):
    def ret(x, t):
        return gaussian_well(min_depth, max_depth, position1, sigma)(x, t) + gaussian_well(min_depth, max_depth, position2, sigma)(x, t) - max_depth

    return ret

@moving
def two_diff_wells(min_depth1: float, max_depth1: float, min_depth2: float, max_depth2: float, position1, position2, sigma):
    def ret(x, t):
        return gaussian_well(min_depth1, max_depth1, position1, sigma)(x, t) + gaussian_well(min_depth2, max_depth2, position2, sigma)(x, t) - max(max_depth1, max_depth2)

    return ret


@accelerating
def acc_two_diff_wells(min_depth1: float, max_depth1: float, min_depth2: float, max_depth2: float, position1, position2, sigma):
    def ret(x, t):
        return gaussian_well(min_depth1, max_depth1, position1, sigma)(x, t) + gaussian_well(min_depth2, max_depth2, position2, sigma)(x, t) - max(max_depth1, max_depth2)

    return ret

@moving
def sine_wave(A, offset, L, k):
    def ret(x, t):
        return A * np.sin(2 * np.pi * k * x / L) + offset

    return ret

@accelerating
def accelerating_sine_wave(A, offset, L, k):
    def ret(x, t):
        return A * np.sin(2 * np.pi * k * x / L) + offset

    return ret

def add_initial_condition(bathymetry, initial_condition):
    return lambda x: bathymetry(x, 0) + initial_condition(x)


@moving
def gaussian(mu, sigma):
    def ret(x):
        return 1 / np.sqrt(2 * np.pi * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    return ret


def accelerating_gaussian(min_depth: float, max_depth: float, position, sigma, v_0, a):
    def ret(x, t):
        return gaussian_well(min_depth, max_depth, position, sigma)(x - v_0 * t - a / 2 * t**2, t)

    return ret
