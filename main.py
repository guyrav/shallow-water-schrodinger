import numpy as np
import matplotlib.pyplot as plt

from params import ShallowWaterParams
from schemes import NLSESplitStepScheme
from bathymetry import flat_bottom, add_initial_condition, gaussian, single_well, gaussian_well, moving_gaussian_well


def init(params, initial_condition, history=True):
    x = np.linspace(0, params.L, params.nx + 1, endpoint=True)
    t = np.linspace(0, params.T, params.nt + 1, endpoint=True)
    u = initial_condition(x)
    history_array = np.zeros((params.nt + 1, params.nx + 1), dtype=complex)
    if history:
        return x, t, u, history_array
    else:
        return x, t, u


def madelung_transform(eta, phi):
    return np.sqrt(eta) * np.exp(1j * phi)


# def inverse_madelung_transform(psi, nx, dx):
#     eta = np.abs(psi) ** 2
#     phi = np.angle(psi)
#     k = np.fft.fftfreq(nx, dx)
#     u = np.fft.ifft(1j * k * np.fft.fft(phi))  # TODO: Deal with endpoint
#
#     return eta, u


def get_ylims(*args):
    """Compute y-axis bounds for readable plots

    Args:
        args : Arrays of values to be be plotted - can be one or more.

    Return:
        ylims : (list[float]) Min and max values for y-axis
    """
    u_min = min(arg.min() for arg in args)
    u_max = max(arg.max() for arg in args)
    u_mean = (u_min + u_max) / 2.
    alpha = 1.4
    ylims = [alpha * u_min + (1 - alpha) * u_mean, alpha * u_max + (1 - alpha) * u_mean]
    return ylims


def plot_evolution(t: np.ndarray[float], x: np.ndarray[float], eta_0, eta: np.ndarray[float], title: str):
    """Animate and plot the evolution of the solution over time.

    Args:
        t : (np.ndarray[float]) Time labels.
        x : (np.ndarray[float]) Space labels.
        eta_0 : (np.ndarray[float]) Bathymetry.
        eta : (np.ndarray[float]) Surface height at all time steps.
        title : (str) Plot title.
    """
    eta_0_realised = eta_0(np.reshape(x, (1,) + x.shape), np.reshape(t, t.shape + (1,)))
    eta -= eta_0_realised
    ylims = get_ylims(*eta, *[-eta_0(x, t_n) for t_n in t])

    plt.plot(x, eta[0], 'k', label='Initial surface height')
    plt.plot(x, -eta_0(x, 0), 'r', label='Initial bottom height')
    plt.title(title)
    plt.legend(loc='best')
    plt.ylabel('velocity')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim(ylims)
    plt.pause(0.5)

    for t_n, eta_n, eta_0_n in zip(t, eta, eta_0_realised):
        plt.cla()
        plt.plot(x, eta_n, 'b', label=f'Time {t_n:.2f}')
        plt.plot(x, -eta_0_n, 'r', label='Bottom height')
        plt.title(title)
        plt.legend(loc='best')
        plt.ylabel('surface height')
        plt.xlabel('x')
        plt.ylim(ylims)
        plt.pause(0.05)

    plt.show()


def main():
    T = 30
    L = 100
    nt = 300
    nx = 300
    params = ShallowWaterParams(T, L, nt, nx, 1.)
    scheme = NLSESplitStepScheme(params)
    # eta_0 = flat_bottom(1.)
    # eta_0 = moving_gaussian_well(1., 1.5, L / 4, 1., 0.5)
    eta_0 = moving_gaussian_well(1.1, 1., L / 4, 1., 0.5)
    # initial_condition = add_initial_condition(eta_0, gaussian(3 * L / 4, 1.))
    initial_condition = add_initial_condition(eta_0, lambda x: 0)
    x, t, eta_0_realised, history = init(params, initial_condition)
    psi_0 = madelung_transform(eta_0_realised, 0.)  # Assuming starting from rest
    _ = scheme(psi_0, eta_0=eta_0, history=history)

    # eta, _ = inverse_madelung_transform(history, nx, params.dx)
    eta = np.abs(history) ** 2
    plot_evolution(t, x, eta_0, eta, "Surface height evolution with flat bottom and initial gaussian")


if __name__ == '__main__':
    main()
