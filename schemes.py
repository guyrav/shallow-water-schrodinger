from typing import Callable

import numpy as np

from params import NumericalSchemeParams, ShallowWaterParams


class Scheme(object):
    """Base class for numerical schemes for solving PDEs.

    Methods:
        __call__(): (virtual) Run the scheme to compute the time evolution of a PDE solution.
        minimal_stable_nt(): (virtual) Compute the minimal number of time steps needed for numerical stability.
        __str__(): (virtual) Print the scheme description.
    """
    def __init__(self, params: NumericalSchemeParams):
        self.params = params

    def __call__(self, initial_condition: np.ndarray[complex], history=None, *args, **kwargs):
        """Evolve the initial condition over time using the numerical scheme.

        Args:
            initial_condition: (ndarray[complex]) Initial condition as a sequence of values.
            history: (ndarray[complex], optional) Table of size (nt + 1) * nx or (nt + 1) * (nx + 1)
                                                to hold all values over time in.

        Note:
            A length of nx + 1 in space is taken to imply that the boundary point is duplicated
            and appears at both ends, as in a periodic boundary.
        """
        pass

    def __str__(self):
        return "Unknown scheme"


class SingleTimeStepScheme(Scheme):
    """Base class for numerical schemes that use only the values at time n to compute the values at time n+1.

    Methods:
        _apply_step(): (virtual) Apply one time step, putting the output back in the current state.
    """
    def _apply_step(self, current_state: np.ndarray[complex], t: float, *args, **kwargs):
        pass

    def __call__(self, initial_condition: np.ndarray[complex], history=None, *args, **kwargs):
        has_padding = (initial_condition.shape[0] > self.params.nx)  # Assumed either nx or nx + 1

        u = np.zeros(self.params.nx, dtype=complex)
        u[:] = initial_condition[:-1] if has_padding else initial_condition[:]

        store_u = (history is not None)

        if store_u:
            history[0] = initial_condition

        for n in range(self.params.nt):
            t = self.params.t(n)
            self._apply_step(u, t, *args, **kwargs)

            if store_u:
                history[n + 1] = np.pad(u, [(0, 1)], mode='wrap') if has_padding else u

        return np.pad(u, [(0, 1)], mode='wrap') if has_padding else u


class NLSESplitStepScheme(SingleTimeStepScheme):
    def __init__(self, params: ShallowWaterParams):
        super().__init__(params)
        self.x = np.linspace(params.x_start, params.x_end, params.nx, endpoint=False)
        self.k = 2 * np.pi * np.fft.fftfreq(params.nx, params.dx)
        self.linear_factor = np.exp(-0.5j * params.dt * self.k**2)

    def _lagrangian_step(self, psi: np.ndarray[complex]):
        psi[:] = np.fft.ifft(self.linear_factor * np.fft.fft(psi))

    def _nonlinear_step(self, psi, eta_0, t, dt):
        t_mid = t + 0.25 * dt
        eta_0_mid = eta_0(self.x, t_mid)
        phase_factor = np.exp(-1j * (dt / 2.0) / self.params.F * (np.abs(psi) ** 2 - eta_0_mid))
        psi *= phase_factor

    def _apply_step(self, current_state: np.ndarray[complex], t: float, eta_0, *args, **kwargs):
        self._nonlinear_step(current_state, eta_0, t, self.params.dt / 2)
        self._lagrangian_step(current_state)
        self._nonlinear_step(current_state, eta_0, t + self.params.dt / 2, self.params.dt / 2)
