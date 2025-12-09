class NumericalSchemeParams(object):
    """General numerical scheme parameters.

    Attributes:
        T: (float) Length of time interval
        L: (float) Length of space interval
        nt: (int) Number of time steps, including initial condition
        nx: (int) Number of points in space, start and end count together as one point
        dt: (float) Time resolution
        dx: (float) Spatial resolution
        x_start: (float) Left end of space interval, default is 0
        x_end: (float) Right end of space interval
        t_start: (float) Initial time, default is 0
        t_end: (float) Final time 
    """

    def __init__(self, T, L, nt, nx, *args, x_start=0, t_start=0, **kwargs):
        """Initialize scheme parameters.

        Args:
            T: (float) Length of time interval
            L: (float) Length of space interval
            nt: (int) Number of time steps, not including initial condition
            nx: (int) Number of points in space, start and end count together as one point
            x_start: (float) Left end of space interval, default is 0
            t_start: (float) Initial time, default is 0
        """
        self.T = float(T)
        self.L = float(L)
        self.nt = nt
        self.nx = nx
        self.dt = self.T / nt
        self.dx = self.L / nx
        self.x_start = x_start
        self.t_start = t_start
        self.x_end = self.x_start + self.L
        self.t_end = self.t_start + self.T


    def t(self, n: int) -> float:
        """Return time at time step n (0-based)
        Args:
            n: (int) Number of time steps
        Returns:
            float: Time at time step n
        """
        return self.t_start + n * self.dt


class ShallowWaterParams(NumericalSchemeParams):
    """Parmeters for numerical schemes solving the shallow-water equation.

    Extends the ViscousParams class.

    Additional attribute:
        F: (float) Constant Froude number.
    """

    def __init__(self, T, L, nt, nx, F, *args, **kwargs):
        super().__init__(T, L, nt, nx, *args, **kwargs)
        self.F = F
