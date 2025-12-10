import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Non-commuting vector fields
# ------------------------------------------------------------
def F(x):
    res = np.array([- x[1], np.cos(x[0])])
    return res / np.sqrt(np.sum(res**2))  # horizontal shear

def G(x):
    res = np.array([- x[1] - 0.3 * x[0], x[0] - 0.3 * x[1]]) - F(x)
    return res
    # return 0.8 * np.array([0.5 + x[1] - x[0], - 0.3 * x[1]])  # vertical shear

def H(x):
    return F(x) + G(x)

# ------------------------------------------------------------
# RK4 step
# ------------------------------------------------------------
def rk4_step(F, x, dt):
    k1 = F(x)
    k2 = F(x + 0.5*dt*k1)
    k3 = F(x + 0.5*dt*k2)
    k4 = F(x + dt*k3)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

def integrate_flow(F, x0, dt, n):
    traj = np.zeros((n+1, 2))
    traj[0] = x0
    x = x0.copy()
    for k in range(n):
        x = rk4_step(F, x, dt)
        traj[k+1] = x
    return traj

# ------------------------------------------------------------
# Lie and Strang segments
# ------------------------------------------------------------
def lie_segments(F, G, x0, dt, n):
    segments = []
    x = x0.copy()
    for k in range(n):
        # F step
        x1 = rk4_step(F, x, dt)
        segments.append((x.copy(), x1.copy(), 'F', False))
        x = x1
        # G step = macro endpoint
        x1 = rk4_step(G, x, dt)
        segments.append((x.copy(), x1.copy(), 'G', True))
        x = x1
    return segments

def strang_segments(F, G, x0, dt, n):
    segments = []
    x = x0.copy()
    for k in range(n):
        # F half
        x1 = rk4_step(F, x, dt/2)
        segments.append((x.copy(), x1.copy(), 'F', False))
        x = x1
        # G full
        x1 = rk4_step(G, x, dt)
        segments.append((x.copy(), x1.copy(), 'G', False))
        x = x1
        # F half = macro endpoint
        x1 = rk4_step(F, x, dt/2)
        segments.append((x.copy(), x1.copy(), 'F', True))
        x = x1
    return segments

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
x0 = np.array([-2, -2])
dt = 0.4
N = 11
T = dt*N
traj_true = integrate_flow(H, x0, dt/10, int(T/(dt/10)))

segs_lie = lie_segments(F, G, x0, dt, N)
segs_str = strang_segments(F, G, x0, dt, N)

# ------------------------------------------------------------
# Vector field grid
# ------------------------------------------------------------
X, Y = np.meshgrid(np.linspace(-2.8, 2.8, 15),
                   np.linspace(-2.8, 2.8, 15))
UF = np.zeros_like(X)
VF = np.zeros_like(Y)
UG = np.zeros_like(X)
VG = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        p = np.array([X[i,j], Y[i,j]])
        UF[i,j], VF[i,j] = F(p)
        UG[i,j], VG[i,j] = G(p)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

for ax in axes:
    # Vector fields
    qF = ax.quiver(X, Y, UF, VF, color='C0', alpha=0.35, label='F')
    qG = ax.quiver(X, Y, UG, VG, color='C1', alpha=0.35, label='G')
    # True flow
    t_line, = ax.plot(traj_true[:,0], traj_true[:,1], 'k-', lw=2, label='Flow of F+G (reference)')

    ax.set_aspect('equal')
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# --- Lie ---
ax = axes[0]
# Dummy handles for legend
hF, = ax.plot([],[],'-C2',lw=2,label='Step along F')
hG, = ax.plot([],[],'-C3',lw=2,label='Step along G')
for xs, xe, which, endpoint in segs_lie:
    color = 'C2' if which=='F' else 'C3'
    ax.plot([xs[0],xe[0]],[xs[1],xe[1]],color=color,lw=2)
    if endpoint:
        ax.scatter(xe[0],xe[1],color='C3',s=40,zorder=5)
ax.set_title('Lie splitting')
ax.legend(handles=[qF,qG,t_line,hF,hG],loc='upper right')

# --- Strang ---
ax = axes[1]
hF, = ax.plot([],[],'-C2',lw=2,label='step along F')
hG, = ax.plot([],[],'-C3',lw=2,label='step along G')
for xs, xe, which, endpoint in segs_str:
    color = 'C2' if which=='F' else 'C3'
    ax.plot([xs[0],xe[0]],[xs[1],xe[1]],color=color,lw=2)
    if endpoint:
        ax.scatter(xe[0],xe[1],color='C2',s=40,zorder=5)
ax.set_title('Strang splitting')
ax.legend(handles=[qF,qG,t_line,hF,hG],loc='upper right')

plt.tight_layout()
plt.show()
