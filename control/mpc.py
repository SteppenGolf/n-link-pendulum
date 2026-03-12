import numpy as np
import casadi as ca
import sys
import os
from scipy.linalg import solve_discrete_are
sys.path.insert(0, "..")
from dynamics.pendulum import NLinkPendulum
import numpy as np
class MPC:
    def __init__(self, Ad, Bd, Q, R, N, u_min, u_max):
        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R
        self.N = N
        self.u_min = u_min
        self.u_max = u_max

    def setup(self):
        nx = self.Ad.shape[0]
        nu = self.Bd.shape[1]
        N  = self.N

        # compute LQR terminal cost
        P_terminal = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)

        X = ca.MX.sym('X', nx, N + 1)
        U = ca.MX.sym('U', nu, N)
        P = ca.MX.sym('P', nx)

        cost = 0
        constraints = []

        # fix initial state
        constraints.append(X[:, 0] - P)

        for k in range(N):
            cost += X[:, k].T @ self.Q @ X[:, k] + U[:, k].T @ self.R @ U[:, k]
            x_next = self.Ad @ X[:, k] + self.Bd @ U[:, k]
            constraints.append(X[:, k+1] - x_next)

        # LQR terminal cost instead of simple Q
        cost += X[:, N].T @ P_terminal @ X[:, N]

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        constraints = ca.vertcat(*constraints)

        nlp = {'x': opt_vars, 'f': cost, 'g': constraints, 'p': P}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.nx = nx
        self.nu = nu

    def control(self, x):
        nx, nu, N = self.nx, self.nu, self.N

        x0_opt = np.zeros(nx * (N + 1) + nu * N)

        lbx = [-np.inf] * (nx * (N + 1)) + [self.u_min] * (nu * N)
        ubx = [np.inf]  * (nx * (N + 1)) + [self.u_max] * (nu * N)

        # initial state constraint + dynamics constraints = 0
        lbg = [0] * (nx * (N + 1))
        ubg = [0] * (nx * (N + 1))

        sol = self.solver(
            x0=x0_opt,
            lbx=lbx, ubx=ubx,
            lbg=lbg, ubg=ubg,
            p=x
        )

        opt_vars = sol['x'].full().flatten()
        u = opt_vars[nx * (N + 1) : nx * (N + 1) + nu]

        return u 
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from dynamics.pendulum import NLinkPendulum
    from simulation.simulator import Simulator
    from scipy.linalg import solve_discrete_are
    import numpy as np

    dt = 0.01
    p = NLinkPendulum(n=2, lengths=[0.5, 0.4], masses=[0.3, 0.2], cart_mass=1.0)
    A, B = p.linearize()
    Ad, Bd = p.discretize(A, B, dt)

    Q = np.eye(p.nx)
    R = 10 * np.eye(p.nu)

    mpc = MPC(Ad, Bd, Q, R, N=20, u_min=-20.0, u_max=20.0)
    mpc.setup()

    sim = Simulator(p, mpc)
    x0 = np.zeros(p.nx)
    x0[1] = 0.05

    # debug 3 steps
    x = x0.copy()
    for i in range(3):
        u = mpc.control(x)
        print(f"step {i}: u={u}, x={np.round(x, 4)}")
        x = p.step_rk4(x, u, dt)

    # full simulation
    x_hist, u_hist = sim.run(x0, dt, steps=300)
    print("Initial state:", np.round(x_hist[0], 4))
    print("Final state:  ", np.round(x_hist[-1], 4))
    print("Stabilized?", np.linalg.norm(x_hist[-1]) < 0.5)