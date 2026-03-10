import numpy as np
from scipy.linalg import solve_discrete_are


class LQR:
    def __init__(self, Ad, Bd, Q, R):
        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R

    def compute_gain(self):
        A, B, Q, R = self.Ad, self.Bd, self.Q, self.R

        P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(B.T @ P @ B + R) @ B.T @ P @ A

    def control(self, x):
        u = -self.K @ x
        return u


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "..")
    from dynamics.pendulum import NLinkPendulum
    import numpy as np

    dt = 0.01
    p = NLinkPendulum(n=2, lengths=[0.5, 0.4], masses=[0.3, 0.2], cart_mass=1.0)
    A, B = p.linearize()
    Ad, Bd = p.discretize(A, B, dt)
    Q = np.eye(p.nx)
    R = np.eye(p.nu)
    lqr = LQR(Ad, Bd, Q, R)
    lqr.compute_gain()

    print("K:", np.round(lqr.K, 3))
    eigvals_cl = np.linalg.eigvals(Ad - Bd @ lqr.K)
    print("Stable?", np.all(np.abs(eigvals_cl) < 1))
