import numpy as np
from scipy.linalg import solve, expm


class NLinkPendulum:
    def __init__(self, n, lengths, masses, cart_mass=None, g=9.81):
        self.n = n  # number of links
        self.L = np.asarray(lengths, dtype=float)
        self.m = np.asarray(masses, dtype=float)
        self.cart_mass = cart_mass
        self.g = g
        self.has_cart = cart_mass is not None
        self.nq = n + (1 if self.has_cart else 0)
        self.nx = 2 * self.nq
        self.nu = 1

    def _mass_matrix(self, q):
        n, L, m = self.n, self.L, self.m
        off = 1 if self.has_cart else 0
        theta = q[off : off + n]
        M = np.zeros((self.nq, self.nq))

        if self.has_cart:
            M[0, 0] = self.cart_mass + np.sum(m)
            for i in range(n):
                val = np.sum(m[i:]) * L[i] * np.cos(theta[i])
                M[0, i + 1] = val
                M[1 + i, 0] = val

        for i in range(n):
            for j in range(n):
                M[off + i, off + j] = (
                    np.sum(m[max(i, j) :]) * L[i] * L[j] * np.cos(theta[i] - theta[j])
                )

        return M

    def _coriolis_gravity(self, q, dq):
        n, L, m, g = self.n, self.L, self.m, self.g
        off = 1 if self.has_cart else 0
        theta = q[off : off + n]
        f = np.zeros(self.nq)

        for i in range(n):
            f[off + i] -= np.sum(m[i:]) * g * L[i] * np.sin(theta[i])

            for j in range(n):
                if i != j:
                    f[off + i] -= (
                        np.sum(m[max(i, j) :])
                        * L[i]
                        * L[j]
                        * np.sin(theta[i] - theta[j])
                        * theta[j] ** 2
                    )

        return f

    def xdot(self, x, u):
        q = x[: self.nq]
        dq = x[self.nq :]
        M = self._mass_matrix(q)
        f = self._coriolis_gravity(q, dq)
        B = np.zeros(self.nq)
        B[0] = 1.0
        ddq = solve(M, f + B * u, assume_a="sym")
        return np.concatenate([dq, ddq])


if __name__ == "__main__":
    p = NLinkPendulum(n=2, lengths=[0.5, 0.4], masses=[0.3, 0.2], cart_mass=1.0)

    # test __init__
    print("--- init test ---")
    print(f"n:  {p.n}")
    print(f"nq: {p.nq}")
    print(f"nx: {p.nx}")
    print(f"nu: {p.nu}")

    # test _mass_matrix
    print("\n--- mass matrix test ---")
    q = np.zeros(p.nq)
    M = p._mass_matrix(q)
    print(M)
    print("Symmetric?", np.allclose(M, M.T))
    print("Positive definite?", np.all(np.linalg.eigvals(M) > 0))

    # test _coriolis_gravity
    print("\n--- coriolis gravity test ---")
    dq = np.zeros(p.nq)
    f = p._coriolis_gravity(q, dq)
    print("Zero velocity:", f)
    q[1] = 0.1
    f = p._coriolis_gravity(q, dq)
    print("Small angle, zero vel:", f)
    q = np.zeros(p.nq)
    dq = np.zeros(p.nq)
    q[1] = 0.3
    q[2] = 0.1
    dq[2] = 1.0
    f = p._coriolis_gravity(q, dq)
    print("Nonzero angle and vel:", f)

    # test xdot
    print("\n--- xdot test ---")
    x0 = np.zeros(p.nx)
    x0[1] = 0.1
    xd = p.xdot(x0, u=0)
    print("state derivative:", xd)
    print("velocities:", xd[: p.nq])
    print("accelerations:", xd[p.nq :])
