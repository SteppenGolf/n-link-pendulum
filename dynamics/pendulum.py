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


if __name__ == "__main__":
    p = NLinkPendulum(n=2, lengths=[0.5, 0.4], masses=[0.3, 0.2], cart_mass=1.0)
    q = np.zeros(p.nq)
    M = p._mass_matrix(q)
    print(M)
    print("Symmetric?", np.allclose(M, M.T))
    print("Positive definite?", np.all(np.linalg.eigvals(M) > 0))
