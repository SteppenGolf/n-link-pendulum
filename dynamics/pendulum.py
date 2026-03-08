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
        dtheta = dq[off : off + n]
        f = np.zeros(self.nq)

        for i in range(n):
            f[off + i] += np.sum(m[i:]) * g * L[i] * np.sin(theta[i])

            for j in range(n):
                if i != j:
                    f[off + i] -= (
                        np.sum(m[max(i, j) :])
                        * L[i]
                        * L[j]
                        * np.sin(theta[i] - theta[j])
                        * dtheta[j] ** 2
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

    def step_rk4(self, x, u, dt):
        k1 = self.xdot(x, u)
        k2 = self.xdot(x + dt / 2 * k1, u)
        k3 = self.xdot(x + dt / 2 * k2, u)
        k4 = self.xdot(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def linearize(self, x_eq=None, u_eq=0.0, eps=1e-5):
        if x_eq is None:
            x_eq = np.zeros(self.nx)  # ← only this inside if

        A = np.zeros((self.nx, self.nx))  # ← outside if
        B = np.zeros((self.nx, 1))

        for j in range(self.nx):
            xp = x_eq.copy()
            xm = x_eq.copy()
            xp[j] += eps
            xm[j] -= eps
            A[:, j] = (self.xdot(xp, u_eq) - self.xdot(xm, u_eq)) / (2 * eps)

        B[:, 0] = (self.xdot(x_eq, u_eq + eps) - self.xdot(x_eq, u_eq - eps)) / (
            2 * eps
        )

        return A, B

    def discretize(self, A, B, dt):
        nx = A.shape[0]
        nu = B.shape[1]
        M = np.zeros((nx + nu, nx + nu))
        M[:nx, :nx] = A * dt
        M[:nx, nx:] = B * dt
        eM = expm(M)

        Ad = eM[:nx, :nx]
        Bd = eM[:nx, nx:]

        return Ad, Bd


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

    # test step_rk4 — energy conservation
    print("\n--- RK4 energy conservation test ---")
    x = np.zeros(p.nx)
    x[1] = 0.5
    dt = 0.001
    steps = 5000
    off = 1 if p.has_cart else 0

    energies = []
    for step in range(steps):
        q = x[: p.nq]
        dq = x[p.nq :]

        PE = 0.0
        for i in range(p.n):
            yi = sum(p.L[k] * np.cos(q[off + k]) for k in range(i + 1))
            PE += p.m[i] * p.g * yi

        M = p._mass_matrix(q)
        KE = 0.5 * dq @ M @ dq
        energies.append(KE + PE)

        if step % 500 == 0:
            print(f"step {step:4d}: KE={KE:.4f}  PE={PE:.4f}  E={KE + PE:.4f}")

        x = p.step_rk4(x, u=0, dt=dt)

    energies = np.array(energies)
    print(f"\nInitial energy:  {energies[0]:.6f}")
    print(f"Final energy:    {energies[-1]:.6f}")
    print(f"Max drift:       {np.max(np.abs(energies - energies[0])):.6f}")
    print(f"Energy conserved? {np.max(np.abs(energies - energies[0])) < 1e-2}")

    print("\n--- linearize test ---")
    A, B = p.linearize()
    eigenvalues = np.linalg.eigvals(A)
    print("A eigenvalues:", eigenvalues)
    print("Any unstable?", np.any(np.real(eigenvalues) > 0))

    print("\nA matrix:")
    print(np.round(A, 3))
    print("\nB matrix:")
    print(np.round(B, 3))
    # The pendulum-only subsystem (rows/cols 1,2,4,5 — dropping cart x and x_dot)
    idx = [1, 2, 4, 5]
    A_pend = A[np.ix_(idx, idx)]
    eigvals_pend = np.linalg.eigvals(A_pend)
    print("Pendulum subsystem eigenvalues:", eigvals_pend)
    print("Any unstable?", np.any(np.real(eigvals_pend) > 1e-6))

    # discretize test
    A, B = p.linearize()
    dt = 0.01
    Ad, Bd = p.discretize(A, B, dt)
    print("Ad shape:", Ad.shape)
    print("Bd shape:", Bd.shape)
    Ad_approx = np.eye(p.nx) + A * dt
    print("Ad ≈ I + A·dt? max diff:", np.max(np.abs(Ad - Ad_approx)))
    eigvals_Ad = np.linalg.eigvals(Ad)
    print("Ad eigenvalues (abs):", np.round(np.abs(eigvals_Ad), 4))
    print("Any unstable (|λ|>1)?", np.any(np.abs(eigvals_Ad) > 1))
