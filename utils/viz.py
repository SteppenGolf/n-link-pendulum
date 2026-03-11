import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.use('TkAgg')

class Visualizer:
    def __init__(self, pendulum):
        self.pendulum = pendulum

    def animate(self, x_hist, dt):
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True)

        cart_width, cart_height = 0.3, 0.15
        cart = plt.Rectangle((0, -cart_height / 2), cart_width, cart_height, color='blue')
        ax.add_patch(cart)
        line, = ax.plot([], [], 'o-', lw=2, color='red')

        def update(frame):
            x = x_hist[frame]
            q = x[:self.pendulum.nq]
            cart_x = q[0] if self.pendulum.has_cart else 0.0
            cart.set_xy((cart_x - cart_width / 2, -cart_height / 2))
            tips = self.pendulum.tip_positions(x)
            xs = [cart_x] + [t[0] for t in tips]
            ys = [0.0]     + [t[1] for t in tips]
            line.set_data(xs, ys)
            return cart, line

        ani = animation.FuncAnimation(
            fig, update, frames=len(x_hist), interval=dt * 1000, blit=True
        )
        plt.show()
        return ani

    def plot_states(self, x_hist, dt):
        time = np.arange(len(x_hist)) * dt
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        labels = ['cart x'] + [f'theta_{i+1}' for i in range(self.pendulum.n)]
        for i in range(self.pendulum.nq):
            axes[0].plot(time, x_hist[:, i], label=labels[i])
        axes[0].set_title("Positions and Angles")
        axes[0].set_ylabel("rad / m")
        axes[0].legend()
        axes[0].grid(True)
        vel_labels = ['cart v'] + [f'dtheta_{i+1}' for i in range(self.pendulum.n)]
        for i in range(self.pendulum.nq):
            axes[1].plot(time, x_hist[:, self.pendulum.nq + i], label=vel_labels[i])
        axes[1].set_title("Velocities")
        axes[1].set_ylabel("rad/s / m/s")
        axes[1].set_xlabel("time (s)")
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        plt.show()

    def plot_control(self, u_hist, dt):
        time = np.arange(len(u_hist)) * dt
        plt.figure(figsize=(10, 3))
        plt.plot(time, u_hist)
        plt.title("Control Input")
        plt.xlabel("time (s)")
        plt.ylabel("force (N)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from dynamics.pendulum import NLinkPendulum
    from control.lqr import LQR
    from simulation.simulator import Simulator

    dt = 0.01
    p = NLinkPendulum(n=2, lengths=[0.5, 0.4], masses=[0.3, 0.2], cart_mass=1.0)
    A, B = p.linearize()
    Ad, Bd = p.discretize(A, B, dt)
    Q = np.eye(p.nx)
    R = np.eye(p.nu)
    lqr = LQR(Ad, Bd, Q, R)
    lqr.compute_gain()

    sim = Simulator(p, lqr)
    x0 = np.zeros(p.nx)
    x0[1] = 0.2
    x_hist, u_hist = sim.run(x0, dt, steps=500)

    viz = Visualizer(p)
    viz.plot_states(x_hist, dt)
    viz.plot_control(u_hist, dt)
    viz.animate(x_hist, dt)