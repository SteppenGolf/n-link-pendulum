import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
from dynamics.pendulum import NLinkPendulum
from control.lqr import LQR
from estimation.kalman import KalmanFilter
class Simulator:
    def __init__(self, pendulum, controller, estimator=None):
        self.pendulum =pendulum
        self.controller = controller
        self.estimator = estimator
        
    
    def run(self, x0, dt, steps):
        self.x0 = x0
        self.dt =dt 
        self.steps = steps

        x= x0.copy()
        x_history = [x.copy()]
        u_history = []

        for _ in range(steps):
            u = self.controller.control(x)
            x = self.pendulum.step_rk4(x,u,dt)
            x_history.append(x.copy())
            u_history.append(u)  
        return np.array(x_history), np.array(u_history)
    
if __name__ == "__main__":
    dt = 0.01
    p = NLinkPendulum(n=2, lengths=[0.5, 0.4], masses=[0.3, 0.2], cart_mass=1.0)
    A, B = p.linearize()
    Ad, Bd = p.discretize(A, B, dt)
    Q = np.eye(p.nx)
    R = np.eye(p.nu)
    lqr = LQR(Ad, Bd, Q, R)
    lqr.compute_gain()
    sim = Simulator(p,lqr)

    x0 = np.zeros(p.nx)
    x0[1] = 0.1
    x_hist, u_hist = sim.run (x0, dt, steps= 500)

    print("Initial state:", np.round(x_hist[0], 4))
    print("Final state:", np.round(x_hist[-1], 4))