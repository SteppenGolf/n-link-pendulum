import numpy as np


class KalmanFilter:
    def __init__(self, Ad, Bd, C, x0, P0, Q, R):
        self.Ad = Ad  # State Transition Matrix

        self.Bd = Bd  # Control Matrix
        self.C = C  # Observation Matrix
        self.x = x0.copy()  # Inital State Estimate
        self.P = P0.copy()  # Initial Error Covariance
        self.Q = Q  # Process noise Covariance
        self.R = R  # Measurement noise Covariance

    def predict(self, u):
        self.x = self.Ad @ self.x + self.Bd @ u
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q

    def update(self, y):
        self.K = (
            (self.P) @ self.C.T @ np.linalg.inv(self.C @ (self.P) @ self.C.T + self.R)
        )
        self.x = (self.x) + self.K @ (y - self.C @ (self.x))
        self.P = (np.eye(self.x.shape[0]) - self.K @ self.C) @ self.P
