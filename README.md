# N-Link Pendulum Control Simulator

Simulation of an N-link pendulum on a cart using LQR and MPC controllers,
with a Kalman filter for state estimation. Built as a research portfolio project.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/streamlit_app.py
```

## Structure

- `dynamics/` — Lagrangian dynamics, RK4 integrator, linearization
- `control/` — LQR (DARE) and MPC (OSQP) controllers
- `estimation/` — Discrete Kalman filter
- `simulation/` — Closed-loop simulation loop
- `utils/` — Visualization and animation
- `app/` — Streamlit interactive demo
