#!/usr/bin/env python3
"""
sir_basic.py

Basic SIR model simulation using RK45 (scipy.solve_ivp).
Saves plot as sir_basic.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def sir_model(t, y, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def run_sir(N=10000, I0=10, beta=0.3, gamma=0.05, days=160, dt=0.1):
    S0 = N - I0
    R0 = 0.0
    y0 = [float(S0), float(I0), float(R0)]
    t_span = (0.0, float(days))
    t_eval = np.linspace(0, days, int(days / dt) + 1)

    sol = solve_ivp(fun=lambda t, y: sir_model(t, y, beta, gamma, N),
                    t_span=t_span, y0=y0, t_eval=t_eval, method='RK45')
 
    if not sol.success:
        raise RuntimeError('ODE solver failed: ' + str(sol.message))

    return sol.t, sol.y


def plot_sir(t, y, filename='sir_basic.png'):
    S, I, R = y
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible', color='C0')
    plt.plot(t, I, label='Infected', color='C1')
    plt.plot(t, R, label='Recovered', color='C2')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model (Beta=0.3, Gamma=0.05)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f'Plot saved to {filename}')


def main():
    # Parameters
    N = 10000
    I0 = 10
    beta = 0.3
    gamma = 0.05
    days = 160

    t, y = run_sir(N=N, I0=I0, beta=beta, gamma=gamma, days=days, dt=0.1)
    plot_sir(t, y, filename='sir_basic.png')

    # Report peak infected
    I = y[1]
    peak_idx = np.argmax(I)
    print(f'Peak infected: {I[peak_idx]:.0f} at day {t[peak_idx]:.1f}')


if __name__ == '__main__':
    main()
