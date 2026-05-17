#!/usr/bin/env python3
"""
sir_vaccination.py

SIR model with vaccination. Runs four scenarios with different
vaccination rates and plots the infected curves together.
Saves plot as sir_vaccination.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def sir_model_vaccination(t, y, beta, gamma, N, v):
    S, I, R = y
    dSdt = -beta * S * I / N - v * S
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I + v * S
    return [dSdt, dIdt, dRdt]


def run_sir_vaccination(N=10000, I0=10, beta=0.3, gamma=0.05, v=0.0, days=160, dt=0.1):
    S0 = N - I0
    R0 = 0.0
    y0 = [float(S0), float(I0), float(R0)]
    t_span = (0.0, float(days))
    t_eval = np.linspace(0, days, int(days / dt) + 1)

    sol = solve_ivp(fun=lambda t, y: sir_model_vaccination(t, y, beta, gamma, N, v),
                    t_span=t_span, y0=y0, t_eval=t_eval, method='RK45')

    if not sol.success:
        raise RuntimeError('ODE solver failed: ' + str(sol.message))

    return sol.t, sol.y


def plot_vaccination_infected(t, results, rates, filename='sir_vaccination.png'):
    plt.figure(figsize=(10, 6))
    for y, v in zip(results, rates):
        I = y[1]
        plt.plot(t, I, label=f'v={v}', linewidth=2)

    plt.xlabel('Time (days)')
    plt.ylabel('Number of infected individuals')
    plt.title('Infected over time for different vaccination rates')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f'Plot saved to {filename}')


def main():
    # Model parameters
    N = 10000
    I0 = 10
    beta = 0.3
    gamma = 0.05
    days = 160
    dt = 0.1

    vaccination_rates = [0.0, 0.005, 0.01, 0.02]

    results = []
    t_ref = None
    for v in vaccination_rates:
        t, y = run_sir_vaccination(N=N, I0=I0, beta=beta, gamma=gamma, v=v, days=days, dt=dt)
        results.append(y)
        t_ref = t if t_ref is None else t_ref

        I = y[1]
        peak_idx = np.argmax(I)
        print(f'v={v}: Peak infected: {I[peak_idx]:.0f} at day {t[peak_idx]:.1f}')

    plot_vaccination_infected(t_ref, results, vaccination_rates, filename='sir_vaccination.png')


if __name__ == '__main__':
    main()
