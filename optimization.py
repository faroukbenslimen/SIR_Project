#!/usr/bin/env python3
"""
optimization.py

Find optimal vaccination rate `v` (between 0 and 0.05) that minimizes
the total infections measured as the area under the infected curve.
Saves a 2-panel figure as `optimization.png`.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


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


def total_infections(v, N=10000, I0=10, beta=0.3, gamma=0.05, days=160, dt=0.1):
    """Run the SIR vaccination model for rate `v` and return area under I(t).

    The area is computed with the trapezoid rule (numpy.trapz).
    """
    t, y = run_sir_vaccination(N=N, I0=I0, beta=beta, gamma=gamma, v=v, days=days, dt=dt)
    I = y[1]
    area = np.trapezoid(I, t)
    return float(area)


def peak_infected(v, N=10000, I0=10, beta=0.3, gamma=0.05, days=160, dt=0.1, _cache={}):
    """Run the SIR vaccination model for rate `v` and return peak infected (max I).

    A tiny cache is used to avoid re-running the solver for the same rounded `v`.
    """
    key = round(float(v), 8)
    if key in _cache:
        return _cache[key]
    t, y = run_sir_vaccination(N=N, I0=I0, beta=beta, gamma=gamma, v=v, days=days, dt=dt)
    I = y[1]
    peak = float(np.max(I))
    _cache[key] = peak
    return peak


def make_plots(v_vals, peak_vals, v_opt, peak_opt, threshold, t_before, I_before, t_after, I_after, filename='optimization.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: peak vs vaccination rate
    axes[0].plot(v_vals, peak_vals, '-b')
    axes[0].plot(v_opt, peak_opt, 'ro', label=f'v*={v_opt:.5f}')
    axes[0].axhline(threshold, color='r', linestyle='--', label=f'threshold={threshold}')
    axes[0].set_xlabel('Vaccination rate v')
    axes[0].set_ylabel('Peak infected individuals')
    axes[0].set_title('Peak infected vs vaccination rate')
    axes[0].legend()
    axes[0].grid(True)

    # Right: epidemic curves before and after found v
    axes[1].plot(t_before, I_before, label='v=0 (no vaccination)', linewidth=2)
    axes[1].plot(t_after, I_after, label=f'v={v_opt:.5f}', linewidth=2)
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Number of infected individuals')
    axes[1].set_title('Epidemic curve: before vs after')
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f'Plot saved to {filename}')


def main():
    # Model parameters
    N = 10000
    I0 = 10
    beta = 0.3
    gamma = 0.05
    days = 160
    dt = 0.1

    # New objective: find minimal v such that peak infected <= threshold
    threshold = 1000.0

    # Peak without vaccination
    peak0 = peak_infected(0.0, N=N, I0=I0, beta=beta, gamma=gamma, days=days, dt=dt)

    # Peak at upper bound to check feasibility
    ub = 0.05
    peak_ub = peak_infected(ub, N=N, I0=I0, beta=beta, gamma=gamma, days=days, dt=dt)

    if peak0 <= threshold:
        v_opt = 0.0
        peak_opt = peak0
        print('No vaccination needed: baseline peak already below threshold')
    elif peak_ub > threshold:
        v_opt = ub
        peak_opt = peak_ub
        print('No v in [0, 0.05] achieves peak < threshold')
    else:
        # Minimize absolute difference to find crossing where peak ~= threshold
        obj = lambda v: abs(peak_infected(v, N=N, I0=I0, beta=beta, gamma=gamma, days=days, dt=dt) - threshold)
        res = minimize_scalar(obj, bounds=(0.0, ub), method='bounded')
        v_guess = float(res.x)
        peak_guess = peak_infected(v_guess, N=N, I0=I0, beta=beta, gamma=gamma, days=days, dt=dt)

        # Refine with bisection to ensure the returned v is the smallest one
        # with peak <= threshold (assumes peak decreases with v)
        low = 0.0
        high = ub
        tol = 1e-6
        # ensure bracket
        if peak_infected(low, N=N, I0=I0, beta=beta, gamma=gamma, days=days, dt=dt) <= threshold:
            v_opt = 0.0
            peak_opt = peak_infected(0.0, N=N, I0=I0, beta=beta, gamma=gamma, days=days, dt=dt)
        else:
            # tighten bracket around guess to speed up
            if peak_guess <= threshold:
                high = v_guess
            else:
                low = v_guess

            while high - low > tol:
                mid = 0.5 * (low + high)
                if peak_infected(mid, N=N, I0=I0, beta=beta, gamma=gamma, days=days, dt=dt) > threshold:
                    low = mid
                else:
                    high = mid

            v_opt = high
            peak_opt = peak_infected(v_opt, N=N, I0=I0, beta=beta, gamma=gamma, days=days, dt=dt)

    # Print requested outputs
    print(f'Minimum v achieving peak < {threshold}: {v_opt:.6f}')
    print(f'Peak infections without vaccination: {peak0:.2f}')
    print(f'Peak infections with v={v_opt:.6f}: {peak_opt:.2f}')
    reduction_pct = 100.0 * (peak0 - peak_opt) / peak0 if peak0 != 0 else 0.0
    print(f'Percentage reduction in peak: {reduction_pct:.2f}%')

    # Build peak vs v curve for plotting
    v_vals = np.linspace(0.0, ub, 51)
    peak_vals = [peak_infected(v, N=N, I0=I0, beta=beta, gamma=gamma, days=days, dt=dt) for v in v_vals]

    # Epidemic curves before (v=0) and after (v_opt)
    t_before, y_before = run_sir_vaccination(N=N, I0=I0, beta=beta, gamma=gamma, v=0.0, days=days, dt=dt)
    t_after, y_after = run_sir_vaccination(N=N, I0=I0, beta=beta, gamma=gamma, v=v_opt, days=days, dt=dt)

    I_before = y_before[1]
    I_after = y_after[1]

    make_plots(v_vals, peak_vals, v_opt, peak_opt, threshold, t_before, I_before, t_after, I_after, filename='optimization.png')


if __name__ == '__main__':
    main()
