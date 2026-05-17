#!/usr/bin/env python3
"""
real_data_fit.py

Load Tunisia COVID-19 data from a local CSV, fit a simple SIR model to
Wave 2 (2021-03-01 to 2021-08-15) by minimizing squared differences
between observed (7-day smoothed) daily cases and model `I(t)`. Saves
`real_data_fit.png` with three panels (data+fit, S/I/R %, info card).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def smooth_series(series, window=7):
    """Return centered rolling mean with minimum 1 observation."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def sir_model(t, y, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def run_sir(beta, gamma, N, I0, days, dt=1.0):
    """Solve SIR ODE for `days` days with daily outputs.

    Returns t (days array) and y (3 x len(t) array).
    """
    S0 = N - I0
    R0 = 0.0
    y0 = [float(S0), float(I0), float(R0)]
    t_span = (0.0, float(days - 1))
    t_eval = np.arange(0, days, dt)
    sol = solve_ivp(fun=lambda t, y: sir_model(t, y, beta, gamma, N),
                    t_span=t_span, y0=y0, t_eval=t_eval, method='RK45')
    if not sol.success:
        raise RuntimeError('ODE solver failed: ' + str(sol.message))
    return sol.t, sol.y


def objective_sse(params, N, I0, real_vals):
    """Objective: sum of squared errors between model I(t) and real_vals."""
    beta, gamma = params
    if beta <= 0 or gamma <= 0:
        return 1e20
    days = len(real_vals)
    try:
        _, y = run_sir(beta, gamma, N=N, I0=I0, days=days)
    except Exception:
        return 1e20
    I_model = y[1]
    if I_model.shape[0] != days:
        return 1e20
    residuals = I_model - real_vals
    return float(np.sum(residuals ** 2))


def fit_parameters(N, I0, real_vals, initial_guesses=None):
    if initial_guesses is None:
        initial_guesses = [(0.3, 0.07), (0.4, 0.08), (0.25, 0.06), (0.5, 0.1)]

    best_res = None
    best_fun = np.inf
    for guess in initial_guesses:
        res = minimize(lambda x: objective_sse(x, N=N, I0=I0, real_vals=real_vals),
                       x0=np.array(guess), method='Nelder-Mead',
                       options={'maxiter': 2000, 'disp': False})
        if res.success and res.fun < best_fun:
            best_fun = res.fun
            best_res = res
    if best_res is None:
        raise RuntimeError('All fits failed')
    return best_res.x, best_fun


def plot_results(dates, real_vals, t_model, S, I, R, N, beta, gamma, R0, rmse,
                 peak_idx, peak_date, peak_val, out_file='real_data_fit.png'):
    """Create white-themed 3-panel figure and save to `out_file`."""
    text_color = 'black'
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3)

    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    # Top panel: real data (filled crimson) and fitted I as royalblue dashed
    ax_top.fill_between(dates, real_vals, color='crimson', alpha=0.6, label='Real (7d avg)', linewidth=0)
    model_dates = pd.to_datetime(dates.iloc[0]) + pd.to_timedelta(t_model, unit='D')
    ax_top.plot(model_dates, I, '--', color='royalblue', linewidth=2, label='Fitted SIR I(t)')
    ax_top.set_ylabel('Daily cases / I(t)', color=text_color)
    ax_top.set_title('Tunisia COVID-19 Wave 2: Data and Fitted SIR Model')
    leg = ax_top.legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_color(text_color)
    ax_top.grid(alpha=0.3)
    ax_top.xaxis.set_major_locator(mdates.MonthLocator())
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_top.tick_params(axis='x', rotation=30)

    # Bottom-left: S, I, R as percentage of population
    ax_bl.plot(model_dates, S / N * 100.0, color='green', label='Susceptible %')
    ax_bl.plot(model_dates, I / N * 100.0, color='red', label='Infected %')
    ax_bl.plot(model_dates, R / N * 100.0, color='blue', label='Recovered %')
    ax_bl.set_xlabel('Date', color=text_color)
    ax_bl.set_ylabel('% of population', color=text_color)
    leg_bl = ax_bl.legend()
    if leg_bl is not None:
        for text in leg_bl.get_texts():
            text.set_color(text_color)
    ax_bl.grid(alpha=0.3)
    ax_bl.xaxis.set_major_locator(mdates.MonthLocator())
    ax_bl.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_bl.tick_params(axis='x', rotation=30)

    # Bottom-right: info card
    ax_br.axis('off')
    info_lines = [
        f'Fitted parameters',
        f'beta: {beta:.5f}',
        f'gamma: {gamma:.5f}',
        f'R0: {R0:.3f}',
        f'Peak cases (model): {peak_val:.0f}',
        f'Peak date (model): {peak_date.strftime("%Y-%m-%d")}',
        f'RMSE: {rmse:.2f}',
        f'Population: {int(N):,}',
    ]
    text = '\n'.join(info_lines)
    ax_br.text(0.02, 0.98, text, va='top', ha='left', color='black', fontsize=12, family='monospace')

    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f'Plot saved to {out_file}')


def main():
    # Load CSV and normalize columns coming from the provided file
    df = pd.read_csv('tunisia_covid.csv')

    # Rename columns to match expected names
    df = df.rename(columns={
        'Entity': 'location',
        'Day': 'date',
        'New cases (per 1M)': 'new_cases_per_million'
    })

    # Convert per-million to absolute new cases (Tunisia population = 12,000,000)
    N = 12_000_000
    df['new_cases'] = (df['new_cases_per_million'] * N / 1_000_000).round()

    # Parse dates and drop negatives
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['new_cases'] >= 0].reset_index(drop=True)

    # Select Wave 2 date window
    start = pd.to_datetime('2021-03-01')
    end = pd.to_datetime('2021-08-15')
    mask = (df['date'] >= start) & (df['date'] <= end)
    df_wave = df.loc[mask].sort_values('date').reset_index(drop=True)
    if df_wave.empty:
        raise RuntimeError('No Tunisia data in the requested date range')

    # Ensure numeric and smooth
    df_wave['new_cases'] = pd.to_numeric(df_wave['new_cases'], errors='coerce').fillna(0.0)
    df_wave['new_cases_smooth'] = smooth_series(df_wave['new_cases'], window=7)

    dates = df_wave['date']
    y_obs = df_wave['new_cases_smooth'].values

    # Initial infected
    I0 = max(float(y_obs[0]), 1.0)

    # Fit beta and gamma using Nelder-Mead from multiple starts
    initial_guesses = [(0.3, 0.07), (0.4, 0.08), (0.25, 0.06), (0.5, 0.1)]
    (beta_hat, gamma_hat), _ = fit_parameters(N=N, I0=I0, real_vals=y_obs, initial_guesses=initial_guesses)

    # Run model with best-fit params
    days = len(y_obs)
    t_model, y_model = run_sir(beta_hat, gamma_hat, N=N, I0=I0, days=days)
    S, I, R = y_model

    # Metrics
    residuals = I - y_obs
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    R0_val = beta_hat / gamma_hat
    peak_idx = int(np.argmax(I))
    peak_val = float(I[peak_idx])
    peak_date = pd.to_datetime(dates.iloc[peak_idx])

    print(f'Fitted beta: {beta_hat:.5f}')
    print(f'Fitted gamma: {gamma_hat:.5f}')
    print(f'Estimated R0: {R0_val:.3f}')
    print(f'RMSE: {rmse:.2f}')

    # Plot and save
    plot_results(dates, y_obs, t_model, S, I, R, N, beta_hat, gamma_hat, R0_val, rmse,
                 peak_idx, peak_date, peak_val, out_file='real_data_fit.png')


if __name__ == '__main__':
    main()
