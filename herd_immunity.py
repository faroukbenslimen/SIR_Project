#!/usr/bin/env python3
"""
herd_immunity.py

Compute R0 and herd immunity threshold for given model parameters,
plot herd immunity threshold vs R0 and mark common diseases.
Saves plot as herd_immunity.png
"""

import numpy as np
import matplotlib.pyplot as plt


def herd_threshold(R0):
    """Return herd immunity threshold (percent) for a given R0.

    H = (1 - 1/R0) * 100
    """
    return (1.0 - 1.0 / R0) * 100.0


def main():
    # Model parameters
    beta = 0.3
    gamma = 0.05

    # Compute R0 and herd immunity for our model
    R0_model = beta / gamma
    H_model = herd_threshold(R0_model)

    print(f"Model parameters: beta={beta}, gamma={gamma}")
    print(f"Calculated R0 = beta/gamma = {R0_model:.3f}")
    print(f"Herd immunity threshold H = (1 - 1/R0) * 100 = {H_model:.2f}%")

    # R0 range for plotting
    R0_vals = np.linspace(1.1, 18.0, 500)
    H_vals = herd_threshold(R0_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(R0_vals, H_vals, color='C0', lw=2)
    plt.fill_between(R0_vals, 0, H_vals, color='C0', alpha=0.12)

    # Diseases to mark: (name, R0, color)
    diseases = [
        ("Seasonal Flu", 1.5, 'C1'),
        ("COVID-19", 2.5, 'C2'),
        ("Mumps", 5.0, 'C3'),
        ("Our model", R0_model, 'C4'),
        ("Measles", 15.0, 'C5'),
    ]

    # Exact annotation positions (data coordinates) and arrows
    xytext_map = {
        "Seasonal Flu": (1.8, 20),
        "COVID-19": (3.5, 52),
        "Mumps": (6.5, 72),
        "Our model": (7.5, 60),
        "Measles": (12.0, 88),
    }

    for name, r0_val, color in diseases:
        h_val = herd_threshold(r0_val)
        plt.scatter(r0_val, h_val, color=color, s=80, zorder=5)
        xtext, ytext = xytext_map.get(name, (r0_val + 0.5, h_val + 5))
        plt.annotate(f"{name}: {h_val:.1f}%", xy=(r0_val, h_val), xytext=(xtext, ytext),
                     fontsize=9, ha='left', va='bottom', color=color,
                     arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    plt.xlabel('Basic reproduction number $R_0$')
    plt.ylabel('Herd immunity threshold (%)')
    plt.title('Herd Immunity Threshold vs $R_0$')
    plt.xlim(1.0, 18.5)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('herd_immunity.png', dpi=150)
    plt.close()

    print('Plot saved to herd_immunity.png')


if __name__ == '__main__':
    main()
