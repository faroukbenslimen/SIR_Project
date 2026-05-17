# SIR Epidemic Modeling Project 🦠📈

This repository contains a mathematical modeling project that simulates the spread of infectious diseases using the **SIR (Susceptible, Infectious, Recovered)** model. It was designed to study epidemiological dynamics, evaluate the impact of vaccination, and calibrate theoretical models against real-world data.

## 🚀 Key Features

*   **Basic SIR Dynamics (`sir_basic.py`)**: Simulates the standard Susceptible-Infectious-Recovered flow over time to demonstrate peak infection rates and epidemic curves.
*   **Vaccination Impact (`sir_vaccination.py`)**: Extends the basic model to show how a vaccinated population "flattens the curve" and reduces the maximum number of simultaneous infections.
*   **Herd Immunity & Optimization (`herd_immunity.py` & `optimization.py`)**: Calculates the critical vaccination threshold needed to achieve herd immunity and prevent an outbreak from growing.
*   **Real Data Calibration (`real_data_fit.py`)**: Fits the SIR model parameters to actual COVID-19 data (specifically Wave 2 data from Tunisia) to demonstrate the real-world applicability of differential equations.

## 🛠️ Technologies Used

*   **Python**: Core logic and simulation.
    *   `NumPy`: Numerical calculations.
    *   `SciPy` (`solve_ivp`): Solving ordinary differential equations (ODEs).
    *   `Matplotlib`: Data visualization and curve plotting.
*   **LaTeX / TikZ**: Academic report generation and complex diagram overlays (PDF outputs).

## 📊 How to Run the Simulations

1. Ensure you have Python installed along with the necessary libraries:
   ```bash
   pip install numpy scipy matplotlib
   ```
2. Navigate to the project directory or the `simulation_python` folder.
3. Run any of the individual scripts, for example:
   ```bash
   python sir_basic.py
   ```
This will run the differential equations and automatically generate the corresponding graphs.
