import numpy as np
import matplotlib.pyplot as plt

# Apply your monitor settings
plt.rcParams.update({'font.size': 24})


def calculate_oc(minima, period, epoch):
    """Calculate O-C values and cycle numbers."""
    cycle_numbers = np.round((minima - epoch) / period).astype(int)

    # tweak:
    # cycle_numbers[cycle_numbers > 150] -= 1

    predicted_minima = epoch + cycle_numbers * period
    oc_values = minima - predicted_minima

    # tweak:
    oc_values[oc_values <= 0] += period

    return oc_values, cycle_numbers


def report_period_change(a, a_err, P, p_err):
    # 1. Dimensionless Rate (dP/dt)
    rate = (2 * a) / P

    # 2. Error Propagation (Quadrature)
    # Relative errors
    rel_err_a = a_err / abs(a)
    rel_err_p = p_err / P

    # Combined relative error
    rel_err_total = np.sqrt(rel_err_a**2 + rel_err_p**2)

    # Absolute error in the rate
    rate_err = abs(rate) * rel_err_total

    # 3. Conversion to seconds per year (for human readability)
    sec_per_year_const = 31557600
    spy = rate * sec_per_year_const
    spy_err = rate_err * sec_per_year_const

    print("\n" + "="*40)
    print("PERIOD CHANGE ANALYSIS (Full Error Budget)")
    print("="*40)
    print(f"Rate (dP/dt):  {rate:.4e} ± {rate_err:.4e}")
    print(f"Change/Year:   {spy:.4f} ± {spy_err:.4f} seconds/year")
    print("-" * 40)
    print(f"Contribution from a: {rel_err_a*100:.2f}%")
    print(f"Contribution from P: {rel_err_p*100:.2f}%")
    print("="*40)


from astropy.modeling import models, fitting


def correct_ephemeris_quadratic(minima, period, epoch, weights=None, max_iter=2):
    """
    Corrects Period and Epoch by absorbing the linear components (b, c)
    of a quadratic O-C fit.
    """
    current_period = period
    current_epoch = epoch

    fitter = fitting.LevMarLSQFitter()
    model_poly = models.Polynomial1D(degree=2)

    for i in range(max_iter):
        # 1. Calculate O-C and Cycles based on current best guess
        # (Assuming you have your calculate_oc method available)
        oc, cycles = calculate_oc(minima, current_period, current_epoch)

        # 2. Fit the parabola: O-C = a*E^2 + b*E + c
        # If no weights provided, use uniform weights
        w = weights if weights is not None else np.ones_like(oc)
        fitted_model = fitter(model_poly, cycles, oc, weights=w)

        # Extract coefficients
        c = fitted_model.c0.value  # Intercept
        b = fitted_model.c1.value  # Linear correction
        a = fitted_model.c2.value  # Quadratic term (Period change)

        # 3. Update the Ephemeris
        # New Epoch is shifted by the intercept
        new_epoch = current_epoch + c

        # New Period is shifted by the linear slope per cycle
        new_period = current_period + b

        print(f"Iteration {i + 1}:")
        print(f"  Linear Shift (b): {b:.8e} days/cycle")
        print(f"  Epoch Shift (c):  {c:.6f} days")
        print(f"  Refined Period:   {new_period:.10f}")
        print(f"  Refined Epoch:    {new_epoch:.6f}")

        current_period = new_period
        current_epoch = new_epoch

        # If the corrections are negligible, stop
        if abs(b) < 1e-9 and abs(c) < 1e-6:
            break

    return current_period, current_epoch, a


def fit(cycles, oc, jd_err):
    from astropy.modeling import models, fitting

    # 1. Setup
    p_model = models.Polynomial1D(degree=2)
    fitter = fitting.LevMarLSQFitter()  # Choose LevMar for errors
    weights = 1.0 / jd_err

    # 2. Fit
    best_fit = fitter(p_model, cycles, oc, weights=weights)

    # 3. Get Errors
    a_err = None
    cov_matrix = fitter.fit_info.get('param_cov')
    if cov_matrix is not None:
        # Errors are the square root of the diagonal elements
        # For degree=2: c0 is index 0, c1 is index 1, c2 (our 'a') is index 2
        errors = np.sqrt(np.diag(cov_matrix))
        a_err = errors[2]
        print(f"a = {best_fit.c2.value} ± {a_err}")
    return best_fit.c2.value, a_err


def main():
    # --- User Input ---
    filename = 'data/TCP_J05415572-2308340/results/all_extrema_sorted.dat'
    user_period = 0.05492936    # user_period = 0.0548999011
    user_period = 0.05508799611846793
    user_period = 0.05508890623505118
    user_period = 0.05496991
    # user_period = 0.0543927674
    # user_epoch = 966.907994313625
    # user_epoch = 966.944945
    user_epoch = 972.782693 - 0.01 + 0.0013
    user_epoch = 978.320583345

    # --- Load Data ---
    # col 0: JD, col 1: JD_Std
    try:
        jd, jd_err = np.loadtxt(filename, comments='#', usecols=(0, 1), unpack=True)
    except Exception as e:
        print(f"Error: {e}")
        return

    # --- Calculations ---
    # oc, cycles = calculate_oc(jd, user_period, user_epoch)
    oc, cycles = calculate_oc(jd, user_period, user_epoch)

    fit(cycles, oc, jd_err)
    # Weights for fitting (1/sigma)
    # Avoid division by zero if error is 0
    weights = 1.0 / np.where(jd_err == 0, 1e-6, jd_err)

    # --- Parabolic Fit (Quadratic: ax^2 + bx + c) ---
    # We fit O-C against Cycles (E) to find period change
    coeffs, cov = np.polyfit(cycles, oc, 2, w=weights, cov=True)
    a, b, c = coeffs

    # Correct period and epoch
    coeffs1, cov1 = np.polyfit(jd, oc, 2, w=weights, cov=True)
    a1, b1, c1 = coeffs1
    new_epoch = user_epoch + c1
    # New Period is shifted by the linear slope per cycle
    new_period_jd = user_period / (1 - b)
    new_period_e = user_period + b
    print(f"  Linear Shift (b): {b:.8e} days/cycle")
    print(f"  Epoch Shift (c):  {c:.6f} days")
    print(f"  Refined Period jd    :   {new_period_jd:.10f}")
    print(f"  Refined Period cycles:   {new_period_e:.10f}")
    print(f"  Refined Epoch:    {new_epoch:.6f}")

    # current_period = new_period
    # current_epoch = new_epoch

    # Errors of coefficients (square root of diagonal of covariance matrix)
    a_err, b_err, c_err = np.sqrt(np.diag(cov))

    # --- Calculate Minimum of Parabola ---
    # For y = ax^2 + bx + c, the extremum is at x = -b / (2a)
    # This represents the cycle number of the true minimum
    E_min = -b / (2 * a)

    # Propagation of error for E_min (simplified)
    # sigma_E = |E_min| * sqrt((sig_b/b)^2 + (sig_a/a)^2)
    E_min_err = abs(E_min) * np.sqrt((b_err / b) ** 2 + (a_err / a) ** 2)

    # Convert cycle minimum back to JD time
    t_min = user_epoch + (E_min * user_period)
    t_min_err = E_min_err * user_period

    # Calculate the rate dP/dt
    rate_dp_dt = (2 * a) / user_period
    rate_err = (2 * a_err) / user_period

    # Convert to seconds per year for a readable result
    # (1 year = 31 557600 seconds = (365*24 + 6)*60*60)
    seconds_per_year = rate_dp_dt * 31557600
    seconds_per_year_err = rate_err * 31557600

    print(f"Dimensionless rate (dP/dt): {rate_dp_dt:.4e} ± {rate_err:.4e}")
    print(f"Period change: {seconds_per_year:.4f} ± {seconds_per_year_err:.4f} seconds/year")
    # --- Report ---
    print("-" * 30)
    print(f"FIT RESULTS (O-C = aE² + bE + c):")
    print(f"a: {a:.4e} ± {a_err:.4e}")
    print(f"b: {b:.4e} ± {b_err:.4e}")
    print(f"c: {c:.4e} ± {c_err:.4e}")
    print("-" * 30)
    print(f"Time of Parabola Minimum (JD):")
    print(f"T_min = {t_min:.6f} ± {t_min_err:.6f}")
    print("-" * 30)

    print('New Report')
    user_period_err = 0.0001
    report_period_change(a, a_err, user_period, user_period_err)
    # --- Plotting ---
    plt.figure(figsize=(16, 10))

    # Plot O-C with error bars
    plt.errorbar(cycles, oc, yerr=jd_err, fmt='bo', capsize=5, label='Observed O-C', markersize=10)

    # Generate smooth parabola line for plotting
    E_fine = np.linspace(min(cycles), max(cycles), 200)
    oc_fit = a * E_fine ** 2 + b * E_fine + c
    # plt.plot(E_fine, oc_fit, 'r-', lw=3, label=f'Parabolic Fit\n$a={a:.2e}$')
    plt.plot(E_fine, oc_fit, 'r-', lw=3, label=f'Fit\n$dP/P={2*a/user_period:.2e}$')

    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel("Cycle Number (E)")
    plt.ylabel("O-C [days]")
    plt.title("O-C Parabolic Approximation")
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
