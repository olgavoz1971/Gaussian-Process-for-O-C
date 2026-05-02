import sys

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})  # Set global font size


def calculate_oc(minima, period, epoch):
    """Calculate O-C values for observed t_minima based on given period and epoch."""
    cycle_numbers = np.round((minima - epoch) / period).astype(int)

    # tweak cycle number here
    # cycle_numbers[cycle_numbers > 150] -= 1
    # cycle_numbers[cycle_numbers > -93] += 1
    # cycle_numbers[cycle_numbers > 50] -= 1

    predicted_minima = epoch + cycle_numbers * period
    oc_values = minima - predicted_minima

    # tweak
    # oc_values[oc_values < 0] += period

    # plt.figure(figsize=(16, 10))
    # plt.scatter(cycle_numbers, oc_values, color='b', label='O-C values')
    # plt.axhline(0, color='r', linestyle='--', lw=1)
    # plt.xlabel("Time [JD]")
    # plt.ylabel("O-C [days]")
    # plt.title('O-C vs cycles')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    return oc_values, cycle_numbers


def plot_oc(minima, oc_values, title="O-C Diagram"):
    """Plot O-C diagram."""
    plt.figure(figsize=(16, 10))
    plt.scatter(minima, oc_values, color='b', label='O-C values')
    plt.axhline(0, color='r', linestyle='--', lw=1)
    plt.xlabel("Time [JD]")
    plt.ylabel("O-C [days]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def fit_linear(t, err, oc):
    """Fit a linear trend to O-C values, return the slope and intercept."""
    # Initial fit
    weights = 1.0 / np.where(err == 0, 1e-6, err)
    coeffs = np.polyfit(t, oc, 1, w=weights)
    slope, intercept = coeffs

    # Simple 3-sigma outlier rejection
    residuals = oc - (slope * t + intercept)
    std_residuals = np.std(residuals)
    inlier_mask = np.abs(residuals) < 3 * std_residuals

    t_inliers = t[inlier_mask]
    oc_inliers = oc[inlier_mask]
    w_inliers = weights[inlier_mask]

    # Final fit on inliers
    slope_final, intercept_final = np.polyfit(t_inliers, oc_inliers, 1, w=w_inliers)

    # Plotting the diagnostic fit
    plt.figure(figsize=(16, 10))
    plt.scatter(t, oc, alpha=0.5, label='O-C values')
    plt.plot(t, slope * t + intercept, 'g--', label='Initial fit')
    plt.plot(t, slope_final * t + intercept_final, 'r-', label='Final (Cleaned) fit')
    plt.axhline(0, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Time (JD)')
    plt.ylabel('O-C (days)')
    plt.title("Linear Trend Fitting")
    plt.legend()
    plt.show()

    return slope_final, intercept_final


def correct_period(minima, err, period, epoch, max_iter=1, tol=1e-6):
    """
    Correction of a constant period and epoch based on O-C slope/intercept

    We assume that the true period P_true is unknown,
    and the t_minima are computed using an incorrect period P_wrong.
    The observed t_minima occur with the true period:
        O_n = epoch + n * P_true
    while the predicted t_minima (calculated with P_wrong) are:
        C_n = epoch + n * P_wrong
    The O-C diagram is constructed as the difference:
        O-C_n = O_n - C_n = (epoch + n * P_true) - (epoch + n * P_wrong)
    Simplifying:
        O-C_n = n * (P_true - P_wrong)
    Since the minimum number n can be expressed in terms of time:
        n = (t - epoch) / P_true
    we obtain the time dependence of O-C:
        O-C(t) = (t - epoch) / P_true * (P_true - P_wrong)
    Now, we compute the slope of the O-C diagram:
        S = d(O-C) / dt
    Differentiating:
        S = d/dt [(t - epoch) / P_true * (P_true - P_wrong)]
    Since (P_true - P_wrong) / P_true is a constant, it can be factored out:
        S = (P_true - P_wrong) / P_true * d/dt (t - epoch)
    The derivative of (t - epoch) with respect to t is 1, so:
        S = (P_true - P_wrong) / P_true
    or, rewritten:
        S = 1 - P_wrong / P_true
    From this, we can express the true period:
        P_true = P_wrong / (1 - S)
    Thus, if we measure the slope S of the O-C diagram,
    the corrected period is calculated as:
        P_corrected = P_wrong / (1 - S)

    This formula allows for an automatic correction of the variable star's period
    based on the O-C diagram data
    """
    current_period = period
    current_epoch = epoch

    for i in range(max_iter):
        oc_values, _ = calculate_oc(minima, current_period, current_epoch)
        slope, intercept = fit_linear(minima, err, oc_values)

        # Update period using your derived formula
        # P_true = P_wrong / (1 - slope)
        new_period = current_period / (1 - slope)

        # The intercept tells us the shift in T0 (Epoch)
        # T0_new = T0_old + intercept
        new_epoch = current_epoch + intercept

        print(f"Iteration {i + 1}:")
        print(f"  Slope: {slope:.8f}")
        print(f"  New Period: {new_period:.8f}")
        print(f"  New Epoch:  {new_epoch:.6f}")

        current_period = new_period
        current_epoch = new_epoch

        if abs(slope) < tol:
            break

    return current_period, current_epoch


def main():
    # --- Configuration ---
    # filename = "minima_data.txt"  # Your input file
    filename = 'data/HD182144/lc_tess_Stitched_curve_HD182144_TIC_406949643_sector__40_41_55_53_54_75_74_extrema.dat'
    # initial_p = 0.5234  # Example initial period

    initial_p = 0.0549
    initial_p = 0.05492936
    initial_p = 0.05508890623505119
    initial_p = 0.055387603453644504
    initial_p = 0.05508799611846793
    initial_epoch = 966.907994313625
    initial_epoch = 972.782693 - 0.01 + 0.0013
    initial_epoch = 978.320583345
    filename = 'data/TCP_J05415572-2308340/results/all_extrema_sorted.dat'

    # --- Data Loading ---
    try:
        # Skips comments starting with # and loads JD_Minimum (col 0)
        jd, err = np.loadtxt(filename, comments='#', usecols=(0, 1), unpack=True)
        if jd.size == 0:
            raise ValueError("File is empty")
        print(f"Successfully loaded {len(jd)} minima.")
    except Exception as e:
        print(f"Error loading file: {e}")
        # Dummy data for demonstration if file not found
        data = np.array([2459407.550650, 2459414.062929, 2459420.575208])
        print("Using dummy data for script testing.")
        sys.exit(0)

    # --- Step 1: Initial O-C State ---
    print("\n--- Initial O-C Analysis ---")
    oc_init, _ = calculate_oc(jd, initial_p, initial_epoch)
    plot_oc(jd, oc_init, title="Initial O-C (Uncorrected)")

    # --- Step 2: Period and Epoch Correction ---
    print("\n--- Running Correction ---")
    corrected_p, corrected_epoch = correct_period(jd, err, initial_p, initial_epoch)
    print(f'corrected period={corrected_p} corrected epoch={corrected_epoch}')

    # --- Step 3: Final Verification ---
    print("\n--- Final O-C Verification ---")
    oc_final, _ = calculate_oc(jd, corrected_p, corrected_epoch)
    plot_oc(jd, oc_final, title="Final O-C (Corrected)")

    print(f"\nFinal Results:")
    print(f"Period: {corrected_p:.10f} days")
    print(f"Epoch:  {corrected_epoch:.6f} JD")


if __name__ == "__main__":
    main()
