"""
binary_processor.py

An integrated module for astronomical time-series processing, period correction,
Analysis of Variance (AoV) significance profiling, and publication-grade folding
incorporating quadratic O-C variations.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from astropy.time import Time
from scipy.special import ndtri
from scipy.stats import f as f_dist

# Set global formatting profile
# plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.size': 24})  # Set global font size


# ==========================  INPUT/OUTPUT/CUTOUT ============

def load_lightcurve_astropy(
        # region unfold
        obs_file: str,
        jd_col: Union[int, str] = 0,
        mag_col: Union[int, str] = 1,
        err_col: Optional[Union[int, str]] = 2,
        file_format: str = 'commented_header'
        # endregion
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load lightcurve time-series from any ASCII file format using Astropy.
    Automatically handles Time objects and maps missing errors to NaN.

    Parameters
    ----------
    obs_file : str
        Path to the space/tab-separated observation data file.
    jd_col : int or str, default 0
        The column name or 0-based index for Julian Dates.
    mag_col : int or str, default 1
        The column name or 0-based index for observations (magnitudes).
    err_col : int or str, optional, default 3
        The column name or 0-based index for uncertainties. If None, returns None.
    file_format : table format (i.g., ecsv
    """
    table = ascii.read(obs_file, format=file_format, header_start=0)
    colnames = table.colnames

    if isinstance(err_col, int):
        if len(table.columns) <= err_col:
            err_col = None
    else:
        if err_col not in colnames:
            err_col = None

    def resolve_col_name(col_identifier: Union[int, str]) -> str:
        if isinstance(col_identifier, str):
            for name in colnames:
                if name.lower() == col_identifier.lower():
                    return name
            raise ValueError(f"Column '{col_identifier}' not found in headers: {colnames}")
        return colnames[int(col_identifier)]

    jd_key = resolve_col_name(jd_col)
    mag_key = resolve_col_name(mag_col)

    jd_column_data = table[jd_key]
    if isinstance(jd_column_data, Time):
        jd_column_data = jd_column_data.jd

    jd_obs = np.ascontiguousarray(jd_column_data, dtype=float)
    mag_obs = np.ascontiguousarray(table[mag_key], dtype=float)

    if err_col is None:
        return jd_obs, mag_obs, None

    err_key = resolve_col_name(err_col)
    err_obs = np.ascontiguousarray(table[err_key], dtype=float)

    no_error_mask = np.isclose(err_obs, 99.99)
    err_obs[no_error_mask] = np.nan

    return jd_obs, mag_obs, err_obs


def load_extrema_txt(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load observation timing extrema minima/maxima files via standard numpy.
    Skips comment lines starting with # and unpacks columns 0 and 1.
    """
    jd, err = np.loadtxt(filename, comments='#', usecols=(0, 1), unpack=True)
    if jd.size == 0:
        raise ValueError("Provided extrema file is empty.")
    return jd, err


def cutout_data(
        # region fold_me
        jd: np.ndarray,
        mag: np.ndarray,
        mag_err: Optional[np.ndarray],
        jd_min: Optional[float],
        jd_max: Optional[float]
        # endregion
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Filters observation arrays tightly within a designated bounding JD range.
    Handles None values by defaulting to the array boundaries.
    """
    # If a bound is None, default to the extreme limits of the dataset
    actual_min = jd_min if jd_min is not None else np.min(jd)
    actual_max = jd_max if jd_max is not None else np.max(jd)

    mask = (jd >= actual_min) & (jd <= actual_max)

    jd_piece = jd[mask]
    mag_piece = mag[mask]
    err_piece = mag_err[mask] if mag_err is not None else None

    return jd_piece, mag_piece, err_piece


def export_folded_data(filename: str, jd: np.ndarray, phases: np.ndarray, mag: np.ndarray, err: Optional[np.ndarray]):
    """Saves calculated phases and observations to a clean whitespace-separated output file."""
    header = "jd phase mag err" if err is not None else "jd phase mag"
    valid = np.isfinite(phases) & np.isfinite(mag)

    if err is not None:
        err = np.where(np.isfinite(err), err, 99.99)
        data = np.column_stack((jd[valid], phases[valid], mag[valid], err[valid]))
        fmt = "%.6f  %.6f  %.4f  %.4f"
    else:
        data = np.column_stack((jd[valid], phases[valid], mag[valid]))
        fmt = "%.6f  %.6f  %.4f"

    # Sort data points ?
    data = data[np.argsort(data[:, 0])]
    np.savetxt(filename, data, header=header, fmt=fmt, comments='# ')
    print(f"[Export] Saved clean phase curves successfully to: '{filename}'")


# =============== O-C  and PARABOLIC EPHEMERIS MATHEMATICS ========

def calculate_oc(minima: np.ndarray, period: float, epoch: float,
                 tweak_oc: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate O-C values for observed t_minima based on given period and epoch."""
    cycle_numbers = np.round((minima - epoch) / period).astype(int)
    predicted_minima = epoch + cycle_numbers * period
    oc_values = minima - predicted_minima

    # Optional tweak hook
    if tweak_oc is not None:
        oc_values[oc_values <= tweak_oc] += period

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


def fit_linear(t: np.ndarray, err: Optional[np.ndarray],
               oc: np.ndarray, plot=False) -> Tuple[float, float]:
    """
    Fit a linear trend to O-C values, return the slope and intercept
    A simple two-steps outliers cleaning procedure.
    The goal -- to make O-C parabola look more symmetrical
    """
    if err is not None:
        # todo: light tweak to control part of O-C with big errors:
        exp = 0.75
        weights = 1.0 / np.pow(np.where(err == 0, 1e-6, err), exp)
        coeffs = np.polyfit(t, oc, 1, w=weights)
    else:
        weights = None
        coeffs = np.polyfit(t, oc, 1)
    slope, intercept = coeffs

    # Simple 3-sigma outlier rejection
    residuals = oc - (slope * t + intercept)
    std_residuals = np.std(residuals)
    inlier_mask = np.abs(residuals) < 3 * std_residuals

    t_inliers = t[inlier_mask]
    oc_inliers = oc[inlier_mask]
    if weights is not None:
        w_inliers = weights[inlier_mask]
        slope_final, intercept_final = np.polyfit(t_inliers, oc_inliers, 1, w=w_inliers)
    else:
        slope_final, intercept_final = np.polyfit(t_inliers, oc_inliers, 1)

    if plot:
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


def fit_quadratic_oc(cycles: np.ndarray,
                     oc: np.ndarray,
                     err: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Fits an O-C vs cycle number distribution with a parabolic 2nd order polynomial."""
    if err is not None:
        weights = 1.0 / np.where(err == 0, 1e-6, err)
        coeffs, cov = np.polyfit(cycles, oc, 2, w=weights, cov=True)
    else:
        coeffs, cov = np.polyfit(cycles, oc, 2, cov=True)
    return coeffs, cov


def fit_linear_oc(cycles: np.ndarray,
                  oc: np.ndarray,
                  err: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Fits an O-C vs cycle number distribution with a linear fit"""
    if err is not None:
        weights = 1.0 / np.where(err == 0, 1e-6, err)
        coeffs, cov = np.polyfit(cycles, oc, 1, w=weights, cov=True)
    else:
        coeffs, cov = np.polyfit(cycles, oc, 2, cov=True)
    return coeffs, cov


def correct_period(minima: np.ndarray, err: Optional[np.ndarray],
                   period: float, epoch: float, max_iter: int = 1,
                   tol: float = 1e-6) -> Tuple[float, float]:
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

        new_period = current_period / (1 - slope)
        new_epoch = current_epoch + intercept

        print(f"Iteration {i + 1}:")
        print(f"  Slope: {slope:.8f} | New Period: {new_period:.8f} | New Epoch: {new_epoch:.6f}")

        current_period = new_period
        current_epoch = new_epoch

        if abs(slope) < tol:
            break

    return current_period, current_epoch


def report_period_change(a: float, a_err: float, P: float, p_err: float):
    """
    Calculates the dimensionless period change rate dP/dt = (2 * a) / P and its error
    """

    # Dimensionless Rate (dP/dt)
    rate = (2.0 * a) / P

    # Error Propagation (Quadrature)
    # Relative errors
    rel_err_a = a_err / abs(a)
    rel_err_p = p_err / P

    # Combined relative error
    rel_err_total = np.sqrt(rel_err_a ** 2 + rel_err_p ** 2)

    # Absolute error in the rate
    rate_err = abs(rate) * rel_err_total

    # 3. Conversion to seconds per year (for human readability)
    sec_per_year_const = 31557600
    spy = rate * sec_per_year_const
    spy_err = rate_err * sec_per_year_const

    print("\n" + "=" * 40)
    print("PERIOD CHANGE ANALYSIS (Full Error Budget)")
    print("=" * 40)
    print(f"Rate (dP/dt):  {rate:.4e} ± {rate_err:.4e}")
    print(f"Change/Year:   {spy:.4f} ± {spy_err:.4f} seconds/year")
    print("-" * 40)
    print(f"Contribution from a: {rel_err_a * 100:.2f}%")
    print(f"Contribution from P: {rel_err_p * 100:.2f}%")
    print("=" * 40)


#  ==================  PHASE FOLDING stuff  =================

def fold_lightcurve(jd: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Fold Julian dates onto [0, 1) phase given a period and initial epoch."""
    return ((jd - epoch) / period) % 1.0


def fold_lightcurve_with_oc(
        # region fold
        jd: np.ndarray,
        mag: np.ndarray,
        mag_err: Optional[np.ndarray],
        user_period: float,
        user_epoch: float,
        a: float,
        b: float,
        c: float
        # endregion
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Phase-folds a lightcurve by applying a quadratic O-C (Observed minus Calculated)
    correction to account for secular period variations.

    Derivation:
    ----------------------------------
    1. Calculated (T_calc) = T_0 + P_0 * E
    2. O-C = Observed - Calculated = a*E^2 + b*E + c
    3. JD = (T_0 + P_0 * E) + (a*E^2 + b*E + c)
          = a*E^2 + (P_0 + b)*E + (T_0 + c)
    4. Solving for continuous cycle number E:
          a*E^2 + (P_0 + b)*E + (T_0 + c - JD) = 0

    Parameters
    ----------
    jd : np.ndarray
        Array of observation Julian Dates (pre-cut/filtered).
    mag : np.ndarray
        Array of observed magnitudes or fluxes.
    mag_err : np.ndarray or None
        Array of 1-sigma uncertainties. Passed through unaltered.
    user_period : float
        The reference period (P_0) used to build the baseline linear ephemeris.
    user_epoch : float
        The reference zero-point epoch (T_0) used to build the baseline linear ephemeris.
    a : float
        Quadratic polynomial coefficient from the O-C curve fit (coefficient of E^2).
    b : float
        Linear polynomial coefficient from the O-C curve fit (coefficient of E).
    c : float
        Constant polynomial coefficient from the O-C curve fit (y-intercept).

    Returns
    -------
    phases : np.ndarray
        Dynamically corrected phases mapped onto the standard fractional interval [0.0, 1.0).
    jd : np.ndarray
        The input Julian Dates array, preserved and returned for downstream plotting/tracking.
    mag : np.ndarray
        The input magnitudes/fluxes array.
    mag_err : np.ndarray or None
        The input 1-sigma uncertainties array.
    """
    jd_arr = np.asarray(jd, dtype=float)
    mag_arr = np.asarray(mag, dtype=float)

    # Validate alignment across core observational vectors
    if jd_arr.shape != mag_arr.shape:
        raise ValueError(
            f"Shape mismatch between time and observation vectors: "
            f"jd shape {jd_arr.shape} does not match mag shape {mag_arr.shape}."
        )

    if mag_err is not None:
        mag_err_arr = np.asarray(mag_err, dtype=float)
        if jd_arr.shape != mag_err_arr.shape:
            raise ValueError(
                f"Shape mismatch in uncertainty vector: "
                f"mag_err shape {mag_err_arr.shape} does not match jd/mag shape {jd_arr.shape}."
            )
    else:
        mag_err_arr = None

    # Update the ephemeris base elements using your O-C fit shifts
    corrected_epoch = user_epoch + c
    corrected_period = user_period + b

    # Solve the quadratic equation for continuous cycle numbers (E)
    # Coeffs: A = a, B = corrected_period, C = corrected_epoch - jd_arr
    if np.isclose(a, 0.0, atol=1e-16):
        continuous_cycles = (jd_arr - corrected_epoch) / corrected_period
    else:
        discriminant = (corrected_period ** 2) - 4.0 * a * (corrected_epoch - jd_arr)
        if np.any(discriminant < 0):
            raise ValueError("Discriminant < 0. Verify timing coordinate anchors.")
        continuous_cycles = (-corrected_period + np.sqrt(discriminant)) / (2.0 * a)

    phases = continuous_cycles % 1.0

    return jd_arr, phases, mag_arr, mag_err_arr


# ============  PERIODICITY ANALYSIS PIPELINES (AoV) ===========

"""
Analysis of Variance (AoV) periodicity test for folded lightcurves.

Implements the Schwarzenberg-Czerny (1989) AoV statistic, which tests whether
a folded lightcurve shows statistically significant phase-dependent variability
against the null hypothesis of a constant (flat) lightcurve.

Reference:
    Schwarzenberg-Czerny, A. (1989), MNRAS, 241, 153
    "On the advantage of using analysis of variance for period search"
"""


@dataclass
class AoVResult:
    """
    Results from the AoV / Schwarzenberg-Czerny test.

    Reference:
        Schwarzenberg-Czerny, A. (1989), MNRAS, 241, 153
        "On the advantage of using analysis of variance for period search"

    Attributes
    ----------
    aov_statistic : float
        The AoV (F-like) test statistic. Larger values indicate stronger
        phase-dependent variability relative to within-bin scatter.
    f_statistic : float
        The classical F-statistic (same as aov_statistic for the unweighted
        case; provided explicitly for clarity).
    p_value : float
        Two-sided p-value from the F-distribution with (n_bins-1, N-n_bins)
        degrees of freedom. Small p-value → reject flat/null hypothesis.
    n_sigma : float
        equivalent Gaussian significance
    df_between : int
        Degrees of freedom between bins (= n_bins - 1).
    df_within : int
        Degrees of freedom within bins (= N - n_bins, where N = number of
        usable observations).
    n_obs : int
        Total number of observations used.
    n_bins : int
        Number of phase bins used.
    bin_phases : np.ndarray
        Phase centres of each bin.
    bin_means : np.ndarray
        Weighted (or unweighted) mean magnitude in each bin.
    bin_edges: float
        stuff for step-function visualization
    bin_stds : np.ndarray
        Standard deviation of magnitudes within each bin.
    bin_counts : np.ndarray
        Number of observations in each bin.
    grand_mean : float
        Weighted (or unweighted) grand mean magnitude over all observations.
    ss_between : float
        Sum of squares between bins (signal variance × weights).
    ss_within : float
        Sum of squares within bins (noise variance × weights).
    ms_between : float
        Mean square between bins = ss_between / df_between.
    ms_within : float
        Mean square within bins  = ss_within  / df_within.
    is_significant : bool
        True if p_value < significance_level.
    significance_level : float
        The significance level used for the is_significant flag.
    weighted : bool
        True if observational weights (from mag_err) were used

    """
    aov_statistic: float
    f_statistic: float
    p_value: float
    n_sigma: float
    df_between: int
    df_within: int
    n_obs: int
    n_bins: int
    bin_phases: np.ndarray
    bin_means: np.ndarray
    bin_edges: np.ndarray
    bin_stds: np.ndarray
    bin_counts: np.ndarray
    grand_mean: float
    ss_between: float
    ss_within: float
    ms_between: float
    ms_within: float
    is_significant: bool
    significance_level: float
    weighted: bool

    def __str__(self) -> str:
        sig_str = "SIGNIFICANT" if self.is_significant else "not significant"
        weight_str = "weighted (using mag_err)" if self.weighted else "unweighted"
        lines = [
            "=" * 60,
            "  AoV / Schwarzenberg-Czerny Test Result",
            "=" * 60,
            f"  Mode              : {weight_str}",
            f"  Observations used : {self.n_obs}",
            f"  Phase bins        : {self.n_bins}",
            f"  Grand mean mag    : {self.grand_mean:.4f}",
            "-" * 60,
            f"  AoV / F statistic : {self.aov_statistic:.4f}",
            f"  df (between/within): ({self.df_between}, {self.df_within})",
            f"  MS between bins   : {self.ms_between:.6f}",
            f"  MS within bins    : {self.ms_within:.6f}",
            f"  p-value           : {self.p_value:.2e}" if self.p_value > 1e-4 else
            f"  p-value           : < 1e-4 (Log-space tracking)",
            f"  n_sigma           : {self.n_sigma:.2f}σ",
            "-" * 60,
            f"  Result @ α={self.significance_level}: {sig_str}",
            "=" * 60,
        ]
        return "\n".join(lines)


def aov_test(
        # region fold-ready
        phases: np.ndarray,
        mag: np.ndarray,
        n_bins: int = 10,
        mag_err: Optional[np.ndarray] = None,
        significance_level: float = 0.01,
        min_points_per_bin: int = 2,
        # endregion
) -> AoVResult:
    """
    Perform the Schwarzenberg-Czerny AoV (Analysis of Variance) test on a folded lightcurve.

    The test statistic is:

        AoV = MS_between / MS_within

    where MS = mean square (sum of squares / degrees of freedom).

    Under the null hypothesis (no phase-dependent signal), AoV follows an
    F-distribution with (n_bins - 1, N - n_bins) degrees of freedom.

    If mag_err is provided, observations are weighted by w_i = 1 / sigma_i^2,
    and the weighted generalization of the F-statistic is used, which gives
    more influence to high-quality observations.

    Parameters
    ----------
    phases : np.ndarray
        Pre-folded fractional observation phases mapped onto [0.0, 1.0).
    mag : np.ndarray
        Magnitudes (or fluxes) at each corresponding epoch phase point.
    n_bins : int, optional
        Number of equal-width phase bins. Default 10.
    mag_err : np.ndarray or None, optional
        1-sigma observational uncertainties. If None, unweighted AoV is used.
    significance_level : float, optional
        Alpha level for the is_significant flag. Default 0.01 (1%).
    min_points_per_bin : int, optional
        Bins with fewer points than this are excluded from the calculation.

    Returns
    -------
    AoVResult dataclass string format profile outputs.
    """
    # 1. Broad Shape and Alignment Validations
    phases = np.asarray(phases, dtype=float)
    mag = np.asarray(mag, dtype=float)

    if phases.shape != mag.shape:
        raise ValueError(
            f"Phase and magnitude arrays must have identical shapes! "
            f"Got phases {phases.shape} vs mag {mag.shape}"
        )

    weighted = mag_err is not None

    # Secure handling of weighted array edge cases (Bug Fix applied here)
    if weighted:
        mag_err = np.asarray(mag_err, dtype=float)
        if mag_err.shape != phases.shape:
            raise ValueError(
                f"Uncertainty array shape mismatch! "
                f"Got mag_err {mag_err.shape} while expecting {phases.shape}"
            )

        nan_count = np.isnan(mag_err).sum()
        if nan_count > 0:
            nan_percentage = (nan_count / len(mag_err)) * 100.0
            max_nan_percentage = 30.0
            if nan_percentage < max_nan_percentage:
                penalty_error = np.nanpercentile(mag_err, 90)
                mag_err = mag_err.copy()  # Avoid modifying original array in-place
                mag_err[np.isnan(mag_err)] = penalty_error
            else:
                raise ValueError(f"Too many missing uncertainties ({nan_percentage:.1f}% NaNs).")

        if np.any(mag_err <= 0):
            raise ValueError("All mag_err values must be strictly greater than 0.")

    # Clean finite observations
    mask = np.isfinite(phases) & np.isfinite(mag)
    if weighted:
        mask &= np.isfinite(mag_err) & (mag_err > 0)

    phase_clean = phases[mask]
    mag_clean = mag[mask]
    err_clean = mag_err[mask] if weighted else None
    weights = (1.0 / err_clean ** 2) if weighted else np.ones(len(phase_clean))

    if len(phase_clean) < n_bins * min_points_per_bin:
        raise ValueError(f"Too few valid observations ({len(phase_clean)}) for {n_bins} bins.")

    # Bin Assignment Mapping
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(phase_clean, bin_edges) - 1, 0, n_bins - 1)

    # We need these bin edges to plot "step function"
    bin_phases = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.full(n_bins, np.nan)
    bin_stds = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_ws = np.zeros(n_bins)

    # Extract Per-Bin Statistics
    for b in range(n_bins):
        idx = (bin_indices == b)
        n_b = idx.sum()
        bin_counts[b] = n_b

        if n_b < min_points_per_bin:
            continue

        w_b = weights[idx]
        m_b = mag_clean[idx]
        w_sum = w_b.sum()
        bin_ws[b] = w_sum

        mu_b = np.sum(w_b * m_b) / w_sum
        bin_means[b] = mu_b

        if n_b >= 2:
            sum_w2 = np.sum(w_b ** 2)
            denom = w_sum - (sum_w2 / w_sum) if weighted else (n_b - 1)
            var_b = np.sum(w_b * (m_b - mu_b) ** 2) / (denom if denom > 0 else 1)
            bin_stds[b] = np.sqrt(max(0.0, var_b))

    # Filter out underpopulated Bins
    good_bins = bin_counts >= min_points_per_bin
    n_good = good_bins.sum()
    if n_good < 2:
        raise ValueError(f"Only {n_good} bins meet the min_points_per_bin threshold.")
    if n_good < n_bins:
        warnings.warn(f"{n_bins - n_good} bins were excluded due to low population.", UserWarning, stacklevel=2)

    good_obs_mask = good_bins[bin_indices]
    N_used = good_obs_mask.sum()

    # --- Exact Weighted Grand Mean (of the model-included subset) ---
    grand_mean = np.sum(bin_ws[good_bins] * bin_means[good_bins]) / bin_ws[good_bins].sum()
    ss_between = np.sum(bin_ws[good_bins] * (bin_means[good_bins] - grand_mean) ** 2)

    ss_within = 0.0
    for b in np.where(good_bins)[0]:
        idx = (bin_indices == b)
        ss_within += np.sum(weights[idx] * (mag_clean[idx] - bin_means[b]) ** 2)

    # Degrees of Freedom and Fit Calculations
    df_between = n_good - 1
    df_within = N_used - n_good

    if df_within <= 0:
        raise ValueError(f"Degrees of freedom error: df_within = {df_within}.")

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within

    # --- Mathematical Transformation to Significance Profile ---
    # Log survival function avoids precision loss under high-amplitude signals
    log_p = f_dist.logsf(f_stat, df_between, df_within)
    p_value = np.exp(log_p)

    # Accurate one-tailed mapping from log-probability to Gaussian Sigmas
    sigmas_log = -ndtri(p_value) if p_value > 0.0 else -ndtri(np.exp(log_p))
    if log_p > -1e-12:
        sigmas_log = 0.0

    result = AoVResult(
        aov_statistic=f_stat, f_statistic=f_stat, p_value=p_value, n_sigma=sigmas_log,
        df_between=df_between, df_within=df_within, n_obs=N_used, n_bins=n_good,
        bin_phases=bin_phases, bin_means=bin_means, bin_stds=bin_stds, bin_counts=bin_counts,
        grand_mean=grand_mean, ss_between=ss_between, ss_within=ss_within,
        ms_between=ms_between, ms_within=ms_within, is_significant=log_p < np.log(significance_level),
        significance_level=significance_level, weighted=weighted, bin_edges=bin_edges
    )

    return result


# =============== VISUALIZATION (DEBUG  AND PUBLICATION GRAPHICS) ==============


def plot_aov_folded_lightcurve(
        # region unfold
        jd: np.ndarray,
        mag: np.ndarray,
        result: AoVResult,
        period: float,
        epoch: float,
        mag_err: Optional[np.ndarray] = None,
        plot_twice: bool = True
        # endregion
) -> None:
    """Plots the phase-folded light curve overlaid with the AoV step model."""
    mask = np.isfinite(jd) & np.isfinite(mag)
    if mag_err is not None:
        mask &= np.isfinite(mag_err) & (mag_err > 0)

    jd_c = jd[mask]
    mag_c = mag[mask]
    err_c = mag_err[mask] if mag_err is not None else None

    phase = fold_lightcurve(jd_c, period=period, epoch=epoch)

    edges = result.bin_edges
    means = result.bin_means
    clean_means = np.where(np.isnan(means), np.nan, means)

    step_phases = edges
    step_mags = np.append(clean_means, clean_means[-1])

    if plot_twice:
        plot_phase = np.concatenate([phase, phase + 1.0])
        plot_mag = np.concatenate([mag_c, mag_c])
        plot_err = np.concatenate([err_c, err_c]) if err_c is not None else None
        step_phases_plot = np.concatenate([step_phases[:-1], step_phases + 1.0])
        step_mags_plot = np.concatenate([clean_means, step_mags])
    else:
        plot_phase = phase
        plot_mag = mag_c
        plot_err = err_c
        step_phases_plot = step_phases
        step_mags_plot = step_mags

    plt.figure(figsize=(14, 8))

    if plot_err is not None:
        plt.errorbar(
            plot_phase, plot_mag, yerr=plot_err,
            fmt='o', color='gray', markersize=3, alpha=0.4,
            label='Observations', elinewidth=0.5
        )
    else:
        plt.scatter(
            plot_phase, plot_mag,
            color='gray', s=8, alpha=0.5, label='Observations'
        )

    plt.step(
        step_phases_plot, step_mags_plot,
        where='post', color='crimson', linewidth=2.5, zorder=5,
        label=f'AoV Step Model'
    )

    plt.axhline(
        result.grand_mean, color='black', linestyle='--', linewidth=1, alpha=0.7,
        label=f'Grand Mean ({result.grand_mean:.4f})'
    )

    plt.gca().invert_yaxis()
    plt.xlim(0.0, 2.0 if plot_twice else 1.0)
    plt.xlabel('Orbital Phase')
    plt.ylabel('Magnitude / Flux')
    plt.title(
        f'AoV Phase-Folded Model (P = {period:.5f} d)\n'
        f'F-Stat: {result.f_statistic:.2f} | Significance: {result.n_sigma:.1f}σ'
    )
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_oc_diagram(
        # region fold
        cycles: np.ndarray,
        oc: np.ndarray,
        jd_err: np.ndarray,
        a: float,
        b: float,
        c: float,
        user_period: float,
        user_epoch: float,
        mode: str = "debug",
        filename_base: str = "oc_diagram"
        # endregion
) -> None:
    """
    Plots an advanced O-C Parabolic Diagram featuring analytical curve fits,
    timing uncertainty profiles, dual timescale horizontal axes (Cycles vs JD),
    and dual residual scales (Days vs Minutes) for paper presentation.

    Parameters
    ----------
    cycles : np.ndarray
        Array of integer or continuous cycle numbers (E).
    oc : np.ndarray
        Calculated O-C residual data points in units of days.
    jd_err : np.ndarray
        1-sigma timing uncertainty measurements for error bar tracking.
    a : float
        Quadratic polynomial coefficient from the O-C parabola fit.
    b : float
        Linear polynomial coefficient from the O-C parabola fit.
    c : float
        Constant offset/y-intercept from the O-C parabola fit.
    user_period : float
        The underlying reference baseline period (P_0) of the system.
    user_epoch : float
        The underlying reference baseline zero-point epoch (T_0).
    mode : str, default "debug"
        Selection profile toggling between 'debug' layout and 'publication' layout.
    filename_base : str, default "oc_diagram"
        Root file string used to export high-DPI vector graphics files.
    """
    # 1. Setup global plotting arrays and smooth analytical curve calculations
    cycles_clean = np.asarray(cycles, dtype=float)
    oc_clean = np.asarray(oc, dtype=float)
    err_clean = np.asarray(jd_err, dtype=float)

    E_fine = np.linspace(np.min(cycles_clean), np.max(cycles_clean), 300)
    oc_fit = a * E_fine ** 2 + b * E_fine + c
    dp_p = (2.0 * a) / user_period

    # ------------- OPTION A: DEBUGGING VIEW --------------

    if mode.lower() == "debug":
        plt.figure(figsize=(16, 10))

        # Plot O-C with vibrant error bars
        plt.errorbar(
            cycles_clean, oc_clean, yerr=err_clean, fmt='bo',
            capsize=5, label='Observed O-C', markersize=10
        )

        # Plot the analytical curve tracker
        plt.plot(
            E_fine, oc_fit, 'r-', lw=3,
            label=f'Fit\n$dP/P={dp_p:.2e}$'
        )

        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        plt.xlabel("Cycle Number (E)")
        plt.ylabel("O-C [days]")
        plt.title("O-C Parabolic Approximation (Debug View)")
        plt.legend(loc='best')
        plt.grid(True, which='both', linestyle=':', alpha=0.5)

        # Dual Upper Time Scale: Convert Cycle limits back to standard JD scales
        ax1 = plt.gca()
        ax_jd = ax1.twiny()
        m1, m2 = ax1.get_xlim()
        ax_jd.set_xlim(user_period * m1 + user_epoch, user_period * m2 + user_epoch)
        ax_jd.set_xlabel("Time [JD]", labelpad=15)

        plt.tight_layout()
        plt.show()

    # ---------- OPTION B: PUBLICATION VIEW -----------------

    elif mode.lower() == "publication":
        # Apply strict publication typography overrides
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "lines.linewidth": 2,
            "axes.linewidth": 1.2
        })

        fig, ax1 = plt.subplots(figsize=(10, 7))

        # Clean black & white markers with error tracking lines
        ax1.errorbar(
            cycles_clean, oc_clean, yerr=err_clean, fmt='o',
            color='black', mfc='white', mec='black', mew=1.5,
            capsize=0, label='Observed $O-C$', markersize=8, zorder=3
        )

        # Solid specialized fit tracker rendering line
        ax1.plot(
            E_fine, oc_fit, color='red', linestyle='-', lw=2.5,
            label=f'Parabolic Fit\n$\\dot{{P}}/P = {dp_p:.2e}$', zorder=2
        )

        ax1.set_xlabel("Cycle Number ($E$)")
        ax1.set_ylabel("$O-C$ [days]")
        ax1.spines['right'].set_visible(False)

        # Cleaner structured caption positioning block
        ax1.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 0.98),
            frameon=False,
            ncol=2,
            columnspacing=1.0
        )

        # Secondary Vertical Axis: Days map cleanly over onto Minutes scale
        ax2 = ax1.twinx()
        # 1 day = 1440 minutes
        # ax2.set_ylim(np.array(ax1.get_ylim()) * 1440.0)  # type: ignore
        # ax2.set_ylabel("$O-C$ [minutes]", rotation=270, labelpad=20)
        # ax2.spines['top'].set_visible(False)

        # Secondary Top Axis: Maps Cycle space onto absolute Julian Dates
        ax_jd = ax1.twiny()
        m1, m2 = ax1.get_xlim()
        ax_jd.set_xlim(user_period * m1 + user_epoch, user_period * m2 + user_epoch)
        ax_jd.set_xlabel("Time [JD]", labelpad=15)

        plt.tight_layout()

        # Save output configurations directly to file paths
        pdf_out = f"{filename_base}.pdf"
        png_out = f"{filename_base}.png"
        plt.savefig(pdf_out, dpi=300)
        plt.savefig(png_out, bbox_inches='tight', dpi=300)
        plt.show()
        print(f"[Export] Saved O-C plots to '{pdf_out}' and '{png_out}'")


def plot_simple_oc(minima: np.ndarray, oc_values: np.ndarray, jd_err: np.ndarray | None, title='O-C'):
    plt.figure(figsize=(16, 10))
    if jd_err is None:
        plt.scatter(minima, oc_values, color='b', label='O-C values')
    else:
        plt.errorbar(minima, oc_values, yerr=jd_err, fmt='bo')
    plt.axhline(0, color='r', linestyle='--', lw=1)
    plt.xlabel("Time [JD]")
    plt.ylabel("O-C [days]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_piecewise_oc_with_period(
        # region fold
        cycles: np.ndarray,
        oc: np.ndarray,
        jd_err: np.ndarray,
        user_period: float,
        user_epoch: float,
        para_coeffs: Optional[Tuple[float, float, float]] = None,
        para_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
        line_coeffs: Optional[Tuple[float, float]] = None,
        line_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
        mode: str = "debug",
        title: str = "O-C and Period Evolution",
        filename_base: str = "oc_piecewise_diagram"
        # endregion
) -> None:
    """
    Plots O-C points alongside separate piecewise linear/parabolic fits,
    and overlays the instantaneous orbital period line using a secondary Y-axis.
    Supports both 'debug' and 'publication' layout profiles.
    """
    # 1. Period Plot Styling Configurations
    period_line_color = "forestgreen"
    period_line_style = "-"  # Solid line style
    period_line_width = 2.5

    # 2. Setup typography standards based on operating mode
    if mode.lower() == "publication":
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "lines.linewidth": 2,
            "axes.linewidth": 1.2
        })
        fig, ax_oc = plt.subplots(figsize=(10, 7))
    else:
        fig, ax_oc = plt.subplots(figsize=(13, 8))

    ax_p = ax_oc.twinx()  # The secondary right axis for the instantaneous period line

    cycles_clean = np.asarray(cycles, dtype=float)
    oc_clean = np.asarray(oc, dtype=float)
    err_clean = np.asarray(jd_err, dtype=float)
    c_min, c_max = np.min(cycles_clean), np.max(cycles_clean)

    # 3. Plot Empirical O-C Data Points
    if mode.lower() == "publication":
        ax_oc.errorbar(
            cycles_clean, oc_clean, yerr=err_clean, fmt='o',
            color='black', mfc='white', mec='black', mew=1.5,
            capsize=0, label='Observed $O-C$', markersize=7, zorder=4
        )
    else:
        ax_oc.errorbar(
            cycles_clean, oc_clean, yerr=err_clean, fmt='bo',
            capsize=5, label='Observed O-C', markersize=8, zorder=4
        )

    period_plot_elements = []

    # 4. Render Parabolic Segment Model Fit
    if para_coeffs is not None:
        a, b, c = para_coeffs
        p_start = para_bounds[0] if (para_bounds and para_bounds[0] is not None) else c_min
        p_end = para_bounds[1] if (para_bounds and para_bounds[1] is not None) else c_max

        E_para = np.linspace(p_start, p_end, 200)
        oc_para = a * E_para ** 2 + b * E_para + c
        ax_oc.plot(E_para, oc_para, color='crimson', linestyle='-', lw=2.5,
                   label='Parabolic Fit', zorder=3)

        # Continuous instantaneous period track: P(E) = P_0 + 2aE + b
        P_para = user_period + 2.0 * a * E_para + b
        period_plot_elements.append((E_para, P_para))

    # 5. Render Linear Segment Model Fit
    if line_coeffs is not None:
        slope, intercept = line_coeffs
        l_start = line_bounds[0] if (line_bounds and line_bounds[0] is not None) else c_min
        l_end = line_bounds[1] if (line_bounds and line_bounds[1] is not None) else c_max

        E_line = np.linspace(l_start, l_end, 200)
        oc_line = slope * E_line + intercept

        line_style = '--' if mode.lower() == "publication" else '--'
        ax_oc.plot(E_line, oc_line, color='darkorange', linestyle=line_style, lw=2.5,
                   label='Linear Fit', zorder=3)

        # Constant period track across stable regime: P(E) = P_0 + slope
        P_line = np.full_like(E_line, user_period + slope)
        # period_plot_elements.append((E_line, P_line)) # I do not want this

    # 6. Render the Continuous Instantaneous Period Track Elements
    for i, (E_seg, P_seg) in enumerate(period_plot_elements):
        lbl = 'Instantaneous Period $P(E)$' if i == 0 else ""
        ax_p.plot(
            E_seg, P_seg, color=period_line_color, linestyle=period_line_style,
            lw=period_line_width, label=lbl, zorder=2
        )

    # 7. Axis Styling and Label Setup
    ax_oc.set_xlabel("Cycle Number ($E$)")
    ax_oc.set_ylabel("$O-C$ [days]", color='black')
    ax_p.set_ylabel("Instantaneous Period $P$ [days]", color=period_line_color, rotation=270, labelpad=25)
    ax_p.tick_params(axis='y', labelcolor=period_line_color)

    if mode.lower() == "publication":
        ax_oc.spines['right'].set_visible(False)
        ax_p.spines['top'].set_visible(False)
        frame_display = False
    else:
        ax_oc.grid(True, linestyle=':', alpha=0.5)
        frame_display = True

    # 8. Unify Custom Dual-Axis Legends
    handles_oc, labels_oc = ax_oc.get_legend_handles_labels()
    handles_p, labels_p = ax_p.get_legend_handles_labels()
    ax_oc.legend(handles_oc + handles_p, labels_oc + labels_p, loc='upper left', frameon=frame_display)

    # 9. Top Horizontal Axis Setup for absolute Julian Dates (JD)
    ax_jd = ax_oc.twiny()
    m1, m2 = ax_oc.get_xlim()
    ax_jd.set_xlim(user_period * m1 + user_epoch, user_period * m2 + user_epoch)
    ax_jd.set_xlabel("Time [JD]", labelpad=12)
    if mode.lower() == "publication":
        ax_jd.spines['right'].set_visible(False)

    plt.title(title, pad=20 if mode.lower() == "debug" else 25)
    plt.tight_layout()

    if mode.lower() == "publication":
        plt.savefig(f"{filename_base}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
        print(f"[Export] Publication figures saved cleanly to {filename_base}.pdf/.png")

    plt.show()


def plot_debugging_folded_view(phases: np.ndarray, mag: np.ndarray, err: Optional[np.ndarray],
                               title: str = "Debug Folded View"):
    """Diagnostic plot displaying two full cycles to verify alignment."""
    plt.figure(figsize=(12, 7))
    p_double = np.concatenate([phases, phases + 1.0])
    m_double = np.concatenate([mag, mag])

    if err is not None:
        e_double = np.concatenate([err, err])
        plt.errorbar(p_double, m_double, yerr=e_double, fmt='.', color='royalblue', ecolor='lightsteelblue', alpha=0.6,
                     elinewidth=0.6, label='Data')
    else:
        plt.scatter(p_double, m_double, color='darkorange', s=8, alpha=0.6, label='Data')

    plt.gca().invert_yaxis()
    plt.xlim(0.0, 2.0)
    plt.axvline(1.0, color='crimson', linestyle='--', alpha=0.5)
    plt.xlabel("Phase (Wrapped 2x)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_aov_step_model(
        # region fold
        phases: np.ndarray,
        mag: np.ndarray,
        aov_result: AoVResult,
        mag_err: Optional[np.ndarray] = None,
        model_style: str = "step",
        mode: str = "debug",
        ylim: Optional[Tuple[float, float]] = None,
        title_suffix: str = "",
        filename_base: str = "aov_profile"
        # endregion
) -> None:
    """
    Plots a phase-folded light curve across two full cycles, overlaid with either
    the standard AoV step model or a broken line tracking bin centers with errors on the mean.
    Supports both 'debug' and 'publication' layout profiles.

    Parameters
    ----------
    phases : np.ndarray
        Pre-folded fractional observation phases mapped onto [0.0, 1.0).
    mag : np.ndarray
        Observed magnitudes or fluxes.
    aov_result : AoVResult
        The dataclass result output from the aov_test function.
    mag_err : np.ndarray or None, optional
        1-sigma observational uncertainties.
    model_style : str, default "step"
        Options: 'step' (staircase binned model) or 'broken_line' (linear joints with error on mean).
    mode : str, default "debug"
        Selection profile toggling between 'debug' layout and 'publication' layout.
    ylim : tuple of (float, float) or None, optional
        Explicit limits for the Y-axis (magnitude/flux). If provided, handles astronomical
        inversion automatically. If None, scales dynamically based on data extremes.
    title_suffix : str, optional
        Additional text to append to the plot title (e.g., target name).
    filename_base : str, default "aov_profile"
        Root file string used to export high-DPI vector graphics files in publication mode.
    """
    # 1. Setup typography and style standards based on operating mode
    if mode.lower() == "publication":
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "lines.linewidth": 2,
            "axes.linewidth": 1.2
        })
        fig, ax = plt.subplots(figsize=(10, 7))

        # Publication styling attributes
        obs_color = "darkgray"
        obs_alpha = 0.4
        model_color = "black"
        node_color = "black"
        grand_mean_color = "gray"
        frame_display = False
        grid_display = False
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        # Debug/Exploratory styling attributes
        obs_color = "gray"
        obs_alpha = 0.3
        model_color = "crimson"
        node_color = "darkred"
        grand_mean_color = "black"
        frame_display = True
        grid_display = True

    # 2. Clean finite observations
    mask = np.isfinite(phases) & np.isfinite(mag)
    if mag_err is not None:
        mask &= np.isfinite(mag_err) & (mag_err > 0)

    phases_c = phases[mask]
    mag_c = mag[mask]
    err_c = mag_err[mask] if mag_err is not None else None

    # Duplicate data over 2 cycles for phase wrap visualization
    plot_phase = np.concatenate([phases_c, phases_c + 1.0])
    plot_mag = np.concatenate([mag_c, mag_c])
    plot_err = np.concatenate([err_c, err_c]) if err_c is not None else None

    # 3. Plot underlying observation data points
    if plot_err is not None:
        plt.errorbar(
            plot_phase, plot_mag, yerr=plot_err, fmt='o', color=obs_color,
            markersize=3, alpha=obs_alpha, elinewidth=0.5, label='Observations', zorder=1
        )
    else:
        plt.scatter(
            plot_phase, plot_mag, color=obs_color, s=8, alpha=obs_alpha,
            label='Observations', zorder=1
        )

    # 4. Handle Model Selection
    if model_style.lower() == "step":
        edges = aov_result.bin_edges
        means = aov_result.bin_means
        clean_means = np.where(np.isnan(means), np.nan, means)

        step_phases = edges
        step_mags = np.append(clean_means, clean_means[-1])

        step_phases_plot = np.concatenate([step_phases[:-1], step_phases + 1.0])
        step_mags_plot = np.concatenate([clean_means, step_mags])

        plt.step(
            step_phases_plot, step_mags_plot, where='post', color=model_color,
            linewidth=2.5, zorder=5, label='AoV Step Model'
        )

    elif model_style.lower() == "broken_line":
        bin_centers = aov_result.bin_phases
        bin_means = aov_result.bin_means
        bin_stds = aov_result.bin_stds
        bin_counts = aov_result.bin_counts

        # Calculate standard error on the mean: sigma_mean = sigma / sqrt(N)
        with np.errstate(divide='ignore', invalid='ignore'):
            err_on_mean = bin_stds / np.sqrt(bin_counts)
        err_on_mean = np.where(np.isnan(err_on_mean), 0.0, err_on_mean)

        line_phases = np.concatenate([bin_centers, bin_centers + 1.0])
        line_mags = np.concatenate([bin_means, bin_means])
        line_errs = np.concatenate([err_on_mean, err_on_mean])

        # Plot the connected broken line path
        line_style = '-' if mode.lower() == "debug" else '--'
        plt.plot(
            line_phases, line_mags, color=model_color, linestyle=line_style,
            linewidth=2.5, zorder=4, label='Binned Mean Profile'
        )
        # Overlay standard error bars on the mean nodes
        plt.errorbar(
            line_phases, line_mags, yerr=line_errs, fmt='o', color=node_color,
            ecolor=node_color, mfc='white' if mode.lower() == "publication" else node_color,
            markersize=6, capsize=3 if mode.lower() == "debug" else 0, elinewidth=1.8,
            zorder=5, label=r'Bin Means ($\pm \sigma_{\mathrm{mean}}$)'
        )

    else:
        raise ValueError(f"Unknown model_style '{model_style}'. Use 'step' or 'broken_line'.")

    # 5. Global Canvas Aesthetics
    plt.axhline(
        aov_result.grand_mean, color=grand_mean_color, linestyle='--', linewidth=1, alpha=0.7,
        label=f'Grand Mean ({aov_result.grand_mean:.4f})'
    )
    plt.axvline(1, color=grand_mean_color, linestyle='--', linewidth=1, alpha=0.7)

    # plt.gca().invert_yaxis()  # Standard astronomical scaling
    plt.xlim(0.0, 2.0)
    plt.xlabel('Orbital Phase')
    plt.ylabel('Magnitude / Flux')

    # Apply Y-axis boundaries safely taking inversion into account
    if ylim is not None:
        y_val1, y_val2 = ylim
        # For standard magnitudes, ensure numerical larger value (fainter) is at the bottom
        plt.ylim(max(y_val1, y_val2), min(y_val1, y_val2))
    else:
        plt.gca().invert_yaxis()  # Default behavior: dynamic inversion

    title_str = (
        f"AoV Phase Profile (Bins: {aov_result.n_bins})\n"
        f"F-Stat: {aov_result.f_statistic:.2f} | Significance: {aov_result.n_sigma:.1f}$\sigma$"
    )
    if title_suffix:
        title_str += f" | {title_suffix}"
    plt.title(title_str)

    plt.legend(loc='best', frameon=frame_display)

    if grid_display:
        plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # 6. Save output configurations directly if in publication mode
    if mode.lower() == "publication":
        plt.savefig(f"{filename_base}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
        print(f"[Export] Saved publication vector figures to {filename_base}.pdf/.png")

    plt.show()


def plot_publication_folded_view(phases: np.ndarray, mag: np.ndarray, err: Optional[np.ndarray],
                                 filename: str = "paper_figure.pdf"):
    """Clean, high-DPI publication vector layout matching paper standards (1.5 cycles)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    mask = phases < 0.5
    p_pub = np.concatenate([phases, phases[mask] + 1.0])
    m_pub = np.concatenate([mag, mag[mask]])

    if err is not None:
        e_pub = np.concatenate([err, err[mask]])
        ax.errorbar(p_pub, m_pub, yerr=e_pub, fmt='.', color='black', ecolor='silver', markersize=2.5, elinewidth=0.4,
                    capsize=0, alpha=0.7)
    else:
        ax.scatter(p_pub, m_pub, color='black', s=3, alpha=0.7)

    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.5)
    ax.set_xlabel(r"Phase $\phi$", fontsize=12)
    ax.set_ylabel("Magnitude (dmag)", fontsize=12)
    ax.tick_params(direction='in', top=True, right=True, labelsize=10)
    ax.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.3)

    fig.tight_layout()
    plt.show()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Paper Export] Vector figure saved cleanly as: '{filename}'")


def plot_debugging_unfolded_lightcurve(jd: np.ndarray, mag: np.ndarray):
    plt.figure(figsize=(14, 8))
    plt.scatter(
        jd, mag,
        color='gray', s=8, alpha=0.5, label='Observations'
    )
    plt.gca().invert_yaxis()
    plt.show()


def plot_unfolded_with_extrema(
        # region unfold
        jd_lc: np.ndarray,
        mag_lc: np.ndarray,
        mag_err_lc: Optional[np.ndarray] = None,
        jd_ext: Optional[np.ndarray] = None,
        jd_err_ext: Optional[np.ndarray] = None,
        title: str = "Unfolded Lightcurve with Extrema Moments"
        # endregion
) -> None:
    """
    Plots the full unfolded raw lightcurve time-series and overlays
    the specific isolated moments of extrema using horizontal timing error bars.

    Parameters
    ----------
    jd_lc : np.ndarray
        Julian Dates of the continuous lightcurve.
    mag_lc : np.ndarray
        Magnitudes or fluxes of the continuous lightcurve.
    mag_err_lc : np.ndarray, optional
        1-sigma uncertainties for the lightcurve magnitudes.
    jd_ext : np.ndarray, optional
        Julian Dates of the isolated timings of extrema.
    jd_err_ext : np.ndarray, optional
        1-sigma timing uncertainties (horizontal error bars) for the extrema.
    title : str
        The plot header title.
    """
    plt.figure(figsize=(20, 10))

    # 1. Clean and plot the underlying continuous lightcurve
    mask_lc = np.isfinite(jd_lc) & np.isfinite(mag_lc)
    j_lc = jd_lc[mask_lc]
    m_lc = mag_lc[mask_lc]

    if mag_err_lc is not None:
        e_lc = mag_err_lc[mask_lc]
        plt.errorbar(
            # j_lc, m_lc, yerr=e_lc, fmt='.', color='gainsboro',
            j_lc, m_lc, yerr=e_lc, fmt='.', color='blue',
            ecolor='gray', markersize=5, elinewidth=1,
            alpha=1.0, label='Lightcurve Data'
        )
    else:
        plt.scatter(
            j_lc, m_lc, color='blue', s=5,
            alpha=1.0, label='Lightcurve Data'
        )

    # 2. Interpolate or estimate magnitudes for the extrema moments to position them on the Y-axis
    if jd_ext is not None and len(jd_ext) > 0:
        jd_ext = np.asarray(jd_ext, dtype=float)

        # Interpolate along the raw lightcurve to place the extrema point perfectly on the track
        # Note: np.interp requires sorted x-coordinates
        sort_idx = np.argsort(j_lc)
        mag_ext_estimated = np.interp(jd_ext, j_lc[sort_idx], m_lc[sort_idx])

        # Setup horizontal error bars (xerr) if timing errors are available
        xerr_data = jd_err_ext if jd_err_ext is not None else None

        plt.errorbar(
            jd_ext, mag_ext_estimated, xerr=xerr_data, fmt='o',
            color='crimson', ecolor='crimson', markersize=7,
            elinewidth=2, capsize=3, zorder=10,
            label=f'Extrema Moments ($N={len(jd_ext)}$)'
        )

    plt.gca().invert_yaxis()  # Standard astronomical magnitude scaling
    plt.xlabel('Time (Julian Date)')
    plt.ylabel('Magnitude / Flux')
    plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


#  ========================= SELF-CONTAINED TESTING SUITE  ===============

def self_aov_test():
    """
    Automated multi-scenario self-test for the refactored AoV engine
    running systematically within a standardized configuration loop.
    """
    print("==========================================================")
    print("        AoV Pipeline Self-Test")
    print("==========================================================")
    rng = np.random.default_rng(42)

    # 1. Setup global simulation parameters (typical Cataclysmic Variable superhump)
    N = 500
    period_true = 0.076  # days
    epoch_true = 2459000.0
    baseline_days = 20.0
    amplitude_true = 0.05  # mag semi-amplitude
    base_noise_floor = 0.08  # mag reference mean uncertainty

    # Generate observational timestamps
    jd_sim = epoch_true + rng.uniform(0, baseline_days, N)

    # Generate realistic variable noise profiles (Chi-squared scaled errors)
    # This replaces the constant np.full array with authentic scattered heteroscedastic errors
    err_sim = base_noise_floor * np.sqrt(rng.chisquare(df=5, size=N) / 5.0)

    # 2. Define the test execution configurations matrix
    test_scenarios = [
        {
            "id": "Test 1: Signal Present (Unweighted View)",
            "use_signal": True,
            "test_period": period_true,
            "weighted": False,
            "style": "broken_line",
            "plot_mode": "debug"
        },
        {
            "id": "Test 2: Signal Present (Weighted View)",
            "use_signal": True,
            "test_period": period_true,
            "weighted": True,
            "style": "step",
            "plot_mode": "debug"
        },
        {
            "id": "Test 3: Pure Gaussian Noise Regime",
            "use_signal": False,
            "test_period": period_true,
            "weighted": True,
            "style": "step",
            "plot_mode": "debug"
        },
        {
            "id": "Test 4: Signal Present with Misaligned Period",
            "use_signal": True,
            "test_period": period_true * 1.12,  # 12% mismatch
            "weighted": True,
            "style": "broken_line",
            "plot_mode": "debug"
        }
    ]

    # 3. Process test scenarios matrix sequentially inside the execution loop
    for run in test_scenarios:
        print(f"\n--- Running {run['id']} ---")

        # Compute synthetic magnitude profiles based on current regime requirements
        if run["use_signal"]:
            phase_sim = ((jd_sim - epoch_true) / period_true) % 1.0
            mag_pure = 14.0 + amplitude_true * np.sin(2.0 * np.pi * phase_sim)
        else:
            mag_pure = np.full(N, 14.0)

        # Inject observation noise tailored individually via the custom variable errors vector
        mag_sim = mag_pure + rng.normal(0.0, err_sim)

        # Route weighting configuration parameter requirements
        active_errs = err_sim if run["weighted"] else None

        phases = fold_lightcurve(jd=jd_sim,
                                 period=run["test_period"],  # type: ignore
                                 epoch=epoch_true)

        # Run decoupled mathematical AoV analysis
        aov_result = aov_test(
            phases=phases,
            mag=mag_sim,
            mag_err=active_errs,
            n_bins=10,
            significance_level=0.01,
        )

        # Display raw calculation outputs summary
        print(
            f"  Result Summary -> F-Stat: {aov_result.f_statistic:.2f} | "
            f"Significance: {aov_result.n_sigma:.1f}sigma | "
            f"Significant: {aov_result.is_significant}"
        )

        # Call the updated decoupled layout plot engine
        plot_aov_step_model(
            phases=phases,
            mag=mag_sim,
            aov_result=aov_result,
            mag_err=active_errs,
            model_style=run["style"],  # type: ignore
            mode=run["plot_mode"],  # type: ignore
            title_suffix=f"Scenario verification loop profiling"
        )


def run_synthetic_verification():
    """Generates synthetic binary data with period change to verify the pipeline."""
    print("==========================================================")
    print("      RUNNING MODULE SELF-TEST ON SYNTHETIC DATA")
    print("==========================================================")

    rng = np.random.default_rng(42)
    n_obs = 600

    # Setup properties
    user_period = 0.0549984613
    user_epoch = 978.4875
    a = 1.486e-06
    b = 2.180e-05
    c = 0.000530

    jd_mock = np.sort(rng.uniform(1000, 2500, n_obs))
    p_corr = user_period + b
    e_corr = user_epoch + c

    # Solve quadratic step to build real physical phases
    disc = (p_corr ** 2) - 4.0 * a * (e_corr - jd_mock)
    E_mock = (-p_corr + np.sqrt(disc)) / (2.0 * a)
    phases_true = E_mock % 1.0

    # Emulate variable profile with scatter noise
    mag_mock = 15.0 + 0.25 * np.sin(2.0 * np.pi * phases_true) + rng.normal(0, 0.03, n_obs)
    err_mock = np.full_like(mag_mock, 0.03)

    print(f"Generated {n_obs} fake data points over a 1500-day baseline.")

    # Step 1: Run the smart quadratic phase folding algorithm
    jds, phases, mags, errs = fold_lightcurve_with_oc(
        jd_mock, mag_mock, err_mock,
        user_period=user_period, user_epoch=user_epoch,
        a=a, b=b, c=c
    )

    # Step 2: Validate periodic signal significance using AoV
    # phases_mock = fold_lightcurve(jd_mock, period=user_period + b, epoch=user_epoch + c)
    aov_res = aov_test(phases, mags, n_bins=12, mag_err=errs)
    plot_aov_step_model(phases, mags, aov_result=aov_res, mag_err=errs, model_style='broken_line')
    print(aov_res)

    # Step 3: Run diagnostics and publication plot exports
    plot_debugging_folded_view(phases, mags, errs, title="Module Verification Debug View")
    plot_publication_folded_view(phases=phases, mag=mags, err=errs, filename="Test_Synthetic_Publication.pdf")
    # phases_mock = fold_lightcurve(jd_mock, period=user_period + b, epoch=user_epoch + c)
    # plot_aov_step_model(phases_mock, mag_mock, aov_res, mag_err=err_mock)

    # Step 4: Validate export formats
    export_folded_data("Test_Smart_Phase_Output.dat", jds, phases, mags, errs)
    print("\n[Self-Test] All procedures verified successfully.")


# ================= Real data examples ============

def aov_realdata_test_1():
    filename_tess = '/home/voz/projects/UPJS/Shugarov/J0541/TESS/TESS_composite_tarasenkov.dat'
    T0_tess = 2458468.2856413433
    period = 0.06606354
    jd_full, mag_full, mag_err_full = load_lightcurve_astropy(filename_tess, jd_col=0, mag_col='flux',
                                                              err_col=None, file_format='commented_header')

    jd_min = None
    jd_max = None
    jd, mag, mag_err = cutout_data(jd=jd_full, mag=mag_full, mag_err=mag_err_full, jd_min=jd_min, jd_max=jd_max)
    phases = fold_lightcurve(jd=jd, period=period, epoch=T0_tess)
    result = aov_test(phases, mag, mag_err=mag_err, n_bins=10)
    plot_aov_step_model(phases, mag, aov_result=result, mag_err=mag_err, model_style='broken_line')


def aov_realdata_test_2():
    filename = 'data/TCP_J05415572-2308340/all_norm.dat'
    jd_full, mag_full, mag_err_full = load_lightcurve_astropy(filename, jd_col=0, mag_col='dmag',
                                                              err_col=None)
    jd, mag, mag_err = cutout_data(jd=jd_full, mag=mag_full, mag_err=mag_err_full, jd_min=60972.3, jd_max=60981)
    period = 0.05502
    epoch = 60971.5687
    phases = fold_lightcurve(jd=jd, period=period, epoch=epoch)
    result = aov_test(phases, mag, mag_err=mag_err, n_bins=10)
    plot_aov_step_model(phases, mag, aov_result=result, mag_err=mag_err)


def oc_realdata_test():
    extrema_filename = 'data/TCP_J05415572-2308340/results/all_extrema_sorted.dat'
    lightcurve_filename = 'data/TCP_J05415572-2308340/all_norm.dat'
    jd_full, mag_full, mag_err_full = load_lightcurve_astropy(lightcurve_filename,
                                                              jd_col=0, mag_col='dmag', err_col='err')
    jd_extrema, jd_err_extrema = load_extrema_txt(extrema_filename)

    jd_full -= 60000.0  # Our exiting lightcurve has different jd0 than extrema minima
    # TODO: That is why we ought to use volightcurve module always
    # plot_unfolded_with_extrema(jd_full, mag_full, mag_err_lc=mag_err_full,
    #                            jd_ext=jd_extrema, jd_err_ext=jd_err_extrema)
    period_0 = 0.0549984613
    epoch_0 = 978.4875
    # phases_full = fold_lightcurve(jd_full, period=period_0, epoch=epoch_0)
    # plot_debugging_folded_view(phases=phases_full, mag=mag_full, err=None, title="The whole interval")

    # ----- O-C stuff ----
    oc_0, cycles_0 = calculate_oc(jd_extrema, period=period_0, epoch=epoch_0, tweak_oc=-0.02)
    # Let's use for calculation only good data
    mask_2 = jd_extrema > 972  # for parabolic fit
    mask_1 = (jd_extrema > 971) & (jd_extrema < 973)  # for linear fit

    coeffs_2, cov_2 = fit_quadratic_oc(cycles=cycles_0[mask_2], oc=oc_0[mask_2], err=jd_err_extrema[mask_2])

    # here add all physics from oc_fit.py:
    a, b, c = coeffs_2
    a_err, b_err, c_err = np.sqrt(np.diag(cov_2))
    # #
    # # --- Calculate Minimum of Parabola ---
    # # For y = ax^2 + bx + c, the extremum is at x = -b / (2a)
    # # This represents the cycle number of the true minimum
    # E_min = -b / (2 * a)
    # #
    # # Propagation of error for E_min (simplified)
    # # sigma_E = |E_min| * sqrt((sig_b/b)^2 + (sig_a/a)^2)
    # E_min_err = abs(E_min) * np.sqrt((b_err / b) ** 2 + (a_err / a) ** 2)
    # #
    # # Convert cycle minimum back to JD time
    # t_min = epoch_0 + (E_min * period_0)
    # t_min_err = E_min_err * period_0
    # #
    # # Calculate the rate dP/dt
    # rate_dp_dt = (2 * a) / period_0
    # rate_err = (2 * a_err) / period_0
    # #
    # # Convert to seconds per year for a readable result
    # # (1 year = 31 557600 seconds = (365*24 + 6)*60*60)
    # seconds_per_year = rate_dp_dt * 31557600
    # seconds_per_year_err = rate_err * 31557600
    # #
    # print(f"Dimensionless rate (dP/dt): {rate_dp_dt:.4e} ± {rate_err:.4e}")
    # print(f"Period change: {seconds_per_year:.4f} ± {seconds_per_year_err:.4f} seconds/year")
    # --- Report ---
    # print("-" * 30)
    # print(f"FIT RESULTS (O-C = aE² + bE + c):")
    # print(f"a: {a:.4e} ± {a_err:.4e}")
    # print(f"b: {b:.4e} ± {b_err:.4e}")
    # print(f"c: {c:.4e} ± {c_err:.4e}")
    # print("-" * 30)
    # print(f"Time of Parabola Minimum (JD):")
    # print(f"T_min = {t_min:.6f} ± {t_min_err:.6f}")
    # print("-" * 30)
    #
    report_period_change(a, a_err, P=period_0, p_err=0.0001)

    coeffs_1, cov_1 = fit_linear_oc(cycles=cycles_0[mask_1], oc=oc_0[mask_1], err=jd_err_extrema[mask_1])

    a2, b2, c2 = coeffs_2
    a1, b1 = coeffs_1
    # plot_oc_diagram(cycles=cycles_0, oc=oc_0,
    #                 jd_err=jd_err_extrema,
    #                 a=a2, b=b2, c=c2, user_period=period_0, user_epoch=epoch_0,
    #                 mode="debug")

    plot_piecewise_oc_with_period(cycles_0, oc=oc_0, jd_err=jd_err_extrema,
                                  user_period=period_0, user_epoch=epoch_0,
                                  para_coeffs=(a2, b2, c2), para_bounds=(-105, None),
                                  line_coeffs=(a1, b1), line_bounds=(None, -105),
                                  mode="debug")
    return a2, b2, c2


def shift_phase_fold_realdata_test(a: float, b: float, c: float):
    lightcurve_filename = 'data/TCP_J05415572-2308340/all_norm.dat'
    jd_full, mag_full, mag_err_full = load_lightcurve_astropy(lightcurve_filename,
                                                              jd_col=0, mag_col='dmag', err_col='err')
    jd_full -= 60000.0  # Our exiting lightcurve has different jd0 than extrema minima
    # TODO: That is why we ought to use volightcurve module always

    period = 0.0549984613
    epoch = 978.4875
    for jd_min, jd_max in [(972.5, 976.5), (976.5, 980), (983, None)]:
        jd, mag, mag_err = cutout_data(jd=jd_full, mag=mag_full, mag_err=mag_err_full,
                                       jd_min=jd_min, jd_max=jd_max)
        _, smart_phases, clean_mags, clean_errs = fold_lightcurve_with_oc(
            jd=jd,
            mag=mag,
            mag_err=mag_err,
            user_period=period,
            user_epoch=epoch,
            a=a, b=b, c=c,
        )
        # plot_debugging_folded_view(phases=smart_phases, mag=clean_mags, err=clean_errs, title=f'jd:{jd_min}-{jd_max}')
        result = aov_test(smart_phases, clean_mags, mag_err=None, n_bins=10)
        plot_aov_step_model(phases=smart_phases, mag=clean_mags,
                            aov_result=result,
                            mag_err=clean_errs, model_style='broken_line',
                            ylim=(0.17, -0.17),
                            title_suffix=f'jd:{jd_min}-{jd_max}')

    jd_, smart_phases, clean_mags, clean_errs = fold_lightcurve_with_oc(
        jd=jd_full,
        mag=mag_full,
        mag_err=mag_err_full,
        user_period=period,
        user_epoch=epoch,
        a=a, b=b, c=c,
    )
    # plot_debugging_folded_view(phases=smart_phases, mag=clean_mags, err=clean_errs, title='jd:983-')

    export_folded_data('J0541.dat', jd_, smart_phases, clean_mags, clean_errs)


def period_correction_realdata_test():
    initial_p = 0.0549984613
    initial_epoch = 978.0
    filename = 'data/TCP_J05415572-2308340/results/all_extrema_sorted.dat'
    jd, err = load_extrema_txt(filename)

    # --- Step 1: Initial O-C State ---
    print("\n--- Initial O-C Analysis ---")
    oc_init, _ = calculate_oc(jd, initial_p, initial_epoch, tweak_oc=None)
    plot_simple_oc(jd, oc_init, err)

    # --- Step 2: Period and Epoch Correction ---
    print("\n--- Running Correction ---")
    corrected_p, corrected_epoch = correct_period(jd, err, initial_p, initial_epoch)
    print(f'corrected period={corrected_p} corrected epoch={corrected_epoch}')

    # --- Step 3: Final Verification ---
    print("\n--- Final O-C Verification ---")
    oc_final, _ = calculate_oc(jd, corrected_p, corrected_epoch)
    plot_simple_oc(jd, oc_final, err, title="Final O-C (Corrected)")

    print(f"\nFinal Results:")
    print(f"Period: {corrected_p:.10f} days")
    print(f"Epoch:  {corrected_epoch:.6f} JD")


if __name__ == "__main__":
    # period_correction_realdata_test()
    # self_aov_test()
    a_, b_, c_ = oc_realdata_test()
    # print(a_, b_, c_)
    # shift_phase_fold_realdata_test(1.4863704378859311e-06, 2.1802734995277483e-05, 0.0005300522755341337)

    # aov_realdata_test_1()
    # aov_realdata_test_2()
    # run_synthetic_verification()
