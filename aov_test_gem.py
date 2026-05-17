"""
Analysis of Variance (AoV) periodicity test for folded lightcurves.

Implements the Schwarzenberg-Czerny (1989) AoV statistic, which tests whether
a folded lightcurve shows statistically significant phase-dependent variability
against the null hypothesis of a constant (flat) lightcurve.

Reference:
    Schwarzenberg-Czerny, A. (1989), MNRAS, 241, 153
    "On the advantage of using analysis of variance for period search"
"""

import numpy as np
import warnings
from matplotlib import pyplot as plt
from scipy import stats
from scipy.special import ndtri
from scipy.stats import f as f_dist
from dataclasses import dataclass
from typing import Optional

plt.rcParams.update({'font.size': 14})  # 20 can sometimes cramp multi-panel plots


@dataclass
class AoVResult:
    """
    Results from the AoV / Schwarzenberg-Czerny test.

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


def fold_lightcurve(jd: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Fold Julian dates onto [0, 1) phase given a period and initial epoch."""
    return ((jd - epoch) / period) % 1.0


def aov_test(
        jd: np.ndarray,
        mag: np.ndarray,
        period: float,
        epoch: float,
        n_bins: int = 10,
        mag_err: Optional[np.ndarray] = None,
        significance_level: float = 0.01,
        min_points_per_bin: int = 2,
        plot: bool = True
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
    jd         : np.ndarray
        Julian dates of observations.
    mag        : np.ndarray
        Magnitudes (or fluxes) at each epoch.
    period     : float
        Folding period (same units as jd).
    epoch      : float
        Reference epoch (same units as jd). Phase 0 is assigned here.
    n_bins     : int, optional
        Number of equal-width phase bins. Default 10.
        Rule of thumb: at least ~5–10 observations per bin on average.
        Too many bins → sparse bins → unstable statistic.
        Too few bins → smears the signal.
    mag_err    : np.ndarray or None, optional
        1-sigma observational uncertainties. If None, unweighted AoV is used.
    significance_level : float, optional
        Alpha level for the is_significant flag. Default 0.01 (1%).
    min_points_per_bin : int, optional
        Bins with fewer points than this are excluded from the calculation.
        Default 2 (need at least 2 points to estimate within-bin variance).
    plot: bool
        Plot results. Or not

    Returns
    -------
    AoVResult dataclass (see class docstring for full field descriptions).

    Raises
    ------
    ValueError
        If inputs have inconsistent shapes, period <= 0, n_bins < 2, or if
        too many bins are underpopulated for the test to be meaningful.
    """

    # --- Input validation ---
    jd = np.asarray(jd, dtype=float)
    mag = np.asarray(mag, dtype=float)
    if jd.shape != mag.shape:
        raise ValueError(f"jd and mag must have the same shape, got {jd.shape} vs {mag.shape}")

    weighted = mag_err is not None

    if weighted:
        nan_count = np.isnan(mag_err).sum()
        nan_percentage = (nan_count / len(mag_err)) * 100   # type: ignore
        max_nan_percentage = 30
        if 0 < nan_percentage < max_nan_percentage:
            # Use the 90th percentile so we don't over-trust the missing-error data
            penalty_error = np.nanpercentile(mag_err, 90)
            mag_err[np.isnan(mag_err)] = penalty_error

            mag_err = np.asarray(mag_err, dtype=float)
            if mag_err.shape != mag.shape:
                raise ValueError(f"mag_err must have the same shape as mag, got {mag_err.shape}")
            if np.any(mag_err <= 0):
                raise ValueError("All mag_err values must be > 0")
        else:
            mag_err = None

    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    # --- Clean finite observations ---
    mask = np.isfinite(jd) & np.isfinite(mag)
    if weighted:
        mask &= np.isfinite(mag_err) & (mag_err > 0)

    jd_clean = jd[mask]
    mag_clean = mag[mask]
    err_clean = mag_err[mask] if weighted else None
    weights = (1.0 / err_clean ** 2) if weighted else np.ones(len(jd_clean))

    if len(jd_clean) < n_bins * min_points_per_bin:
        raise ValueError(f"Too few valid observations ({len(jd_clean)}) for {n_bins} bins.")

    # --- Phase folding & Binning setup ---
    phase = fold_lightcurve(jd_clean, period, epoch)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(phase, bin_edges) - 1, 0, n_bins - 1)

    # --- Container Allocations ---
    bin_phases = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.full(n_bins, np.nan)
    bin_stds = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_ws = np.zeros(n_bins)

    # --- Extract Per-Bin Statistics ---
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

        # Weighted sample mean
        mu_b = np.sum(w_b * m_b) / w_sum
        bin_means[b] = mu_b

        if n_b >= 2:
            # Unbiased variance calculation for structural verification
            sum_w2 = np.sum(w_b ** 2)
            denom = w_sum - (sum_w2 / w_sum) if weighted else (n_b - 1)
            var_b = np.sum(w_b * (m_b - mu_b) ** 2) / (denom if denom > 0 else 1)
            bin_stds[b] = np.sqrt(max(0.0, var_b))

    # --- Filter out underpopulated Bins ---
    good_bins = bin_counts >= min_points_per_bin
    n_good = good_bins.sum()
    if n_good < 2:
        raise ValueError(f"Only {n_good} bins meet the min_points_per_bin threshold.")
    if n_good < n_bins:
        warnings.warn(f"{n_bins - n_good} bins were excluded due to low population.", UserWarning, stacklevel=2)

    # Map selected usable observations
    good_obs_mask = good_bins[bin_indices]
    N_used = good_obs_mask.sum()

    # --- Exact Weighted Grand Mean (of the model-included subset) ---
    grand_mean = np.sum(bin_ws[good_bins] * bin_means[good_bins]) / bin_ws[good_bins].sum()

    # --- Sum of Squares calculations ---
    ss_between = np.sum(bin_ws[good_bins] * (bin_means[good_bins] - grand_mean) ** 2)

    ss_within = 0.0
    for b in np.where(good_bins)[0]:
        idx = (bin_indices == b)
        ss_within += np.sum(weights[idx] * (mag_clean[idx] - bin_means[b]) ** 2)

    # --- Degrees of Freedom & Mean Squares ---
    df_between = n_good - 1
    df_within = N_used - n_good

    if df_within <= 0:
        raise ValueError(f"Degrees of freedom error: df_within = {df_within}.")

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    if ms_within == 0.0:
        raise ValueError("Within-bin scatter is exactly zero. Check for duplicate records.")

    f_stat = ms_between / ms_within

    # --- Mathematical Transformation to Significance Profile ---
    # Log survival function avoids precision loss under high-amplitude signals
    log_p = f_dist.logsf(f_stat, df_between, df_within)
    p_value = np.exp(log_p)

    # Accurate one-tailed mapping from log-probability to Gaussian Sigmas
    sigmas_log = -ndtri(p_value) if p_value > 0.0 else -ndtri(np.exp(log_p))
    if log_p > -1e-12:  # Sub-noise checking
        sigmas_log = 0.0

    result = AoVResult(
        aov_statistic=f_stat,
        f_statistic=f_stat,
        p_value=p_value,
        n_sigma=sigmas_log,
        df_between=df_between,
        df_within=df_within,
        n_obs=N_used,
        n_bins=n_good,
        bin_phases=bin_phases,
        bin_means=bin_means,
        bin_stds=bin_stds,
        bin_counts=bin_counts,
        grand_mean=grand_mean,
        ss_between=ss_between,
        ss_within=ss_within,
        ms_between=ms_between,
        ms_within=ms_within,
        is_significant=log_p < np.log(significance_level),
        significance_level=significance_level,
        weighted=weighted,
        bin_edges=bin_edges,
    )

    if plot:
        plot_aov_folded_lightcurve(
            jd=jd, mag=mag, result=result, period=period,
            epoch=epoch, mag_err=mag_err, plot_twice=True
        )
    return result


def plot_aov_folded_lightcurve(
        jd: np.ndarray,
        mag: np.ndarray,
        result: AoVResult,
        period: float,
        epoch: float,
        mag_err: Optional[np.ndarray] = None,
        plot_twice: bool = True
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


# --------- Test -------------

def self_test():
    """
    Quick self-test
    """
    rng = np.random.default_rng(42)

    # Simulate a noisy lightcurve with a known sinusoidal signal
    N = 500
    period_true = 0.076  # days, typical superhump period
    epoch_true = 2_459_000.0
    jd_sim = epoch_true + rng.uniform(0, 20, N)  # 20 days baseline
    noise = 0.08  # mag, generous noise
    amplitude_true = 0.05  # mag semi-amplitude

    phase_sim = ((jd_sim - epoch_true) / period_true) % 1.0
    # mag_sim = 14.0 + amplitude_true * np.sin(2 * np.pi * phase_sim) + rng.normal(0, noise, N)
    mag_true = 14.0 + amplitude_true * np.sin(2 * np.pi * phase_sim)
    mag_sim = mag_true + rng.normal(0, noise, N)
    err_sim = np.full(N, noise)

    # plot_it(phase_sim, mag_true, mag_sim)

    print("\n--- Test 1: Signal present (unweighted) ---")
    result1 = aov_test(jd_sim, mag_sim, period_true, epoch_true, n_bins=10)
    print(result1)
    # plot_aov_folded_lightcurve(jd=jd_sim, mag=mag_sim, result=result1, period=period_true, epoch=epoch_true,
    #                            mag_err=None, plot_twice=True)

    print("\n--- Test 2: Signal present (weighted) ---")
    result2 = aov_test(jd_sim, mag_sim, period_true, epoch_true, n_bins=10, mag_err=err_sim)
    print(result2)

    print("\n--- Test 3: Pure noise (should NOT be significant) ---")
    mag_noise = 14.0 + rng.normal(0, noise, N)
    result3 = aov_test(jd_sim, mag_noise, period_true, epoch_true, n_bins=10, mag_err=err_sim)
    print(result3)

    print("\n--- Test 4: Wrong period (should NOT be significant) ---")
    wrong_period = period_true * 1.12
    result4 = aov_test(jd_sim, mag_sim, wrong_period, epoch_true, n_bins=10, mag_err=err_sim)
    print(result4)


# --------------- Real observations stuff --------

# def load_data(obs_file):
#     # numpy automatically ignores rows starting with '#' by default
#     jd_obs, mag_obs = np.loadtxt(obs_file, unpack=True)
#     return jd_obs, mag_obs, None


def load_data(obs_file: str):
    """
    Load astronomical data from a text file.

    Reads the 0th column as Julian Date, the 1st column (dmag) as the core magnitude,
    and the 3rd column as the magnitude error. Values of 99.99 in the error column
    are converted to np.nan so they can be handled gracefully by the downstream
    masking system.

    Parameters
    ----------
    obs_file : str
        Path to the space/tab-separated observation data file.

    Returns
    -------
    jd_obs : np.ndarray
        Array of observation times.
    mag_obs : np.ndarray
        Array of differential magnitudes (dmag).
    err_obs : np.ndarray
        Array of 1-sigma observational errors (with 99.99 mapped to NaN).
    """
    # Column indices: 0 -> JD, 1 -> dmag, 3 -> err
    # numpy automatically skips rows starting with '#'
    jd_obs, mag_obs, err_obs = np.loadtxt(
        obs_file,
        unpack=True,
        usecols=(0, 1, 3)
    )

    # Locate the dummy "no error" placeholders and replace them with NaN
    # Using np.isclose accounts for any minor floating-point storage variations
    no_error_mask = np.isclose(err_obs, 99.99)
    err_obs[no_error_mask] = np.nan

    return jd_obs, mag_obs, err_obs


def cutout_data(jd: np.ndarray, mag: np.ndarray, mag_err: np.ndarray, jd_min: float, jd_max: float):
    """Filters data within a JD range"""
    mask = (jd >= jd_min) & (jd <= jd_max)
    jd_piece = jd[mask]
    mag_piece = mag[mask]
    mag_err = mag_err[mask]
    return jd_piece, mag_piece, mag_err


def main(obs_filename: str, jd_min: float, jd_max: float, period: float, epoch: float):
    jd_full, mag_full, mag_err_full = load_data(obs_filename)
    jd, mag, mag_err = cutout_data(jd=jd_full, mag=mag_full, mag_err=mag_err_full, jd_min=jd_min, jd_max=jd_max)
    print("\n--- Observations: ---")
    result1 = aov_test(jd, mag, period, epoch, mag_err=mag_err, n_bins=10, plot=True)
    print(result1)
    result1 = aov_test(jd, mag, period, epoch, mag_err=None, n_bins=10, plot=True)
    print(result1)

    # print("\n\n--- Test with shuffled mags: ---")
    # shuffled_mag = np.random.permutation(mag)
    # result2 = aov_test(jd, shuffled_mag, period, epoch, n_bins=10, plot=True)
    # print(result2)
    #
    # print("\n\n--- Test with a wrong period ---")
    # wrong_period = period * 1.1
    # result3 = aov_test(jd, mag, period=wrong_period, epoch=epoch, n_bins=10, plot=True)
    # print(result3)


if __name__ == "__main__":
    # self_test()
    # import sys
    # sys.exit(0)

    # filename = 'data/TCP_J05415572-2308340/Summ.DAT'
    filename = 'data/TCP_J05415572-2308340/all_norm.dat'
    P_1 = 0.05285203  # Period in days
    T0_1 = 60971.5687  # Initial epoch in JD
    jd_min_1 = 0
    jd_max_1 = 60972.0
    # ----
    P_2 = 0.05496991
    T0_2 = T0_1
    jd_min_2 = 60972.3
    jd_max_2 = 60981
    main(obs_filename=filename, period=P_1, epoch=T0_1, jd_min=0, jd_max=60970.8)
    # main(obs_filename=filename, period=P_1, epoch=T0_1, jd_min=jd_min_1, jd_max=jd_max_1)
    # main(obs_filename=filename, period=P_2, epoch=T0_2, jd_min=jd_min_2, jd_max=jd_max_2)
