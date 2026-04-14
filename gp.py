from fileinput import filename

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from scipy.stats import median_abs_deviation

# =========================
# INPUT PARAMETERS (User defined)
# =========================

# cvs space-separated file with a lightcurve.
# Should contain 2 or 3 columns: jd, mag, mag_err (optional).
# If mag_err is missing, errors can be estimated automatically (see GUESS_SIGMA).
# FILENAME_IN = 'data/TCP_J05415572-2308340/orig/CO.DAT'
# FILENAME_IN = 'data/TCP_J05415572-2308340/orig/FJH.DAT'
FILENAME_IN = 'data/TCP_J05415572-2308340/orig/VSNET.DAT'

# File with jd intervals around a lightcurve maxima(each line: jd_left jd_mid jd_right).
# The pipeline processes each interval separately to find local maxima.
INTERVALS_FILE = 'data/TCP_J05415572-2308340/intervals.dat'

# If True: ignore provided errors and estimate them from data scatter (robust MAD).
# If False: use observer-provided mag_err (recommended when reliable).
GUESS_SIGMA = False

# Scaling factor applied to estimated noise when GUESS_SIGMA=True.
# Larger value → smaller assumed errors → more sensitive (more wiggly fit).
# Smaller value → larger errors → smoother fit.
# NOISE_SCALE_DIVISOR = 2.5
NOISE_SCALE_DIVISOR = 2

# =======================================
# GP scale length
# =======================================
# GP regression is quite sensitive to the scale length parameter.
# This defines how quickly the model is allowed to vary with time.
# It controls smoothness of the fit: smaller values -- more flexible (wiggly) model;
# larger values -- smoother model.
#
# Practical interpretation:
# - length_scale ~= characteristic width of a local feature in the lightcurve
#   (e.g. rise+fall of a maximum, eclipse width, asymmetric bump)
# - Smaller values --> GP follows small-scale variations (risk: overfitting noise)
# - Larger values --> GP produces smoother curve (risk: missing real structure)
#
# Reasonable first guess:
# - Estimate a typical width of a visible feature
# - Set length_scale_init ~ feature_width / 2
LENGTH_SCALE_INIT = 0.054 / 2

# =========================
# Lower bound control
# =========================
# We set lower bound of the length scale depending on the observation sampling (cadence),
# which represents the typical time resolution of observations:
# sampling_scale = median(dt)
# The user can tune this changing SAMPLING_SCALE_FACTOR
# Lower scale bound = sampling_scale * SAMPLING_SCALE_FACTOR,
#
# Lower bound control prevents GP from fitting structures smaller than data resolution.
# If too small -- model overfits noise.
# SAMPLING_SCALE_FACTOR = 3
# I've changed my mind. This parameter and 'sampling-based' bounding confuses even me.
# Let's shift to physical units, user directly see on the graph (abscissa-axis, i.g., JD)
LENGTH_SCALE_MIN = LENGTH_SCALE_INIT / 10.0

# ==========================
# Upper bound control
# ==========================
# It controls maximum allowed smoothness
# We set it depending on initial length_scale:
# Upper bound = length_scale * LENGTH_SCALE_FACTOR
# User can tune LENGTH_SCALE_FACTOR
# Again, increasing LENGTH_SCALE_FACTOR allows the model bahave smoothly
# LENGTH_SCALE_FACTOR = 2
# Again, Let's shift to physical units, visible directly by the user:
LENGTH_SCALE_MAX = LENGTH_SCALE_INIT * 3


# ========================
# WhiteKernel parameters
# ========================
# Initial value for additional (unknown) noise in the data.
# This represents extra scatter not captured by measurement errors.
# It is quite common for astronomical observations to underestimate uncertainties.
# We allow the model to correct this by adding a WhiteKernel component.
#
# When we trust the observer (i.e. believe that provided mag_err values are realistic),
# we should constrain the WhiteKernel noise by decreasing WHITE_NOISE_LEVEL_INIT and
# adjusting WHITE_NOISE_LEVEL_MIN accordingly.
#
WHITE_NOISE_LEVEL_INIT = 1e-3
#
# Bounds:
#
# Minimum allowed value for additional noise (WhiteKernel).
# Prevents the model from assuming unrealistically perfect data.
WHITE_NOISE_LEVEL_MIN = 1e-3
#
# Maximum allowed value for additional noise.
# Prevents the model from explaining all variability as noise.
WHITE_NOISE_LEVEL_MAX = 1
#
# The values used here are variances of the normalised flux, so:
# noise level = 1e-3 corresponds to expected uncertainties of ~0.03 mag
#
#
# Minimum number of points required in an interval to perform GP fitting.
# Intervals with fewer points are skipped.
LEN_MIN = 5


def plot_with_errors(x, y, y_err, title='guess errors'):
    plt.figure(figsize=(16, 10))  # for my fucking display!
    plt.errorbar(x, y, yerr=y_err, fmt='.',
                 markersize=8, ecolor='gray', elinewidth=1, capsize=2, label='data')
    plt.xlabel("JD")
    plt.ylabel("Flux normalized")
    plt.title(title)
    plt.legend(fontsize=13)
    plt.show()

    plt.figure(figsize=(16, 10))


def plot_GP_results(x, y_norm, noise_sigma_norm,
                    jd_peak, mean_peak, peaks_jd, jd_max_guess, jd_peak_std,
                    jd_grid, mean_grid, std_grid, n_samples_uncert):
    plt.figure(figsize=(16, 10))  # for my fucking display
    y_plot = y_norm
    yerr_plot = np.full_like(y_plot, noise_sigma_norm)

    plt.errorbar(x, y_plot, yerr=yerr_plot, fmt='o', markersize=6,
                 ecolor='gray', elinewidth=1, capsize=2, label='data (with estimated errors)')

    plt.scatter(x, y_plot, s=30, color='k')

    plt.plot(jd_grid.ravel(), mean_grid.ravel(), color='tab:blue', lw=2, label='GP mean')
    plt.fill_between(jd_grid.ravel(),
                     mean_grid.ravel() - std_grid.ravel(),
                     mean_grid.ravel() + std_grid.ravel(),
                     color='tab:blue', alpha=0.25, label='GP ±1σ')

    # mark posterior-sampled maxima (light points)
    plt.scatter(peaks_jd, np.full_like(peaks_jd, 0.98 * mean_peak),
                s=30, color='orange', alpha=0.1)
    plt.scatter(peaks_jd, np.full_like(peaks_jd, 0.98 * mean_peak),
                s=30, color='orange', alpha=0.1,
                label=f'Posterior peak draws (n={n_samples_uncert})')

    # verticals: guess and estimated peak
    if jd_max_guess is not None:
        plt.axvline(jd_max_guess, color='green', linestyle=':', lw=1.5, label='Peak guess')
    plt.axvline(float(jd_peak - jd_peak_std), color='magenta', linestyle=':', lw=2)
    plt.axvline(float(jd_peak), color='magenta', linestyle='--', lw=2, label=f'GP peak: {jd_peak:.8f}')
    plt.axvline(float(jd_peak + jd_peak_std), color='magenta', linestyle=':', lw=2)
    plt.fill_betweenx(
        [plt.ylim()[0], plt.ylim()[1]],
        float(jd_peak - jd_peak_std),
        float(jd_peak + jd_peak_std),
        color='magenta', alpha=0.1, label='±1σ range'
    )

    plt.xlabel('JD')
    plt.ylabel('Normalised flux')
    plt.title('GP fit and peak estimate')
    plt.legend(fontsize=14)
    plt.show()


# =========================
# BASIC UTILITIES
# =========================

def select_jd_interval(df, jd_min, jd_max):
    """Select data inside JD interval."""
    return df[(df["jd"] >= jd_min) & (df["jd"] <= jd_max)].copy()


def read_lc(full_filename):
    """
    Read file with:
    jd mag [mag_err]

    If mag_err is missing → filled with NaN
    """
    print('read_lc')
    print(f'Reading {full_filename}')
    df = pd.read_csv(full_filename, sep=r'\s+', comment='#', header=None)

    if df.shape[1] == 2:
        df.columns = ["jd", "mag"]
        df["mag_err"] = np.nan
    else:
        df.columns = ["jd", "mag", "mag_err"]
    print(df.shape)

    return df


def load_intervals(file_obj):
    print(f'Loading intervals from file_obj...')
    result = []
    for line in file_obj:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        a, b, c = map(float, line.split())
        result.append((a, b, c))
    return result


def load_intervals_from_file(filename_intervals):
    print(f'Loading intervals from file...{filename_intervals}')
    result = []
    with open(filename_intervals) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            a, b, c = map(float, line.split())
            result.append((a, b, c))
    return result


def add_flux(df):
    mag0 = 20
    df['flux'] = 10 ** (-0.4 * (df['mag'] - mag0))

    # handle missing mag_err
    if not df['mag_err'].isna().all():
        df['flux_err'] = 0.921034 * df['flux'] * df['mag_err']
    else:
        df['flux_err'] = np.nan
    return df


def residual_noise_estimate(x, y, baseline, ampl_guess):
    """
    Estimate photometric noise from residuals after subtracting a simple peak model.

    My brave assumption:
    I pretend that the light-curve segment is dominated by a single smooth feature
    and approximate it with a very simple symmetric triangular shape.

    What I actually do:
    1. I draw two straight lines: up to the peak and down from the peak.
    2. I subtract this “masterpiece” from the data.
    3. I declare whatever is left (residuals) to be observational noise.

    I want to remove large-scale variability first, so that the remaining scatter
    mostly reflects measurement errors rather than real signal.

    The real light curve is more complex (at lest asymmetric).
    Anyway I provide user with tuning factor, so, they can fix things
    """

    # robust estimate of total scatter in flux
    mad_raw = median_abs_deviation(y, scale='normal')

    # --- define simple symmetric triangular model of maximum ---
    y_vals = y
    x_left = x.min()
    x_right = x.max()
    x_center = 0.5 * (x_left + x_right)

    y_base = baseline
    y_peak = baseline + ampl_guess

    y_model = np.zeros_like(y_vals)

    # left branch (rising)
    left_mask = x <= x_center
    y_model[left_mask] = y_base + (y_peak - y_base) * (
            (x[left_mask] - x_left) / (x_center - x_left)
    )

    # right branch (falling)
    right_mask = x > x_center
    y_model[right_mask] = y_peak - (y_peak - y_base) * (
            (x[right_mask] - x_center) / (x_right - x_center)
    )

    # --- residuals ---
    residuals = y_vals - y_model

    # robust noise estimate from residuals
    mad = median_abs_deviation(residuals, scale='normal')

    print(f'{mad=:.3f} {np.std(residuals) * 0.5=:.3f} {mad_raw=:.3f}')
    mad = mad if mad > 0 else np.std(residuals) * 0.5
    noise_sigma = min(mad_raw, mad)  # type: ignore

    return noise_sigma


# MAIN GP PIPELINE


def gp_peak_pipeline(
        df: pd.DataFrame,
        jd_left: float,
        jd_right: float,
        params,
        n_grid=2000,
        n_samples_uncert=300,
        random_state=0,
        plot_final=False,
) -> dict:
    """
    Fit GP to a fragment and estimate peak position (JD) with uncertainty.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'jd' and 'flux'. May contain 'flux_err'
    jd_left, jd_right : float
        Interval to consider (inclusive).
    params : dict
        GP regression parameters:
        - 'guess_sigma' If guess_sigma=True OR no valid errors → use MAD
        - 'noise_scale_divisor' Empirical factor, allow user tune sigma estimated by algorthm
        - 'length_scale_init' Initial guess about GP lenght scale
        - 'length_scale_min', 'length_scale_max'    Bounds
        - 'white_noise_level_init' Initial guess about White Kernel noise level
        - 'white_noise_level_min', 'white_noise_level_max' Bounds

    n_grid : int
        Number of points in the fine evaluation grid (for mean/derivative).
    n_samples_uncert : int
        Number of posterior samples used to estimate JD uncertainty.
    random_state : int
        Seed for reproducible posterior sampling.
    plot_final  : plot results as matplotlib graph (debug)

    Returns
    -------
    result : dict
        {
            'jd_peak': float,               # estimated peak JD (from GP mean)
            'jd_peak_std': float,           # uncertainty (std) from posterior samples
            'gp': GaussianProcessRegressor, # fitted GP object
            'n_samples_uncert'              # number of samples to estimate moment uncertainty
            'mean_peak'                     # mean peak
            'peaks_jd'                      # raw peak JDs from posterior samples
            'jd_grid'                       # evaluation grid
            'mean_grid'                     # GP mean on grid
            'std_grid'
            'noise_sigma_norm'              # normalised sigma of the input data
        }
    """

    # --- 1. select fragment ---
    frag = select_jd_interval(df, jd_left, jd_right)
    if frag.empty:
        raise ValueError("No data in the provided JD interval.")

    x = frag['jd'].values.copy()
    y = frag['flux'].values.copy()
    z = frag['flux_err'].values.copy()

    # --- 2. baseline and amplitude ---
    baseline = float(np.percentile(y, 5))
    ampl_guess = np.percentile(y, 95) - baseline
    print(f'{baseline=:.3f} {ampl_guess=:.3f}')

    if ampl_guess <= 0:
        ampl_guess = np.std(y) if np.std(y) > 0 else 1.0

    # --- 4. normalisation ---
    y_norm = (y - baseline) / ampl_guess

    # --- 3. estimate noise ---
    # If guess_sigma=True OR no valid errors → use MAD
    if params['guess_sigma'] or np.all(np.isnan(z)):
        noise_sigma = residual_noise_estimate(x, y, baseline, ampl_guess)
        noise_sigma /= params['noise_scale_divisor']  # empirical factor, allow user tune it
        print(f'guessed {noise_sigma=:.3f}')
        # propagate noise into normalized units
        noise_sigma_norm = noise_sigma / ampl_guess
    else:
        print(f'noise_sigma mean {np.mean(z):.3f}')
        noise_sigma_norm = z / ampl_guess

    # --- 5. kernel ---
    length_scale = params['length_scale_init']
    print(f'length_scale_guess={length_scale:.3f}')

    y_norm_var = np.var(y_norm)
    print(f'{y_norm_var=:.3f}')

    # sampling scale (from data spacing)
    dt = np.diff(np.sort(df['jd']))
    sampling_scale = np.median(dt)

    kernel = (
            # ConstantKernel = amplitude (vertical scale) of the GP signal
            # constant_value=1.0 because we work with normalised fluxes
            ConstantKernel(
                constant_value=1.0,
                constant_value_bounds=(y_norm_var * 0.01, y_norm_var * 100.0)
            ) *
            Matern(length_scale=length_scale,
                   # length_scale_bounds=(
                   #     sampling_scale * params['sampling_scale_factor'],
                   #     length_scale * params['length_scale_factor']
                   # ),
                   length_scale_bounds=(
                       params['length_scale_min'], params['length_scale_max']
                   ),
                   nu=2.5) +
            WhiteKernel(
                noise_level=params['white_noise_level_init'],
                noise_level_bounds=(params['white_noise_level_min'], params['white_noise_level_max']
                )
            )
    )

    print('Start Gaussian Process')

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise_sigma_norm ** 2,
        normalize_y=False,
    )

    print('...')

    # --- 6. fit ---
    gp.fit(x.reshape(-1, 1), y_norm)  # sklearn expects a table of features, even if there is only one column (time).

    print(gp.kernel_)
    k = gp.kernel_
    length_scale_final = k.k1.k2.length_scale
    noise_level_final = k.k2.noise_level
    amplitude_final = k.k1.k1.constant_value

    print(f'{length_scale_final=:.3f} {noise_level_final=:.3f} {amplitude_final=:.3}')

    print('Fit is ready')

    # --- 7. predict ---
    # ------- 7.1 grid ---
    # padding around an interval:
    pad = max((jd_right - jd_left) * 0.02, length_scale_final * 0.2)
    grid_min = max(jd_left - pad, x.min())
    grid_max = min(jd_right + pad, x.max())
    jd_grid = np.linspace(grid_min, grid_max, n_grid).reshape(-1, 1)

    # ------- 7.2 predict ---
    mean_grid, std_grid = gp.predict(jd_grid, return_std=True)

    # --- 8. peak ---
    idx_peak = np.argmax(mean_grid.ravel())
    jd_peak = jd_grid.ravel()[idx_peak]
    mean_peak = mean_grid.ravel()[idx_peak]

    # --- 9. uncertainty on peak via posterior sampling ---
    # draw samples of functions on the grid from the posterior and find maxima positions
    # sample_y returns shape (n_points, n_samples)
    samples = gp.sample_y(jd_grid, n_samples=n_samples_uncert, random_state=random_state)
    peaks = np.argmax(samples, axis=0)
    peaks_jd = jd_grid.ravel()[peaks]
    jd_peak_std = float(np.std(peaks_jd))

    # --- 10. plotting ---
    if plot_final:
        plot_GP_results(x, y_norm, noise_sigma_norm,
                        jd_peak, mean_peak, peaks_jd, None, jd_peak_std,
                        jd_grid, mean_grid, std_grid, n_samples_uncert)

    return {
        # "x": x,
        # "y_norm": y_norm,
        "noise_sigma_norm": noise_sigma_norm,
        "jd_grid": jd_grid,
        "mean_grid": mean_grid,
        "std_grid": std_grid,
        "peaks_jd": peaks_jd,
        "jd_peak": jd_peak,
        "jd_peak_std": jd_peak_std,
        "mean_peak": mean_peak,
        "n_samples_uncert": n_samples_uncert,
        "gp": gp,
    }


# =========================
# MAIN
# =========================

def main():
    df0 = read_lc(FILENAME_IN)
    pieces_list = load_intervals_from_file(INTERVALS_FILE)

    df0 = add_flux(df0)

    # handle missing mag_err
    if not df0['mag_err'].isna().all():
        df0['flux_err'] = 0.921034 * df0['flux'] * df0['mag_err']
    else:
        df0['flux_err'] = np.nan

    with open('maxima_gp.dat', 'a') as f:
        for piece in pieces_list:
            jd_min, jd_max = piece[0], piece[-1]

            print(f'Start with {jd_min} : {jd_max} piece')

            if len(select_jd_interval(df0, jd_min, jd_max)) < LEN_MIN:
                continue

            gp_res = gp_peak_pipeline(
                df0,
                jd_min,
                jd_max,
                params={
                    "noise_scale_divisor": NOISE_SCALE_DIVISOR,
                    "length_scale_init": LENGTH_SCALE_INIT,
                    # "sampling_scale_factor": SAMPLING_SCALE_FACTOR,
                    # "length_scale_factor": LENGTH_SCALE_FACTOR,
                    "length_scale_min": LENGTH_SCALE_MIN,
                    "length_scale_max": LENGTH_SCALE_MAX,
                    "white_noise_level_init": WHITE_NOISE_LEVEL_INIT,
                    "white_noise_level_min": WHITE_NOISE_LEVEL_MIN,
                    "white_noise_level_max": WHITE_NOISE_LEVEL_MAX,
                    "guess_sigma": GUESS_SIGMA
                },
                plot_final=True,
            )

            f.write(f'GP peak = {gp_res["jd_peak"]:.6f}  std = {gp_res["jd_peak_std"]:.6f}\n')


if __name__ == "__main__":
    main()
