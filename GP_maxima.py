import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from scipy.stats import median_abs_deviation

plt.rcParams.update({'font.size': 24})  # Set global font size

# SCALE controls length scale. Increase to get smooths the curve
SCALE = 1
# SCALE = 2
FILTER = 'R'
# FILTER = 'B'
# FILTER = 'V'
# FILTER = 'I'


def plot_with_errors(x, y, y_err, title='guess errors'):
    plt.figure(figsize=(16, 10))
    plt.errorbar(x, y, yerr=y_err, fmt='.',
                 markersize=8, ecolor='gray', elinewidth=1, capsize=2, label='data')
    plt.xlabel("JD")
    plt.ylabel("Flux normalized")
    plt.title(title)
    plt.legend(fontsize=13)
    plt.show()


def select_jd_interval(df, jd_min, jd_max):
    """Return a new DataFrame containing only rows with jd within [jd_min, jd_max]."""
    return df[(df["jd"] >= jd_min) & (df["jd"] <= jd_max)].copy()


def read_from_file(full_filename):
    df = pd.read_csv(full_filename, delim_whitespace=True, comment="#", names=["jd", "mag"])
    # print(df.head())
    return df


def gp_peak_pipeline(
        df,
        jd_left,
        jd_right,
        fwhm_guess=0.02,
        jd_max_guess=None,
        normalize=True,
        n_grid=2000,
        n_samples_uncert=300,
        plot=True,
        random_state=0
):
    """
    Fit GP to a fragment and estimate the JD of the maximum (with uncertainty).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'jd' and 'flux'.
    jd_left, jd_right : float
        Interval to consider (inclusive).
    fwhm_guess : float
        Approximate FWHM in JD (used to set length scale). Default 0.02. Controls sharpness of the peak
    jd_max_guess : float or None
        Optional initial guess for the peak JD; used for plotting and selecting grid window.
    normalize : bool
        If True, subtract local baseline (min) and divide by amplitude (max-min).
    n_grid : int
        Number of points in the fine evaluation grid (for mean/derivative).
    n_samples_uncert : int
        Number of posterior samples used to estimate JD uncertainty.
    plot : bool
        Whether to create the two-step plot.
    random_state : int
        Seed for reproducible posterior sampling.

    Returns
    -------
    result : dict
        {
            'jd_peak': float,              # estimated peak JD (from GP mean)
            'jd_peak_std': float,          # uncertainty (std) from posterior samples
            'gp': GaussianProcessRegressor,# fitted GP object
            'jd_grid': ndarray,            # evaluation grid
            'mean_grid': ndarray,          # GP mean on grid
            'std_grid': ndarray,           # GP std on grid
            'samples_peak_jds': ndarray    # raw peak JDs from posterior samples
        }
    """

    # --- 1. select fragment ---
    frag = df[(df['jd'] >= jd_left) & (df['jd'] <= jd_right)].copy()
    if frag.empty:
        raise ValueError("No data in the provided JD interval.")

    x = frag['jd'].values.reshape(-1, 1)
    y = frag['flux'].values.copy()

    # --- 2. estimate robust noise (MAD -> sigma) and baseline/amplitude ---
    # robust sigma estimate (on raw flux)
    mad = median_abs_deviation(y, scale='normal')  # equivalent to 1.4826*MAD
    noise_sigma = mad if mad > 0 else np.std(y) * 0.5  # fallback if MAD=0
    if FILTER == 'R':
        noise_sigma /= 5  # I've said!
    elif FILTER == 'B':
        noise_sigma /= 3    # I've said
    elif FILTER == 'V':
        noise_sigma /= 3    # I've said
    elif FILTER == 'I':
        noise_sigma /= 3    # I've said
    else:
        raise NotImplementedError(f'Unimplemented filter {FILTER}')

    baseline = np.min(y)
    ampl_guess = np.max(y) - baseline
    if ampl_guess <= 0:
        ampl_guess = np.std(y) if np.std(y) > 0 else 1.0

    # convert Poisson-like idea into per-point sigma: sqrt(flux) scaling is only a guess
    # we'll use robust MAD-based variance scaled to amplitude after normalization (see below)
    # For the GP white kernel we will supply variance in normalized units (see normalization step).

    # --- 3. normalization (optional) ---
    if normalize:
        y_norm = (y - baseline) / ampl_guess
        # propagate noise estimate to normalized units
        noise_sigma_norm = noise_sigma / ampl_guess
    else:
        y_norm = y.copy()
        noise_sigma_norm = noise_sigma

    plot_with_errors(frag['jd'], y_norm, noise_sigma_norm)
    # --- 4. build fine grid for evaluation ---
    # expand grid a bit around the fragment to allow small extrapolation
    pad = max((jd_right - jd_left) * 0.02, 1.5 * (fwhm_guess / 10.0))
    grid_min = max(jd_left - pad, frag['jd'].min())
    grid_max = min(jd_right + pad, frag['jd'].max())
    jd_grid = np.linspace(grid_min, grid_max, n_grid).reshape(-1, 1)

    # --- 5. kernel & hyperparameters (fixed) ---
    # convert FWHM to sigma and set length scale L = sigma
    sigma_guess = fwhm_guess / 2.3548200450309493  # conversion constant
    length_scale = max(SCALE * sigma_guess, 1e-6)
    # length_scale = max(sigma_guess, 1e-6)
    # length_scale = max(sigma_guess * 2, 1e-6)
    # length_scale = max(sigma_guess * 3, 1e-6)

    # use Constant * Matern(5/2) + WhiteKernel with fixed hyperparams (no optimizer)
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * \
             Matern(length_scale=length_scale, nu=2.5) + \
             WhiteKernel(noise_level=(noise_sigma_norm ** 2), noise_level_bounds="fixed")

    print('Start Gaussian Process')
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False, random_state=random_state)
    print('...')

    # --- 6. fit GP ---
    gp.fit(x, y_norm)
    print('Fit is ready')

    # --- 7. predict on grid (mean and std) ---
    mean_grid, std_grid = gp.predict(jd_grid, return_std=True)

    # --- 8. find peak on GP mean ---
    idx_peak = np.argmax(mean_grid.ravel())
    jd_peak = jd_grid.ravel()[idx_peak]
    mean_peak = mean_grid.ravel()[idx_peak]

    # --- 9. uncertainty on peak via posterior sampling ---
    # draw samples of functions on the grid from the posterior and find maxima positions
    # sample_y returns shape (n_points, n_samples)
    samples = gp.sample_y(jd_grid, n_samples=n_samples_uncert, random_state=random_state)

    peaks = np.argmax(samples, axis=0)  # indices of maxima in each sample
    peaks_jd = jd_grid.ravel()[peaks]
    jd_peak_std = float(np.std(peaks_jd))  # use std as uncertainty estimate

    # Optionally refine uncertainty by converting to t distribution or quantiles:
    # jd_peak_ci = np.percentile(peaks_jd, [16,84])  # 68% CI

    # --- 10. plotting ---
    if plot:
        plt.figure(figsize=(16, 10))  # twice larger for your display

        # fragment with (estimated) errorbars
        # scale back yerr for plotting (in original flux units if normalized)
        if normalize:
            y_plot = y_norm
            yerr_plot = np.full_like(y_plot, noise_sigma_norm)
            ylabel = 'Normalized flux'
        else:
            y_plot = y
            yerr_plot = np.full_like(y_plot, noise_sigma)
            ylabel = 'Flux'

        plt.errorbar(frag['jd'], y_plot, yerr=yerr_plot, fmt='o', markersize=6,
                     ecolor='gray', elinewidth=1, capsize=2, label='data (with estimated errors)')

        # scatter as well (size=30) for better visibility
        plt.scatter(frag['jd'], y_plot, s=30, color='k')

        # GP mean and uncertainty band
        plt.plot(jd_grid.ravel(), mean_grid.ravel(), color='tab:blue', lw=2, label='GP mean')
        plt.fill_between(jd_grid.ravel(),
                         mean_grid.ravel() - 1.0 * std_grid.ravel(),
                         mean_grid.ravel() + 1.0 * std_grid.ravel(),
                         color='tab:blue', alpha=0.25, label='GP ±1σ')

        # mark posterior-sampled maxima (light points)
        plt.scatter(peaks_jd, np.full_like(peaks_jd, 0.98 * mean_peak), s=30, color='orange', alpha=0.1,
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
        plt.ylabel(ylabel)
        plt.title('GP fit and peak estimate')
        plt.legend(fontsize=14)
        plt.show()

    # --- 11. return results ---
    result = {
        'jd_peak': float(jd_peak),
        'jd_peak_std': float(jd_peak_std),
        'gp': gp,
        'jd_grid': jd_grid.ravel(),
        'mean_grid': mean_grid.ravel(),
        'std_grid': std_grid.ravel(),
        'samples_peak_jds': peaks_jd
    }
    return result


def main():
    # '/home/voz/projects/UPJS/VO/skvo_veb_project/skvo_veb/auxiliary/data/Shugarov/V_joined_shifted.dat'
    if FILTER == 'R':
        filename_in = '/home/voz/projects/Shugarov/TCP_J10240289+4808512/data/R_joined_shifted.dat'
    elif FILTER == 'B':
        filename_in = '/home/voz/projects/Shugarov/TCP_J10240289+4808512/data/B_sorted.dat'
    elif FILTER == 'V':
        filename_in = '/home/voz/projects/Shugarov/TCP_J10240289+4808512/data/V_sorted.dat'
    elif FILTER == 'I':
        filename_in = '/home/voz/projects/Shugarov/TCP_J10240289+4808512/data/I_sorted.dat'
    else:
        raise NotImplementedError('What filter do you mean???')

    df0 = read_from_file(filename_in)

    mag0 = 20
    df0['flux'] = 10 ** (-0.4 * (df0['mag'] - mag0))

    # I:
    pieces_631_I = [(58631.270, 58631.330, 58631.300)]
    pieces_633_I = [(58633.300, 58633.365, 58633.330)]
    pieces_636_I = [(58636.277, 58636.340, 58636.29915)]
    pieces_640_I = [(58640.290, 58640.357, 58640.324),
                    (58640.360, 58640.410, 58640.387)]

    I_list = [pieces_631_I, pieces_633_I, pieces_636_I, pieces_640_I]

    # V:
    pieces_631_V = [(58631.270, 58631.330, 58631.300)]
    pieces_633_V = [(58633.300, 58633.365, 58633.330)]
    pieces_636_V = [(58636.277, 58636.340, 58636.29915)]
    pieces_639_V = [(58639.330, 58639.375, 58639.351)]
    pieces_640_V = [(58640.290, 58640.350, 58640.324),
                    (58640.350, 58640.402, 58640.387)]

    V_list = [pieces_631_V, pieces_633_V, pieces_636_V, pieces_639_V, pieces_640_V]

    # B:
    pieces_631_B = [(58631.270, 58631.330, 58631.300)]
    pieces_636_B = [(58636.277, 58636.340, 58636.29915)]
    pieces_640_B = [(58640.298, 58640.350, 58640.324),
                    (58640.366, 58640.400, 58640.387)]

    B_list = [pieces_631_B, pieces_636_B, pieces_640_B]

    pieces_630_ = [(58630.3271, 58630.4199, 58630.3938),
                   (58630.4199, 58630.46577, 58630.45188)]
    pieces_631_ = [(58631.270, 58631.3305, 58631.300)]
    pieces_632 = [(58632.3957, 58632.4548, 58632.4224),
                  (58632.4548, 58632.5331, 58632.4900)]

    pieces_633 = [
        (58633.3090, 58633.3735, 58633.3302),
        (58633.3735, 58633.437, 58633.3962),
        (58633.437, 58633.5003, 58633.4597)
    ]
    pieces_636 = [(58636.27783, 58636.33383, 58636.29915)]
    pieces_637 = [(58637.31800, 58637.35840, 58637.33629)]
    pieces_640 = [(58640.30404, 58640.36564, 58640.32031),
                  (58640.36564, 58640.40089, 58640.38356)]
    pieces_641_ = [(58641.2783, 58641.35742, 58641.33031)]  # This is a minimum, just mirror it
    pieces_643_ = [(58643.3363, 58643.367, 58643.35074)]
    pieces_644_ = [(58644.29684, 58644.34566, 58644.31555)]

    R_list = [pieces_630_, pieces_631_, pieces_632, pieces_633, pieces_636, pieces_637,
              pieces_640, pieces_641_, pieces_643_, pieces_644_]

    if FILTER == 'R':
        pieces_list = R_list
    elif FILTER == 'B':
        pieces_list = B_list
    elif FILTER == 'V':
        pieces_list = V_list
    elif FILTER == 'I':
        pieces_list = I_list
    else:
        raise NotImplementedError(f'Unrecognized filter {FILTER}')

    with open('maxima_gp.dat', 'a') as f:
        for pieces in pieces_list:
            pieces_bounds = (pieces[0][0], pieces[-1][1])
            piece_left, piece_right = pieces_bounds

            df = select_jd_interval(df0, piece_left, piece_right)

            if FILTER == 'R':
                if 58630.3271 < pieces[0][1] < 58630.45188:
                    df['flux'] = df['flux'] - (-2.2861e+07 + 389.93 * df['jd']) + 780
                elif 58632.3957 < pieces[0][1] < 58632.4900:
                    df['flux'] = df['flux'] - (-1.6684e+07 + 284.57 * df['jd']) + 1050
                elif 58641.278 < pieces[0][1] < 58641.358:     # mirror minimum to turn it into maximum
                    df['flux'] = 650.0 - df['flux']

            for jd_min, jd_max, jd_split in pieces:
                print(f'Start with {jd_min} : {jd_max} piece')
                gp_res = gp_peak_pipeline(df, jd_min, jd_max, fwhm_guess=0.04)
                f.write(f'GP peak = {gp_res["jd_peak"]:.6f}  std = {gp_res["jd_peak_std"]:.6f}\n')


if __name__ == "__main__":
    main()
