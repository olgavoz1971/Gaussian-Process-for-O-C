import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 24})  # Set global font size


class EclipsingBinaryModel:
    def __init__(self, np_terms=1, nc=1):
        self.np_terms = np_terms
        self.nc = nc
        self.popt = None
        self.stats = {}

    # MATH COMPONENTS

    def eclipses(self, theta, B, A1, A2, D, Gamma):
        """Two eclipse dips """
        total = np.full_like(theta, B, dtype=float)
        # Ensure D is never zero to avoid ZeroDivisionError
        safe_D = np.maximum(D, 1e-8)

        for k, Ak in enumerate((A1, A2), start=1):
            # Improved wrapping logic
            phi_k = (theta - 0.5 * (k - 1) + 0.5) % 1.0 - 0.5

            x = np.clip(phi_k / safe_D, -50, 50)
            inner = 1 - np.exp(1 - np.cosh(x))

            # Ensure 'inner' is non-negative before applying Gamma power
            total += Ak * (1 - np.power(np.maximum(inner, 0), Gamma))
        return total

    def harmonics(self, theta, A_cos, A_sin=0.0):
        """Cosine and optional sine terms
        Explicitly handles A_sin as a default 0.0 if not provided.
        """
        theta = np.asarray(theta)
        fp = np.zeros_like(theta)

        # Use enumerate to be safer with indexing
        for j, coeff in enumerate(A_cos):
            fp += coeff * np.cos(2 * np.pi * (j + 1) * theta)

        # Only calculate sine if nc is active AND we have a non-zero coefficient
        fc = A_sin * np.sin(2 * np.pi * theta) if self.nc == 1 else 0.0
        return fp + fc

    def full_model(self, theta, params):
        """ Complete model with an improved slicing safety"""
        phase_shift = params[0]
        B, A1, A2, D, Gamma = params[1:6]

        # Slice exactly what we need based on np_terms
        A_cos = params[6: 6 + self.np_terms]

        # Extract A_sin only if nc is 1, otherwise default to 0.0
        A_sin = params[6 + self.np_terms] if (self.nc == 1 and len(params) > 6 + self.np_terms) else 0.0

        theta_shifted = np.mod(theta - phase_shift, 1.0)

        base = self.eclipses(theta_shifted, B, A1, A2, D, Gamma)
        harm = self.harmonics(theta_shifted, A_cos, A_sin)
        return base + harm

    from typing import Literal
    LossType = Literal[
        "linear",
        "soft_l1",
        "huber",
        "cauchy",
        "arctan"
    ]

    def fit(self, phase, flux, flux_err,
            loss: LossType = 'soft_l1', f_scale=1.0,
            custom_bounds=None):
        """
        Performs a robust least-squares optimization of the phenomenological model.

        Parameters
        ----------
        phase : array-like
            Orbital phase values, typically in the range [0, 1].
        flux : array-like
            Observed flux (normalized).
        flux_err : array-like
            Uncertainties in the observed flux.
        loss : LossType, optional
            The loss function used for the objective.
            - 'linear': Standard least-squares (sensitive to outliers).
            - 'soft_l1': Robust loss; ignores outliers beyond f_scale.
            - 'huber': Hybrid; linear for small residuals, quadratic for large.
            Default is 'soft_l1'.
        f_scale : float, optional
            The "outlier threshold". Smaller values make the fit more resistant
            to bad data points (flares, cosmic rays) by reducing their weight.
            Default is 1.0.
        custom_bounds : tuple of lists, optional
            Manual (lower, upper) bounds. If None, uses defaults optimized for
            eclipsing binaries.

        Notes on Default Bounds
        -----------------------
        The model utilizes the following bounding logic:
        1. phase_shift [-0.5, 0.5]: Allows the primary eclipse to center at 0.0.
        2. B (Baseline) [0.5, 1.5]: Constrains the out-of-eclipse flux near 1.0.
        3. A1, A2 (Depths) [-2, 0.1]: Central eclipse depths. Upper bound of 0.1
           allows for minor positive deviations (e.g., reflection effects).
        4. D (Width) [1e-3, 0.5]: Half-width of eclipses. Cannot be zero (math error)
           or exceed 0.5 (entire orbital period).
        5. Gamma [0.1, 20.0]: Kurtosis of eclipse wings. Higher values create
           sharper ingress/egress. Limited to 20.0 to maintain numerical stability.
        6. Harmonics [-1.0, 1.0]: Constrains out-of-eclipse modulations to
           physical amplitudes.
        """

        # Initial guess logic based on flux percentiles
        B_guess = np.percentile(flux, 95)
        low5 = np.percentile(flux, 5)
        depth_est = B_guess - low5

        p0 = (
                [0.0, B_guess, -0.6 * depth_est, -0.3 * depth_est, 0.08, 2.0]
                + [0.01] * self.np_terms
                + ([0.0] if self.nc == 1 else [])
        )

        if custom_bounds is None:
            # Default bounds:
            #       [shift, B,   A1,  A2,    D,   Gamma,  A_cos...,                 A_sin]
            lower = [-0.5, 0.5, -2.0, -2.0, 1e-3, 0.1] + [-1.0] * self.np_terms + ([-1.0] if self.nc == 1 else [])
            upper = [0.5, 1.5, 0.0, 0.0, 0.5, 20.0] + [1.0] * self.np_terms + ([1.0] if self.nc == 1 else [])
        else:
            lower, upper = custom_bounds

        def res_fun(params, x, y, yerr):
            # Evaluate model and handle potential numerical NaNs
            model_vals = self.full_model(x, params)
            residuals_ = (model_vals - y) / yerr
            return np.nan_to_num(residuals_, nan=1e6)  # High penalty for invalid math regions

        # Execute optimization
        res = least_squares(
            res_fun,
            x0=p0,
            args=(phase, flux, flux_err),
            bounds=(lower, upper),
            loss=loss,
            f_scale=f_scale,
            max_nfev=200000,
        )

        self.popt = res.x

        # Statistical analysis
        residuals = flux - self.full_model(phase, self.popt)
        chi2 = np.sum((residuals / flux_err) ** 2)
        ndof = len(flux) - len(self.popt)

        self.stats = {
            "chi2": chi2,
            "chi2_red": chi2 / ndof,
            "ndof": ndof,
            "popt": self.popt,
            "success": res.success,
            "message": res.message
        }

        return self.popt, self.stats


def normalize_flux(flux, flux_err):
    f_min = np.min(flux)
    f_max = np.max(flux)
    scale = f_max - f_min if f_max > f_min else 1.0
    return (flux - f_min) / scale, flux_err / scale, f_min, f_max


def main(filename):
    path = 'data/' + filename

    try:
        tab = ascii.read(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return

    # Prep Data
    f_norm, ferr_norm, f_min, f_max = normalize_flux(np.array(tab["flux"]), np.array(tab["flux_err"]))
    phase = np.array(tab["phase"])

    # Initialize and Fit
    model = EclipsingBinaryModel(np_terms=1, nc=1)
    popt, stats = model.fit(phase, f_norm, ferr_norm)

    # Output Results
    param_names = ["phase_shift", "B", "A1", "A2", "D", "Gamma"] + \
                  [f"A_cos{i + 1}" for i in range(model.np_terms)] + \
                  (["A_sin"] if model.nc == 1 else [])

    print("\nBest-fit parameters:")
    for name, val in zip(param_names, popt):
        print(f" {name:>10s} = {val:.6f}")
    print(f"\nχ² = {stats['chi2']:.3f}, reduced χ² = {stats['chi2_red']:.3f}")

    # Plotting
    new_phase = np.linspace(0, 1, 400)
    new_flux = model.full_model(new_phase, popt)

    plt.figure(figsize=(16, 10))
    plt.errorbar(phase, f_norm, yerr=ferr_norm, fmt=".", markersize=10, alpha=0.6,
                 label="Observed", color='gray',
                 zorder=2)  # we use zorder to control plot order. This time we want to put dots underneath
    plt.plot(new_phase, new_flux, "r-", lw=2, label="Model fit",
             zorder=3)
    plt.xlabel("Phase")
    plt.ylabel("Normalized Flux")
    plt.title('Phenomenological Modeling')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for filename_ in [
        'lc_gaia_4585381817643702528_G.ecsv',
        'AB_AND_56.ecsv',
        'lc_tess_FFI__AB_And_sparce.ecvs',
        'lc_gaia_1936512041221649536_Bp.ecsv',
        'lc_tess_TPF_V0477_Lyr_TIC_423311936_sector_26_SPOC.ecsv'
    ]:
        main(filename_)
