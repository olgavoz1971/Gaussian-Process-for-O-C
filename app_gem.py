# --- Documentation Content ---

# DOC_MARKDOWN = """
# # Lightcurve Extrema Modeler (Working draft)
#
#
# ## The Core Concept: Non-Parametric Modeling
# Unlike traditional fitting, which assumes a fixed shape (like a parabola), this tool treats the lightcurve
# as a **stochastic process**. Specifically, we assume that our observations are realizations of
# a **Gaussian Process (GP)**.
#
# In the language of **sklearn.gaussian_process**, we are performing Bayesian inference in a function space.
# We don't ask "What are the parameters of this curve?", but rather "Which smooth functions are most likely
# to have generated these data points?"
#
# ## Why Gaussian Processes (GP)?
# When you have noisy time-series with gaps and asymmetrical details, GP is often superior
# to polynomials or splines because:
# * Gap-Resilient: It models the signal's covariance, avoiding artifacts in areas of uneven cadence or gaps.
# * Shape-Flexible: As a non-parametric tool, it captures rapid changes and asymmetries without
# the rigidity of fixed equations.
# * Noise-Aware: It separates observational jitter from the real signal, preventing outliers from biasing the fit.
#
# ---
#
# ## GP Parameters: Mathematical Meaning and Intuition
#
# The "behavior" of the GP is governed by the **Kernel** (the covariance function). We utilize a composite kernel:
# `Constant * Matern(nu=2.5) + WhiteKernel`.
#
# ### 1. RBF / Matern Length Scale (ℓ) * **Strong Term:** The *characteristic length scale* of the covariance. *
# **Intuition:** This is the "stiffness" of your curve. * **Small (ℓ):** The GP is very flexible and "wiggly." It
# will follow every small data bump. * **Large (ℓ):** The GP becomes stiff and smooth. * **Tweak Advice:** If the
# fit is "overshooting" or missing the peak by being too flat, **decrease** the maximum bound of the length scale. If
# it is fitting the noise (jitter) instead of the signal, **increase** the minimum bound.
#
# ### 2. White Noise Level (σ²_noise)
# * **Strong Term:** The *nugget effect* or the diagonal of the covariance matrix.
# * **Intuition:** This represents the photometric "floor" of your noise. It accounts for the fact that data points
# don't lie perfectly on a smooth line.
# * **Tweak Advice:** If the GP mean ignores your data points too much, the noise level might be set too high.
# Use the **Noise Divisor** or **Guess Sigma** to force the GP to trust the data points more.
#
# ### 3. Noise Divisor
# * **Intuition:** This is an empirical scaling factor for your input error bars ($flux\_err$).
# * **Tweak Advice:** If you suspect your original errors are overestimated (causing the GP to
# "smooth out" the real peak), increase this divisor to shrink the errors and "tighten" the fit.
#
# ## GP Parameters: practical hints:
# * GUESS_SIGMA
# Ignore provided flux(magnitude) unceratinties and estimate noise from scatter.
# Use ON if errors are missing/unreliable. Keep OFF if your uncertainties are trustworthy.
#
# * NOISE_SCALE_DIVISOR
# Empirical correction factor for noise estimated from data (used when GUESS_SIGMA=True).
# The noise estimate is only rough guess; this parameter lets the user adjust it.
# Higher values reduce assumed errors → more sensitive, more wiggly fit.
# Lower values increase assumed errors → smoother, more conservative fit.
# Typical tuning: increase if the GP is too flat; decrease if it starts fitting noise.
#
# * LENGTH_SCALE (MIN/ INIT / MAX)
# Controls how fast the GP can vary in time (i.e. smoothness).
# INIT: starting guess; set ~ half of a typical feature width (eclipse, peak).
# MIN: smallest allowed scale; prevents fitting noise (too small → overfitting).
# MAX: largest allowed scale; prevents over-smoothing (too large → missed structure).
# If fit is too wiggly → increase scales. If too smooth → decrease.
#
# * WHITE_NOISE_LEVEL (MIN /INIT / MAX)
# Extra noise the model can add on top of measurement errors.
# Use to compensate underestimated uncertainties.
# Higher values → smoother fit (more treated as noise).
# Lower values → GP tries to follow data more closely.
# If your flux(mag) uncertainties is reliable, keep these small and tightly bounded.
#
# * EXTREMA_MODE
# Select what to detect:
# 'min' - minima (eclipses)
# 'max' - maxima (peaks)
# Choose according to the feature of interest in the selected interval.
#
# ---
#
# ## Extremum Times and Uncertainty Estimation
#
# How do we find the "Time of Minimum/Maximum" and its error?
#
# 1.  **Mean Prediction:** We first find the extremum of the **GP Mean Map**. This is our most probable
# value JD_extremum.
# 2.  **Posterior Sampling:** To find the uncertainty, we draw _N_ random functions from
# the **Posterior Distribution**.
# 3.  **Distribution Analysis:** For each of these _N_ random "possible lightcurves" we find the peak.
# The spread (standard deviation) of these peaks gives us our **JD Uncertainty σ_JD**.
#
# This method is mathematically superior to simple "error propagation" because it accounts for the entire shape
# of the probability landscape around the feature.
#
# ---
#
# ## References
#
# * **Pedregosa et al. (2011):** [Scikit-learn: Machine Learning in Python]
# (https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), JMLR 12, pp. 2825-2830. Specifically the `GaussianProcessRegressor` module.
# * **Rasmussen & Williams (2006):** [Gaussian Processes for Machine Learning]
# (http://www.gaussianprocess.org/gpml/), MIT Press. (The "Bible" of GPs).
# """


DOC_MARKDOWN = """

# Lightcurve Extrema Modeler

## What this tool does
This tool models a selected fragment of a light curve and determines the time of a minimum or maximum, together with its uncertainty.

Instead of fitting a fixed function (e.g. parabola), it builds a **smooth probabilistic model** of the data using a **Gaussian Process (GP)**.  
In practice, this means:
- the model adapts to the actual shape of the light curve,
- it handles uneven sampling and gaps naturally,
- it accounts for observational noise.

You can think of it as drawing a smooth curve through the data, while also estimating how uncertain that curve is.

---

## When to use it
This method is useful when:
- the extremum is not perfectly symmetric,
- the data are noisy,
- there are gaps or irregular cadence,
- simple polynomial fits give unstable or biased results.

---

## How the model behaves
The model is controlled by a few parameters that define how “flexible” or “smooth” the curve is, and how much it trusts the data.

Internally, we use a kernel of the form:

`Constant × Matern(ν = 2.5) + White noise`

You do not need to work with this directly — the parameters below are the practical controls.

---

## Parameters

**GUESS_SIGMA**
Ignore provided uncertainties and estimate noise from data scatter.  
Use when `flux_err`(`mag_err`) are missing or unreliable.

---

**NOISE_SCALE_DIVISOR**
Empirical correction for estimated noise (used when `GUESS_SIGMA=True`).  
The automatic estimate is only approximate — this parameter lets you adjust it.

- Higher → smaller errors → more sensitive (wiggly) fit  
- Lower → larger errors → smoother fit  

Typical use:
- Increase if the model is too flat  
- Decrease if it starts fitting noise

---

**LENGTH_SCALE (MIN / INIT / MAX)**
Controls how quickly the model can change in time (smoothness).

- *INIT*: initial guess (≈ half of a typical feature width)  
- *MIN*: prevents fitting very small-scale noise  
- *MAX*: prevents over-smoothing  

Practical tuning:
- Too wiggly → increase values  
- Too smooth → decrease values  

---

**WHITE_NOISE_LEVEL (MIN / INIT / MAX)**
Additional noise the model can include beyond measurement errors.

Use this if your uncertainties are underestimated.

- Higher → smoother fit  
- Lower → model follows data more closely  

If your `flux_err` is reliable, keep this small.

---

**EXTREMA_MODE**
Select what to detect:
- `min` — minima (eclipses)  
- `max` — maxima (peaks)

---

## How the extremum and its uncertainty are computed

1. **Best estimate**  
   The extremum is taken from the smooth model curve (the GP mean).

2. **Uncertainty**  
   The model generates many possible light curves consistent with the data.  
   The extremum is measured for each of them.

3. **Final error**  
   The spread of these extrema gives the uncertainty in time.

This approach naturally accounts for noise, sampling, and shape of the feature.

---
## If you are curious how it works

### What is a Gaussian Process

A Gaussian Process (GP) is a way to describe **a whole family of possible curves** that could pass 
through your data. (It does not fit the data with a Gaussian-shaped function; 
the term Gaussian there refers to the use of the normal distribution).

Instead of choosing a fixed formula (parabola, spline, etc.), we assume:
> “The true lightcurve is one of many smooth functions consistent with the observations.”

The model does not try to find *the* curve.  
It assigns a **probability to every possible curve**, favouring those that:
- pass near the data points,
- remain reasonably smooth,
- respect the assumed noise level.

---

### Why “Gaussian”?
The term *Gaussian* refers to how uncertainties are described.

For any set of time points, the model assumes that the flux values follow a **multivariate normal distribution**.  
In simpler terms:
- each point has an uncertainty,
- nearby points are **correlated** (they tend to vary together),
- this correlation is described in a mathematically convenient way.

This assumption makes the problem tractable and stable, while still being flexible enough for real light curves.

---

### What is actually being modeled?
The key idea is not the curve itself, but **how points are related to each other**.

We define:
- how strongly two observations are connected depending on their time separation,
- how smooth the curve should be,
- how much noise is present.

From this, the GP builds a smooth curve that:
- follows large-scale structure,
- ignores random scatter,
- adapts to asymmetry and irregular sampling.

The result is:
- a most probable curve (the mean),
- and a full description of uncertainty around it.

---

### Implementation
This tool uses the `GaussianProcessRegressor` from the  
`sklearn.gaussian_process` module in Python.

---

## References
- **Pedregosa et al. (2011):** [Scikit-learn: Machine Learning in Python]
(https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), JMLR 12, pp. 2825-2830. Specifically the `GaussianProcessRegressor` module.
- **Rasmussen & Williams (2006):** [Gaussian Processes for Machine Learning]
(http://www.gaussianprocess.org/gpml/), MIT Press. (The "Bible" of GPs).
"""

import dash
import diskcache
import pandas as pd
import json
from dash import dcc, html, Input, Output, State, ALL, DiskcacheManager
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import callback_context

import base64
import io
import numpy as np

import traceback
import logging

from gp import (
    GUESS_SIGMA, LEN_MIN,
    read_lc, load_intervals, add_flux, select_jd_interval, gp_peak_pipeline,
    NOISE_SCALE_DIVISOR,
    LENGTH_SCALE_INIT, LENGTH_SCALE_MIN, LENGTH_SCALE_MAX,
    WHITE_NOISE_LEVEL_INIT, WHITE_NOISE_LEVEL_MIN, WHITE_NOISE_LEVEL_MAX
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEBUG = False

params_float = {
    "noise_scale_divisor": NOISE_SCALE_DIVISOR,
    "length_scale_init": LENGTH_SCALE_INIT,
    "length_scale_min": LENGTH_SCALE_MIN,
    "length_scale_max": LENGTH_SCALE_MAX,
    "white_noise_level_init": WHITE_NOISE_LEVEL_INIT,
    "white_noise_level_min": WHITE_NOISE_LEVEL_MIN,
    "white_noise_level_max": WHITE_NOISE_LEVEL_MAX
}
# Initialize Diskcache for background callbacks
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = dash.Dash(__name__,
                external_stylesheets=[
                    dbc.themes.FLATLY,
                    dbc.icons.BOOTSTRAP
                ],
                background_callback_manager=background_callback_manager
                )


# ==================== Layout utils ===================

def create_float_input(label, default_val, tooltip_text, index, step=1.0):
    """Helper to create labeled float inputs with tooltips."""
    return html.Div([
        dbc.Label(label, id=f"label-{index}"),
        dbc.Tooltip(tooltip_text, target=f"label-{index}"),
        dbc.Input(id={'type': 'float-input', 'index': index},
                  type="number", value=default_val, step=step),
    ], className="mb-2")


# def create_parameter_triple(main_label, main_tooltip, prefix, defaults: dict, step=0.001):
#     """
#     Creates a grouped set of 3 inputs (Min, Init, Max) with a common header.
#     - prefix: the base string for the dictionary keys (e.g., 'white_noise_level')
#     - defaults: the params_float dictionary
#     """
#     return html.Div([
#         # Main Category Label with Tooltip
#         html.Div([
#             html.B(main_label, id=f"triple-label-{prefix}", style={"cursor": "help"}),
#             dbc.Tooltip(main_tooltip, target=f"triple-label-{prefix}"),
#         ], className="mt-3 mb-1"),
#
#         # Row of 3 inputs
#         dbc.Row([
#             # MIN
#             dbc.Col([
#                 dbc.Input(id={'type': 'float-input', 'index': f"{prefix}_min"},
#                           type="number", value=defaults[f"{prefix}_min"], step=step, size="sm"),
#                 html.Small("Min", className="text-muted d-block text-center")
#             ], width=4, className="pe-1"),
#
#             # INIT
#             dbc.Col([
#                 dbc.Input(id={'type': 'float-input', 'index': f"{prefix}_init"},
#                           type="number", value=defaults[f"{prefix}_init"], step=step, size="sm",
#                           style={"border-color": "#007bff"}),  # Highlight initial guess
#                 html.Small("Init", className="text-muted d-block text-center")
#             ], width=4, className="px-1"),
#
#             # MAX
#             dbc.Col([
#                 dbc.Input(id={'type': 'float-input', 'index': f"{prefix}_max"},
#                           type="number", value=defaults[f"{prefix}_max"], step=step, size="sm"),
#                 html.Small("Max", className="text-muted d-block text-center")
#             ], width=4, className="ps-1"),
#         ], className="g-0")  # No gutters for maximum compactness
#     ], className="mb-2 p-2 border-bottom")


def LegendItem(color, label, mode='line'):
    # mode can be: 'line', 'dashed', or 'circle'
    # Base style for the visual indicator
    style = {
        "display": "inline-block",
        "margin-right": "10px",
        "vertical-align": "middle",
    }

    if mode == 'circle':
        style.update({
            "background-color": color,
            "width": "10px",
            "height": "10px",
            "border-radius": "50%"
        })
    elif mode == 'line':
        style.update({
            "background-color": color,
            "width": "20px",
            "height": "3px",
            "border-radius": "0px"
        })
    elif mode == 'dashed':
        # For dashed, we use a border instead of background
        style.update({
            "width": "20px",
            "height": "0px",
            "border-top": f"3px dashed {color}",
            "background-color": "transparent"
        })

    return html.Div([
        html.Span(style=style),
        html.Span(label, style={"font-size": "0.85rem", "vertical-align": "middle"})
    ], style={"margin-bottom": "4px", "display": "flex", "align-items": "center"})


# ==================== Lightcurve GUI ==================

sidebar_lc = html.Div([
    # 1. PHASE FOLDING CONTROLS
    html.Label("Phase Folding", className="fw-bold", id='phase-folding-label'),
    dbc.Tooltip('Out of operation', target="phase-folding-label"),
    dbc.Checklist(
        options=[{"label": "Fold", "value": 1}],  # type: ignore
        value=[],
        id="folding-switch",
        switch=True,
        className="mb-2"
    ),
    dbc.InputGroup([
        dbc.InputGroupText("P"),
        dbc.Input(id="input-period", type="number", placeholder="Period (days)"),
    ], size="sm", className="mb-1"),
    dbc.InputGroup([
        dbc.InputGroupText("T₀"),
        dbc.Input(id="input-epoch", type="number", placeholder="Epoch (JD)"),
    ], size="sm", className="mb-3"),

    html.Hr(),

    # 2. VIEW SETTINGS
    html.Label("View Settings", className="fw-bold", id="view-settings-label"),
    dbc.Tooltip('Out of operation', target="view-settings-label"),
    dbc.RadioItems(
        options=[  # type: ignore
            {"label": "Magnitudes", "value": "mag"},
            {"label": "Flux", "value": "flux"},
        ],
        value="mag",
        id="view-mode-radio",
        className="mb-3",
        style={"fontSize": "0.9rem"}
    ),

    html.Hr(),

    #  3. ACTION BUTTONS
    html.Label("Interval control", className="fw-bold"),
    dbc.Button(
        [html.I(className="bi bi-plus-circle me-2"), "Add Interval"],
        id="btn-add-interval", color="primary", className="w-100 mb-2"
    ),
    dbc.Tooltip('Register selected JD interval as a target for extremum analysis', target="btn-add-interval"),
    # CLEAR button
    dbc.Button(
        [html.I(className="bi bi-trash3 me-2"), "Clear All Intervals"],
        id="btn-clear-intervals", color="outline-danger",
        className="w-100", size="sm"
    ),

    html.Hr(),

    # 4. --- EXPORT ---
    html.Label("Export Settings", className="fw-bold"),
    # dbc.InputGroup([
    #     dbc.InputGroupText("Filename"),
    dbc.Input(
        id="export-intervals-filename",
        placeholder="intervals_export",
        type="text",
        value="my_intervals"  # Default value
    ),
    # dbc.InputGroupText(".intervals"),
    # ], size="sm", className="mb-2"),

    # DOWNLOAD button
    dbc.Button(
        [html.I(className="bi bi-download me-2"), "Download File"],
        id="btn-download-intervals",
        color="success",
        className="w-100 mb-2",
        size="sm"
    ),
    dcc.Download(id="download-intervals-file")
], className="p-3 bg-light border rounded shadow-sm")

graph_lc = html.Div([
    dcc.Graph(
        id='prep-graph',
        config={  # type: ignore
            'scrollZoom': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'zoomIn2d',  # Hide Zoom In
                'zoomOut2d',  # Hide Zoom Out
                'lasso2d',  # Hide Lasso
                # 'select2d'  # Hide the default box-select (if you only want 'drawrect')
            ],
            'modeBarButtonsToAdd': ['drawrect', 'eraseshape'],
            # 'showAxisDragHandles': True,
            # 'showAxisRangeEntryBoxes': True,
        },
        # responsive=True,  # This tells Plotly to watch the container
        style={'height': '600px'}
    ),
    dbc.Alert(
        "Tip: Use the 'Box Select' tool to highlight a region for the GP fit.",
        color="info", className="mt-2 py-1 small"
    )
], className="border rounded p-2 bg-white")

intervals_registry = html.Div([
    html.H6("Selected Intervals", className="fw-bold mb-3"),
    html.Div(id='registry-list-container', children=[
        # We'll use a Dash Table or a List of Cards here
        html.P("No intervals selected.", className="text-muted small")
    ])
], className="p-3 border rounded bg-light", style={'height': '500px', 'overflowY': 'auto'})

registry_toggle_btn = dbc.Button(
    # region unfold
    html.I(className="bi bi-chevron-right", id="registry-toggle-icon"),
    id="btn-toggle-registry",
    color="light",
    size="sm",
    className="border shadow-sm p-1",
    style={
        "position": "absolute", "right": "0", "top": "50%",
        "zIndex": "1000", "transform": "translateX(50%)",
        "borderRadius": "50%", "width": "30px", "height": "30px"
    }
    # endregion
)
# ===================  GP GUI ===========================


sidebar_gp = html.Div([
    # 1. COLLAPSIBLE LEGEND
    dbc.Button(
        "Show Legend",
        id="toggle-legend-btn",
        color="link",
        # size="sm",
        className="p-0 mb-2 text-decoration-none"
    ),
    dbc.Collapse(
        html.Div([
            LegendItem("black", "Data Points", mode='circle'),
            LegendItem("rgb(31, 119, 180)", "GP Mean", mode='line'),
            LegendItem("rgba(31, 119, 180, 0.25)", "GP ±1σ Confidence", mode='line'),
            LegendItem("magenta", "Peak Estimate", mode='dashed'),
            LegendItem("orange", "Posterior Draws", mode='circle'),
            LegendItem("green", "Guess", mode='dashed'),
        ], className="p-2 border rounded bg-white mb-3 small"),
        id="legend-collapse",
        is_open=False
    ),

    # 2. PRIMARY ACTION BUTTONS
    dbc.Row([
        dbc.Col(dbc.Button("Run GP", id="run-btn", color="primary", className="w-100"), width=7),
        dbc.Col(dbc.Button("Cancel", id="cancel-btn", color="danger", outline=True, className="w-100"), width=5),
    ], className="g-2 mb-3"),

    # 3. GLOBAL MODEL SETTINGS
    dbc.Row([
        dbc.Col([
            dbc.Select(
                id="extrema-mode",
                options=[  # type: ignore
                    {"label": "Search Minima", "value": "min"},
                    {"label": "Search Maxima", "value": "max"},
                ],
                value="max",
                size="sm",
            )
        ], width=12),

    ], className="g-2 mb-2"),

    dbc.Row([
        dbc.Col([html.Label("Guess sigma", className="small fw-bold",
                            id='guess-sigma-label')], width=6),
        dbc.Col([html.Label("Noise divisor", className="small fw-bold",
                            id='noise-divisor-label')], width=6),
        dbc.Tooltip("Empirical noise correction (↑ wiggly, ↓ smooth)", target='noise-divisor-label'),
        dbc.Tooltip("Auto-estimate noise (ignore provided uncertainties)", target='guess-sigma-label')
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Checkbox(id="guess-sigma", value=GUESS_SIGMA, className="form-check-input"),
            dbc.Label("", html_for="guess-sigma", className="small ms-2"),
        ], width=6, className="d-flex align-items-center mb-3"),
        dbc.Col([
            dbc.Input(
                id={"type": "float-input", "index": "noise_scale_divisor"},
                type="number", size="sm", step=0.1,
                value=params_float["noise_scale_divisor"]
            ),
            # dbc.Tooltip("Noise Divisor", target={"type": "float-input", "index": "noise_scale_divisor"})
        ], width=6),
    ]),

    html.Hr(className="my-2"),

    # 4. KERNEL PARAMETERS (Compact Triples)
    html.Label("White Kernel (Min / Val /  Max)", className="small fw-bold", id='wk-label'),
    dbc.Tooltip("Extra noise allowance", target='wk-label'),
    dbc.Row([
        dbc.Col(dbc.Input(
            id={'type': 'float-input', 'index': "white_noise_level_min"},
            size="sm",
            type="number", step=0.001,
            value=params_float["white_noise_level_min"]), width=4),
        dbc.Col(dbc.Input(
            id={'type': 'float-input', 'index': "white_noise_level_init"},
            size="sm",
            type="number", step=0.001,
            value=params_float["white_noise_level_init"]), width=4),
        dbc.Col(dbc.Input(
            id={'type': 'float-input', 'index': "white_noise_level_max"},
            size="sm",
            type="number", step=0.001,
            value=params_float["white_noise_level_max"]), width=4),
    ]),

    html.Label("Length Scale ( Min / Val / Max)", className="small fw-bold", id='ls-label'),
    dbc.Tooltip("GP smoothness (in days, as x-axis). Increase if fit is too wiggly, "
                "decrease if it misses structure.", target='ls-label'),
    dbc.Row([
        dbc.Col(dbc.Input(
            id={'type': 'float-input', 'index': "length_scale_min"},
            size="sm",
            type="number", step=0.001,
            value=params_float["length_scale_min"]), width=4),
        dbc.Col(dbc.Input(
            size="sm",
            id={'type': 'float-input', 'index': "length_scale_init"},
            type="number", step=0.001,
            value=params_float["length_scale_init"]), width=4),
        dbc.Col(dbc.Input(
            size="sm",
            id={'type': 'float-input', 'index': "length_scale_max"},
            type="number", step=0.001,
            value=params_float["length_scale_max"]), width=4),
    ], className="g-1 mb-3"),

    dbc.Button("Reset Defaults", id="reset-btn", color="secondary", outline=True, size="sm", className="w-100 mt-2"),

], className="p-3 bg-light border rounded shadow-sm")


# =====================  LAYOUT ==================================================
graph_gp = html.Div([
    html.Div(id='finished-signal', style={'display': 'none'}),
    # signal for gp status, Allowed: "WAITING" and "FINISHED"

    # 1. DYNAMIC HEADER: Changes based on state
    html.Div(id='gp-header-area', children=[
        html.Div([
            html.H5("GP Processing View", className="fw-bold mb-0"),
            dbc.Badge("Waiting for Run", color="secondary", id='gp-view-badge', className="ms-2")
        ], className="d-flex align-items-center mb-3")
    ]),

    # 1. LIVE VIEW: Shows only graphs, no interactivity
    dbc.Row(id='live-graphs-container', style={'display': 'flex'}, className="g-2"),

    # 3. FINAL REVIEW
    html.Div(id='final-review-container', style={'display': 'none'}, children=[
        html.Hr(className="my-4"),
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.H6("Review and Export", className="mb-0 fw-bold"), width="auto"),
                    dbc.Col(dbc.Button("Select All", id="select-all-btn", size="sm", color="link"), width="auto"),
                    dbc.Col(dbc.Button("Unselect All", id="unselect-all-btn", size="sm", color="link"), width="auto"),

                    # Compact Filename + Download Group
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.Input(id="export-filename", placeholder="results.dat", type="text", size="sm"),
                            dbc.Button([html.I(className="bi bi-download me-2"), "Download"],
                                       id="save-file-btn", color="success", size="sm"),
                        ], className="ms-auto", style={"width": "350px"})
                    ], width="auto", className="ms-auto"),
                ], className="align-items-center"),
            ], className="py-2")  # Thinner padding
        ], className="bg-light mb-3"),

        dbc.Row(id='graphs-container', className="g-2"),
    ]),

    dcc.Download(id="download-results"),
    dcc.Store(id='store-results-data')
], className="p-2")


app.layout = dbc.Container([
    # --- HEADER SECTION ---
    dbc.Row([
        dbc.Col([
            # html.H1("Astro-GP", className="display-4 text-primary mb-0"),
            # html.P("Lightcurves Extrema Modeler", className="lead text-muted")
            html.H1("Lightcurve Extrema Modeler (working draft)")  # , className="display-4 text-primary mb-0")
        ], width="auto"),
        dbc.Col(
            dbc.Button([html.I(className="bi bi-question-circle me-2"), "About"],
                       id="open-help", color="outline-secondary", className="mb-2"),
            width="auto", className="ms-auto d-flex align-items-end"
        ),
    ], className="mb-4 border-bottom pb-3"),

    # --- THE HELP MODAL ---
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Description")),
        dbc.ModalBody(
            dcc.Markdown(DOC_MARKDOWN, dangerously_allow_html=True),
            style={"maxHeight": "75vh", "overflowY": "auto"}
        ),
        dbc.ModalFooter(
            dbc.Button("Stop talking!", id="close-help", className="ms-auto", n_clicks=0)
        ),
    ], id="help-modal", size="xl", is_open=False),

    dcc.Store(id='store-lc-data'),
    dcc.Store(id='store-intervals-data'),
    dcc.Store(id='store-active-intervals-name', data="Drag or Select"),

    # --- 2. GLOBAL DATA HUB

    dbc.Card([
        # dbc.CardHeader(html.B("Data Management")),
        dbc.CardBody([
            dbc.Row([
                # Lightcurve Upload
                dbc.Col([
                    html.Label("Lightcurve", className="small fw-bold"),
                    dcc.Upload(
                        id='upload-lc',
                        children=html.Div(['Drag or ', html.A('Select')], id='upload-lc-text'),
                        className="upload-box",  # We style this in the local style.css
                        style={'border': '1px dashed', 'padding': '5px', 'borderRadius': '5px', 'textAlign': 'center'},
                    ),
                ], width=6),

                # Intervals Upload
                dbc.Col([
                    html.Label("Intervals", className="small fw-bold"),
                    dcc.Upload(
                        id='upload-intervals',
                        children=html.Div(['Drag or ', html.A('Select')], id='upload-intervals-text'),
                        className="upload-box",  # We style this in the local style.css
                        style={'border': '1px dashed', 'padding': '5px', 'borderRadius': '5px', 'textAlign': 'center'},
                    ),
                ], width=6),

                # # Global Summary Metrics
                # dbc.Col([
                #     html.Div([
                #         html.P(id='data-summary-text', children="No data loaded.",
                #                className="text-muted small mb-0")
                #     ], className="p-2 border rounded bg-light", style={'height': '75px'})
                # ], width=4),
            ])
        ])
    ], className="mb-4 shadow-sm"),

    # --- 3. THE WORKFLOW ACCORDION ---

    dbc.Accordion([
        dbc.AccordionItem(
            item_id="accordion-lc",
            title="Lightcurve and Intervals",
            children=[
                dbc.Row([
                    # Sidebar (Column 1 - Fixed)
                    dbc.Col(sidebar_lc, width=3),

                    # Working Area (Column 2 & 3 combined into a Flex container)
                    dbc.Col([
                        html.Div([
                            # Graph Area (Grows automatically)
                            html.Div([
                                graph_lc,
                                registry_toggle_btn
                            ],
                                style={
                                "flex": "1",
                                "minWidth": "0",
                                "position": "relative",
                                "transition": "none",  # Kill the animation for instant speed
                                # "transition": "flex 0.3s ease"
                            } ),

                            # Registry Area (Collapses horizontally)
                            dbc.Collapse(
                                intervals_registry,
                                id="registry-collapse",
                                is_open=True,
                                dimension="width",
                                style={
                                    # "transition": "0.3s ease",
                                    "transition": "none",  # Kill the animation for instant speed
                                    "flexShrink": "0",
                                    # "minWidth": "0px"
                                }
                            )
                        ], style={
                            "display": "flex",
                            "flexDirection": "row",
                            "flexWrap": "nowrap",
                            "overflow": "hidden",
                            "alignItems": "stretch"
                        })
                    ], width=9)
                    # dbc.Col(graph_lc, width=6),
                    # dbc.Col(intervals_registry, width=3),  # COLUMN 3: THE REGISTRY TABLE (intervals)
                ]),
            ]),

        # --- STEP 2: ANALYSIS (The "Monster") ---
        dbc.AccordionItem(
            item_id="accordion-gp",
            title="Gaussian Process",
            children=[
                dbc.Row([
                    dbc.Col(sidebar_gp, width=3),
                    dbc.Col(graph_gp, width=9),
                ]),
                html.Div(id="main-analysis-wrapper")
            ],
        ),
    ], id="main-workflow-accordion", active_item="accordion-lc"),
], fluid=True)


# ===================== CALLBACKS ================================================

@app.callback(
    # region unfold
    Output("help-modal", "is_open"),
    [Input("open-help", "n_clicks"), Input("close-help", "n_clicks")],
    [State("help-modal", "is_open")],
    # endregion
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# --- Open/Close registry
@app.callback(
    # region unfold
    Output("registry-collapse", "is_open"),
    Output("registry-toggle-icon", "className"),
    Input("btn-toggle-registry", "n_clicks"),
    State("registry-collapse", "is_open"),
    prevent_initial_call=True
    # endregion
)
def toggle_registry(n_clicks, is_open):
    if is_open:
        # Closing: return is_open=False and the Left arrow icon
        return False, "bi bi-chevron-left"
    else:
        # Opening: return is_open=True and the Right arrow icon
        return True, "bi bi-chevron-right"


@app.callback(
    # region unfold
    Output("legend-collapse", "is_open"),
    Output("toggle-legend-btn", "children"),
    Input("toggle-legend-btn", "n_clicks"),
    State("legend-collapse", "is_open"),
    prevent_initial_call=True
    # endregion
)
def toggle_gp_legend(n_clicks, is_open):
    if is_open:
        return False, "Show Legend"
    return True, "Hide Legend"


# ------ lightcurve visualisation -------
@app.callback(
    # region unfold
    Output('prep-graph', 'figure'),
    Input('store-lc-data', 'data'),  # Assuming data is stored here after upload
    Input('store-intervals-data', 'data'),
    Input('folding-switch', 'value'),
    Input('input-period', 'value'),
    Input('input-epoch', 'value'),
    Input('view-mode-radio', 'value')
    # endregion
)
def update_prep_graph(lc_json_string, intervals_data, folding_on, period, epoch, view_mode):
    if not lc_json_string:
        return go.Figure().update_layout(title="Upload data to see plot")

    di = json.loads(lc_json_string)
    df = pd.DataFrame(data=di['data'], columns=di['columns'])

    x_data = df['jd']
    y_data = df['mag']
    x_label = "Julian Date (JD)"

    if folding_on and period and period > 0:
        t0 = epoch if epoch is not None else x_data.min()
        x_data = ((x_data - t0) / period) % 1.0
        x_label = f"Phase (P={period} d, T₀={t0})"

    # 3. Build Figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data,
        mode='markers',
        hoverinfo='none',
        marker=dict(
            size=4,
            color='blue',  # Standard Plotly Blue, or use '#003366' for even darker
            opacity=0.7,
            line=dict(width=0.5, color='White')  # Adds definition to overlapping points
        ),
        name="Data"
    ))

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="Magnitude" if view_mode == 'mag' else "Flux",
        yaxis_autorange='reversed' if view_mode == 'mag' else True,
        margin=dict(l=10, r=10, t=20, b=40),
        template="plotly_white",
        hovermode=False,
        dragmode='pan',  # 'zoom',  # Default to selection tool
        # Optional: Make the selection color distinctive (e.g., gold)
        # so it's clear when the user switches to the Box tool
        selectdirection='h',  # Usually we only care about the JD range (horizontal)
        # newshape=dict(line_color='#FFD700', fillcolor='#FFD700', opacity=0.3),
        # This controls BOTH drawline and drawrect
        newshape=dict(
            line_color='red',  # Much easier to see than yellow
            line_width=3,
            opacity=0.5,
            # drawdirection='horizontal'      # Helps keep the ruler straight if you only want JD
        ),
        # THE RULER SETTING: This enables the pop-up labels for distance
        # modebar=dict(
        #     addcategory='drawlabel'     # Some versions of Plotly use this to force labels
        # ),
        # xaxis=dict(
        #     showspikes=True,  # Show the vertical line
        #     spikemode='across',
        #     spikesnap='cursor',
        #     spikethickness=1,
        #     spikedash='dash',
        #     spikecolor='grey'
        # ),
        # yaxis=dict(
        #     showspikes=True,  # Show the horizontal line
        #     spikemode='across',
        #     spikesnap='cursor',
        #     spikethickness=1,
        #     spikedash='dash',
        #     spikecolor='grey'
        # ),
        # hovermode='x',
        # modebar=dict(
        #     orientation='h',
        #     bgcolor='rgba(255,255,255,0.7)'
        # )
    )

    # 4. Mark selected intervals if any
    if intervals_data:
        for i, interval in enumerate(intervals_data):
            # interval is [jd_start, jd_end]
            fig.add_vrect(
                x0=interval[0], x1=interval[1],
                fillcolor="green", opacity=0.15,
                layer="below", line_width=1,
                line_color="green",
                # annotation_text=f"{i + 1}",
                # annotation_position="top left"
            )

    return fig


# ------ Lightcurve ----

@app.callback(
    # region unfold me
    Output('store-lc-data', 'data'),
    Output('upload-lc-text', 'children'),  # Targets the text inside the box
    # Input('upload-lc-prep', 'contents'),
    Input('upload-lc', 'contents'),
    State('upload-lc', 'filename'),  # Grabs the filename
    prevent_initial_call=True
    # endregion
)
def upload_lc(contents, filename):
    print(f'Uploading {filename}')
    if contents is None:
        return dash.no_update, dash.no_update
    # Decode the upload
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = read_lc(io.BytesIO(decoded))
        df = add_flux(df)

        # Serialise to JSON for dcc.Store
        lc = df.to_dict(orient='split', index=False)
        lc_data = json.dumps(lc)
        # --- UI Success Feedback ---
        new_label = html.Div([
            html.I(className="bi bi-check-circle-fill me-2", style={"color": "#28a745"}),
            html.Span(f"{filename}", style={"fontSize": "0.9rem", "fontWeight": "bold"})
        ])

        return lc_data, new_label

    except Exception as e:
        # 1. Detailed Console Logging
        logging.error(f"Failed to process file: {filename}")
        logging.error(traceback.format_exc())  # Prints the full stack trace to terminal
        # 2. User-Friendly but Verbose GUI Feedback
        # error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)

        return dash.no_update, html.Div([
            html.I(className="bi bi-exclamation-triangle-fill me-2", style={"color": "#dc3545"}),
            # Terse text with the Tooltip target
            html.Span(
                [html.B("Error: "), filename],
                id=f"err-target-{filename.replace('.', '-')}",  # Clean ID (no dots)
                style={"color": "#dc3545", "fontSize": "0.85rem", "cursor": "help"}
            ),
            # The full error message only appears on hover
            dbc.Tooltip(
                f"Traceback: {str(e)}",
                target=f"err-target-{filename.replace('.', '-')}",
                placement="right",
                style={"fontSize": "0.75rem"}
            ),
        ])


# Callbacks for Intervals file upload/download/signs

@app.callback(
    Output('upload-intervals-text', 'children'),
    Input('store-active-intervals-name', 'data')
)
def update_global_intervals_label(stored_content):
    if not stored_content:
        return html.Div(['Drag or ', html.A('Select')])

    return stored_content
    # return html.Div([
    #     html.I(className="bi bi-check-circle-fill me-2", style={"color": "#28a745"}),
    #     html.B(active_name)
    # ])


@app.callback(
    # region unfold me
    Output('store-intervals-data', 'data'),
    Output('store-active-intervals-name', 'data', allow_duplicate=True),
    # Output('upload-intervals-text', 'children'),
    Input('upload-intervals', 'contents'),
    State('upload-intervals', 'filename'),
    prevent_initial_call=True
    # endregion
)
def upload_intervals(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Convert bytes -> text -> StringIO
        text = decoded.decode('utf-8', errors='ignore')  # "ignore"  to prevent the whole app
        # from crashing over a single non-ASCII character.
        intervals_list = load_intervals(io.StringIO(text))
        success_ui = html.Div([
            html.I(className="bi bi-check-circle-fill me-2", style={"color": "#28a745"}),
            html.B(filename)
        ])
        return intervals_list, success_ui
        # --- Success UI ---
        # return intervals_list, filename
        # return intervals_list, html.Div([
        #     html.I(className="bi bi-check-circle-fill me-2", style={"color": "#28a745"}),
        #     html.B(filename)
        # ])
    except Exception as e:
        # 1. Log full traceback for terminal
        logging.error(f"Error processing interval file {filename}:")
        logging.error(traceback.format_exc())

        # 2. Terse UI for sidebar_gp with hover details
        # We replace dots/spaces with hyphens for a valid HTML ID
        safe_id = f"err-int-{filename.replace('.', '-').replace(' ', '-')}"
        error_ui = html.Div([
            html.I(className="bi bi-exclamation-octagon-fill me-2", style={"color": "#dc3545"}),
            html.Span(
                [html.B("Error: "), filename],
                id=safe_id,
                style={"color": "#dc3545", "fontSize": "0.85rem", "cursor": "help"}
            ),
            dbc.Tooltip(f"Interval Load Error: {str(e)}", target=safe_id)
        ])
        return dash.no_update, error_ui
        # return dash.no_update, html.Div([
        #     html.I(className="bi bi-exclamation-octagon-fill me-2", style={"color": "#dc3545"}),
        #     html.Span(
        #         [html.B("Error: "), filename],
        #         id=safe_id,
        #         style={"color": "#dc3545", "fontSize": "0.85rem", "cursor": "help"}
        #     ),
        #     dbc.Tooltip(
        #         f"Interval Load Error: {str(e)}",
        #         target=safe_id,
        #         placement="right",
        #         style={"fontSize": "0.75rem"}
        #     )
        # ])

    # content_type, content_string = contents.split(',')
    # decoded = base64.b64decode(content_string)
    #
    # # convert bytes -> text -> StringIO
    # text = decoded.decode('utf-8')
    # intervals_list = load_intervals(io.StringIO(text))
    # return intervals_list


@app.callback(
    # region infold
    Output("download-intervals-file", "data"),
    Output("store-active-intervals-name", "data"),  # Store the "real" name here
    Input("btn-download-intervals", "n_clicks"),
    State("store-intervals-data", "data"),
    State("export-intervals-filename", "value"),
    prevent_initial_call=True,
    # endregion
)
def download_intervals(n_clicks, intervals, custom_name):
    if not n_clicks or not intervals:
        return dash.no_update, dash.no_update

    # 1. Create the string content
    # Format: Start_JD  End_JD
    content = "# Interval_Start  Interval_End\n"
    for start, end in intervals:
        content += f"{start:<20} {end:<20}\n"

    # 2. Use the custom name (fallback to default if empty)
    export_name = custom_name if custom_name else "my_intervals.dat"

    # --- DOWNLOAD SUCCESS UI ---
    download_ui = html.Div([
        html.I(className="bi bi-check-circle-fill me-2", style={"color": "#007bff"}),
        html.B(export_name)
    ])

    return dict(content=content, filename=export_name), download_ui
    # return dict(content=content, filename=export_name), export_name


@app.callback(
    # region unfold
    Output('store-intervals-data', 'data', allow_duplicate=True),
    Input('btn-clear-intervals', 'n_clicks'),
    prevent_initial_call=True
    # endregion
)
def clear_all_intervals(n_clicks):
    if n_clicks:
        return []  # Return empty list to the store
    return dash.no_update


# ------- graphical interval selection
@app.callback(
    # region infold
    Output('store-intervals-data', 'data', allow_duplicate=True),
    Input('btn-add-interval', 'n_clicks'),
    State('prep-graph', 'selectedData'),
    State('store-intervals-data', 'data'),
    State('folding-switch', 'value'),  # We need to know if we are in Phase or JD!
    prevent_initial_call=True
    # endregion
)
def add_selection_to_registry(n_clicks, selected_data, current_intervals, folding_on):
    if not n_clicks or not selected_data:
        return dash.no_update

    # 1. Extract the X-range from the selection
    # SelectedData contains 'range': {'x': [min, max]}
    if 'range' in selected_data:
        x_min, x_max = selected_data['range']['x']

        # BETA-TESTER WARNING:
        # If folding is ON, x_min/max are PHASES (0-1).
        # If folding is OFF, they are JD.
        # For now, let's assume the user selects in JD (unfolded) mode.
        if folding_on:
            # We might want to warn the user or handle phase-to-jd conversion later
            return dash.no_update

            # 2. Append to our list
        new_interval = [round(x_min, 6), round(x_max, 6)]

        # current_intervals is usually a list of lists: [[start1, end1], [start2, end2]]
        updated_list = current_intervals if current_intervals else []

        # Prevent exact duplicates
        if new_interval not in updated_list:
            updated_list.append(new_interval)
            # Sort by JD start time
            updated_list.sort(key=lambda x: x[0])

        return updated_list

    return dash.no_update


# ------- interval registry stuff ----
@app.callback(
    Output('registry-list-container', 'children'),
    Input('store-intervals-data', 'data')
)
def render_registry(intervals):
    if not intervals:
        return html.P("No intervals selected.", className="text-muted small italic")

    cards = []
    for i, interval in enumerate(intervals):
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            # html.Small(f"Interval {i+1}", className="text-muted d-block"),
                            html.B(f"{interval[0]:.3f}", className="small"),
                            html.Span(" - ", className="mx-1"),
                            html.B(f"{interval[1]:.3f}", className="small"),
                        ], width=9),
                        dbc.Col([
                            dbc.Button(
                                html.I(className="bi bi-trash"),
                                id={'type': 'del-int', 'index': i},
                                color="link",  # Removes the button box entirely
                                className="text-danger p-0",  # "text-danger" keeps the icon red
                                style={"textDecoration": "none", "fontSize": "0.9rem"},
                                title="Delete"
                            )
                        ], width=3, className="text-end")
                    ], className="align-items-center")
                ], className="p-2")
            ], className="mb-2 shadow-sm")
        )
    return cards


# --------  Delete individual interval ----------
@app.callback(
    # region unfold
    Output('store-intervals-data', 'data', allow_duplicate=True),
    Input({'type': 'del-int', 'index': ALL}, 'n_clicks'),
    State('store-intervals-data', 'data'),
    prevent_initial_call=True
    # endregion
)
def delete_interval(n_clicks_list, current_intervals):
    # Check if any button was actually clicked
    if not any(n_clicks_list):
        return dash.no_update

    # Find which button index was triggered
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    # Extract the index from the triggered ID string
    # e.g., '{"index":2,"type":"del-int"}.n_clicks' -> 2
    import json
    triggered_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
    idx_to_remove = triggered_id['index']

    # Remove the item and return the new list
    if current_intervals and idx_to_remove < len(current_intervals):
        new_list = [item for i, item in enumerate(current_intervals) if i != idx_to_remove]
        return new_list

    return dash.no_update


# ================= GP callbacks ===================

@app.callback(
    # region unfold
    Output('gp-header-area', 'children'),
    Input('run-btn', 'n_clicks'),          # User clicks Run
    Input('finished-signal', 'children'),  # The Monster finishes
    prevent_initial_call=True
    # endregion
)
def update_gp_status_ui(n_clicks, signal_status):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # 1. THE START: User clicks Run
    if trigger_id == 'run-btn' and n_clicks > 0:
        return html.Div([
            html.H5("GP Processing View", className="fw-bold mb-0"),
            dbc.Badge([
                html.I(className="bi bi-hourglass-split me-2"),
                "RUNNING - Modeling..."
            ], color="warning", className="ms-2")
        ], className="d-flex align-items-center mb-3")

    # 2. THE INTERMEDIATE: Plotting has started but isn't over
    if trigger_id == 'finished-signal' and signal_status == "WAITING":
        return html.Div([
            html.H5("GP Processing View", className="fw-bold mb-0"),
            dbc.Badge([
                html.I(className="bi bi-gear-wide-connected me-2"),
                "GENERATING PLOTS..."
            ], color="info", className="ms-2")
        ], className="d-flex align-items-center mb-3")

    # 3. THE END: Only show the "Finished" badge for the explicit final signal
    if trigger_id == 'finished-signal' and signal_status == "FINISHED":
        return html.Div([
            html.H5("Results: Normalised flux vs JD", className="fw-bold mb-0"),
            dbc.Badge([
                html.I(className="bi bi-check-all me-2"),
                "FINISHED"
            ], color="success", className="ms-2")
        ], className="d-flex align-items-center mb-3")

    return dash.no_update


def create_gp_plot(gp_res, jd_max_guess=None):
    fig = go.Figure()

    # region extract data from gp_res
    gp = gp_res['gp']
    x = gp.X_train_.ravel()
    y = gp.y_train_
    noise_sigma_norm = gp_res['noise_sigma_norm']
    x_grid = gp_res['jd_grid'].ravel()
    y_mean = gp_res['mean_grid'].ravel()
    y_std = gp_res['std_grid'].ravel()

    jd_peak = gp_res['jd_peak']
    jd_peak_std = gp_res['jd_peak_std']
    peaks_jd = gp_res['peaks_jd']
    mean_peak = gp_res['mean_peak']
    n_samples_uncert = gp_res['n_samples_uncert']
    # endregion

    # 1. Data Points: Custom hover format
    fig.add_trace(go.Scatter(
        # region unfold me
        x=x, y=y,
        mode='markers',
        marker=dict(color='black', size=6),
        error_y=dict(type='data', array=np.full_like(y, noise_sigma_norm),
                     visible=True, thickness=1, width=2,
                     color='gray'),
        # Map the error to customdata so the template can see it
        customdata=np.full_like(y, noise_sigma_norm),
        # hovertemplate="<b>JD:</b> %{x:.3f}<br><b>norm flux:</b> %{y:.3f}<extra></extra>",
        hovertemplate="Data: %{y:.3f} ± %{customdata:.3f}<extra></extra>",
        name='Data'
        # endregion
    ))

    # 2. GP Mean: Custom hover format
    fig.add_trace(go.Scatter(
        x=x_grid, y=y_mean,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)', width=2),
        # Pass the grid of standard deviations to customdata
        customdata=y_std,
        hovertemplate="GP Mean: %{y:.3f} ± %{customdata:.3f}<extra></extra>",
        name='GP mean',
        # showlegend=False,
    ))

    # 3. GP Confidence Interval (±1σ)
    # hover info is hidden
    # Plotly trick: Create a continuous boundary by reversing the lower bound
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_grid, x_grid[::-1]]),
        y=np.concatenate([y_mean + y_std, (y_mean - y_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.25)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='GP ±1σ'
    ))

    # 4. Posterior-sampled maxima (Alpha blending)
    # hover info is hidden
    fig.add_trace(go.Scatter(
        x=peaks_jd,
        y=np.full_like(peaks_jd, 0.98 * mean_peak),
        mode='markers',
        marker=dict(color='orange', size=8, opacity=0.1),
        hoverinfo="skip",
        showlegend=False,
        # name=f'Posterior peak draws (n={n_samples_uncert})',
    ))

    # 5. Vertical Lines (Shapes/Vlines)
    # Peak Guess
    if jd_max_guess is not None:
        fig.add_vline(x=jd_max_guess, line_width=1.5, line_dash="dot",
                      line_color="green",
                      # annotation_text="Guess"
                      )

    # GP Peak and Sigma range
    fig.add_vline(x=float(jd_peak), line_width=2, line_dash="dash", line_color="magenta",
                  # annotation_text=f"GP peak: {jd_peak:.3f}"
                  )

    # 6. Sigma Range Shaded Area (Vertical Fill)
    fig.add_vrect(
        x0=float(jd_peak - jd_peak_std),
        x1=float(jd_peak + jd_peak_std),
        fillcolor="magenta", opacity=0.1,
        layer="below", line_width=0,
        name='±1σ range'
    )

    # Add the boundary dotted lines for the sigma range
    fig.add_vline(x=float(jd_peak - jd_peak_std), line_width=2, line_dash="dot", line_color="magenta")
    fig.add_vline(x=float(jd_peak + jd_peak_std), line_width=2, line_dash="dot", line_color="magenta")

    # 7. Layout and Labels
    fig.update_layout(
        # 1. Reduce margins to almost zero (top, bottom, left, right)
        margin=dict(l=0, r=10, t=20, b=20),
        showlegend=False,
        # 2. Place Legend INSIDE the plot area
        # legend=dict(
        #     yanchor="top",
        #     y=0.99,
        #     xanchor="right",
        #     x=0.99,
        #     bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent
        #     font=dict(size=10)  # Smaller font
        # ),
        # 3. Design tweaks
        title=dict(text=f'   Peak: {jd_peak:.3f}', font=dict(size=14), y=0.95),
        template='plotly_white',
        height=400,  # Shorter height to see more rows at once
        xaxis=dict(
            # This controls the BIG bold number at the top of the unified tooltip
            # '.3f' ensures 4 decimal places for the Julian Date
            hoverformat='.3f',
            # title='jd',
            tickfont=dict(size=10)
        ),
        hovermode='x unified',  # Show one tooltip for all traces.
        # Plotly tries to be helpful by stacking information from every single trace into one giant box
        # Seems, too helpful and too gigantic
        # 'x unified' is great, but we need to style the label
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.9)",
            font_size=12,
            font_family="Rockwell"
        )
    )

    return fig


@app.callback(
    # region unfold me
    Output('graphs-container', 'children', allow_duplicate=True),
    Output('finished-signal', 'children', allow_duplicate=True),  # Final signal
    Output('store-results-data', 'data'),  # this data will be downloaded by user
    Input('run-btn', 'n_clicks'),
    State('store-lc-data', 'data'),
    State('store-intervals-data', 'data'),
    State('guess-sigma', 'value'),
    State('extrema-mode', 'value'),
    State({'type': 'float-input', 'index': ALL}, 'id'),  # Get the IDs
    State({'type': 'float-input', 'index': ALL}, 'value'),  # Get the values
    background=True,
    cancel=[Input("cancel-btn", "n_clicks")],
    running=[
        (Output("run-btn", "disabled"), True, False),
        (Output("cancel-btn", "disabled"), False, True),
    ],
    progress=[Output("live-graphs-container", "children"),
              Output("finished-signal", "children")],  # Updates UI during execution
    # this is why we place set_progress between input arguments
    prevent_initial_call=True
    # endregion
)
def run_gp(set_progress, n_clicks, lc_json_string, intervals, guess_sigma, extrema_mode, ids, float_values):
    set_progress(([], "WAITING"))
    p = {val_id['index']: float(val) for val_id, val in zip(ids, float_values)}
    # Add a standalone guess_sigma
    p['guess_sigma'] = guess_sigma
    p['extrema_mode'] = extrema_mode

    # 1. Validation: Ensure files are loaded
    if DEBUG:
        from gp import FILENAME_IN, INTERVALS_FILE, load_intervals_from_file
        intervals = load_intervals_from_file(INTERVALS_FILE)
        df_lc = read_lc(FILENAME_IN)
        df_lc = add_flux(df_lc)
    else:
        if not lc_json_string or not intervals:
            error_alert = dbc.Alert("Please upload both lightcurve and intervals files.", color="warning")
            return error_alert, "FINISHED", None
            # return dbc.Alert("Please upload both lightcurve and intervals files.", color="warning")
        di = json.loads(lc_json_string)
        df_lc = pd.DataFrame(data=di['data'], columns=di['columns'])

    live_figs = []
    results_for_storage = []

    for i, piece in enumerate(intervals):
        jd_min, jd_max = piece[0], piece[-1]
        jd_max_guess = piece[1] if len(piece) > 2 else None

        if len(select_jd_interval(df_lc, jd_min, jd_max)) < LEN_MIN:
            continue

        try:
            # --- THE FRAGILE MAGIC ---
            # 1. Calculations
            gp_res = gp_peak_pipeline(df_lc, jd_min, jd_max, params=p)

            # 2. Build "Light Mode" Graph for Live View
            fig = create_gp_plot(gp_res, jd_max_guess=jd_max_guess)

            # Store data for the final phase
            results_for_storage.append({'jd_peak': gp_res["jd_peak"],
                                        'jd_peak_std': gp_res["jd_peak_std"],
                                        'figure': fig})
            # Extract kernel params
            optimized_kernel = gp_res['gp'].kernel_
            optimized_params = optimized_kernel.get_params()
            opt_l = optimized_params.get('k1__k2__length_scale', 0.0)
            opt_w = optimized_params.get('k2__noise_level', 0.0)

            # Define colours
            l_color = "danger" if (opt_l <= p['length_scale_min'] * 1.01 or
                                   opt_l >= p['length_scale_max'] * 0.99) else "info"
            w_color = "danger" if (opt_w <= p['white_noise_level_min'] * 1.01 or
                                   opt_w >= p['white_noise_level_max'] * 0.99) else "info"

            # Create the successful graph card
            item_to_append = dbc.Col(
                html.Div([
                    # Metadata Badge Row
                    html.Div([
                        dbc.Badge(f"Scale: {opt_l:.4f}", color=l_color, className="me-1"),
                        dbc.Badge(f"White Noise: {opt_w:.4f}", color=w_color, className="me-1"),
                        dbc.Badge(f"σ: {gp_res['jd_peak_std']:.4f}", color="secondary"),
                    ], style={"textAlign": "center", "marginBottom": "2px"}),
                    dcc.Graph(
                        figure=fig,
                        config={  # type: ignore
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d',
                                                       'select2d',
                                                       'zoomIn2d', 'zoomOut2d']
                        }
                    ),
                ], style={"border": "1px solid #eee", "padding": "5px", "borderRadius": "5px"}),
                width=6, className="px-1 mb-2"  # "px-1" reduces horizontal padding between columns
            )

        except Exception as e:
            # --- THE SAFETY NET ---
            logging.error(f"GP Failure at {jd_min}: {str(e)}")
            logging.error(traceback.format_exc())

            err_id = f"err-gp-{str(jd_min).replace('.', '')}"

            item_to_append = dbc.Col(
                html.Div([
                    dbc.Alert([
                        html.I(className="bi bi-exclamation-octagon me-2"),
                        html.B("GP Fit Failed"),
                        html.Div(f"Interval: {jd_min:.2f} - {jd_max:.2f}",
                                 style={"fontSize": "0.8rem"}),
                        html.Hr(),
                        html.Div("Hover for technical details", id=err_id,
                                 style={"fontSize": "0.7rem", "cursor": "help"})
                    ], color="danger", style={"height": "400px", "display": "flex",
                                              "flexDirection": "column", "justifyContent": "center",
                                              "textAlign": "center"}),
                    dbc.Tooltip(str(e), target=err_id)
                ], style={"padding": "5px"}),
                width=6, className="px-1 mb-2"
            )

            # Append whatever we created (Graph or Error Alert)

        live_figs.append(item_to_append)

        # 3. Update the Live UI immediately
        set_progress((live_figs, "WAITING"))

    # --- FINAL PHASE ---(Review) --------------
    # Now we build the "Review Mode" graphs with checkboxes
    review_figs = []
    numeric_data = []  # we do not want to store figures

    for i, res in enumerate(results_for_storage):
        # Build the Review UI
        review_figs.append(dbc.Col([
            html.Div([
                dbc.Checkbox(id={'type': 'fit-selector', 'index': i}, value=True, label="Keep"),
                dcc.Graph(figure=res['figure'],
                          config={  # type: ignore
                              'displaylogo': False,
                              'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'zoomIn2d', 'zoomOut2d']
                          })  # The saved figure objects
            ], className="p-2 border rounded")
        ], width=6))

        # Build the numeric storage (Exclude the 'figure' object!)
        numeric_data.append({
            'jd_peak': res['jd_peak'],
            'jd_peak_std': res['jd_peak_std']
        })

    return review_figs, "FINISHED", numeric_data


@app.callback(
    # region unfold me
    Output({'type': 'float-input', 'index': ALL}, 'value'),
    Output('guess-sigma', 'value'),
    Input('reset-btn', 'n_clicks'),
    State({'type': 'float-input', 'index': ALL}, 'id'),
    prevent_initial_call=True
    # endregion
)
def reset_params(n_clicks, ids):
    if n_clicks is None:
        return dash.no_update, dash.no_update

    # 1. Reset floats from the dictionary
    float_resets = [str(params_float[val_id['index']]) for val_id in ids]
    # Create a list of return values based on the 'index' stored in the ID
    # This pulls directly from your global 'params_float' dictionary
    # 2. Reset the boolean to your default constant
    return float_resets, GUESS_SIGMA


# ---- Download results logic

@app.callback(
    Output('export-intervals-filename', 'value'),
    Input('upload-lc', 'filename'),
    prevent_initial_call=True
)
def update_intervals_output_filename(filename):
    if filename:
        # Strip the old extension and add '_intervals.dat'
        base = filename.rsplit('.', 1)[0]
        return f"{base}_intervals.dat"
    return "results_intervals.dat"


# Build GP output filename
@app.callback(
    Output('export-filename', 'value'),
    Input('upload-lc', 'filename'),
    prevent_initial_call=True
)
def update_default_filename(filename):
    if filename:
        # Strip the old extension and add '_maxima.dat'
        base = filename.rsplit('.', 1)[0]
        return f"{base}_extrema.dat"
    return "results_extrema.dat"


@app.callback(
    # region unfold me
    Output("download-results", "data"),
    Input("save-file-btn", "n_clicks"),
    State("export-filename", "value"),
    State({'type': 'fit-selector', 'index': ALL}, 'value'),
    State('store-results-data', 'data'),
    State('extrema-mode', 'value'),
    prevent_initial_call=True
    # endregion
)
def trigger_download(n_clicks, filename_input, selection_mask, results, extrema_mode):
    print(f'------------- downloading {filename_input=} {selection_mask=} {results=} {extrema_mode=}')
    if not n_clicks or not results:
        return dash.no_update

    mode_label = "Minimum" if extrema_mode == 'min' else "Maximum"

    final_filename = filename_input if filename_input else f"gp_results_{extrema_mode}.dat"

    # 3. Build the content with mode-aware headers
    # We use \t for easy import into Excel/Topcat/Python
    lines = [
        f"# GP {mode_label} Results\n",
        f"# JD_{mode_label}\tJD_Std\n"
    ]

    # Filter by user checkboxes
    # Note: we use row['jd_peak'] by historical reasons
    for is_selected, row in zip(selection_mask, results):
        if is_selected:
            lines.append(f"{row['jd_peak']:.6f}\t{row['jd_peak_std']:.6f}\n")

    # Send as a downloadable text file
    return dcc.send_string("".join(lines), final_filename)


# -------------- Graphs with fits ---- two containers: Working and Final -----

@app.callback(
    # region unfold me
    Output('final-review-container', 'style'),
    Output('live-graphs-container', 'style'),
    Input('finished-signal', 'children'),  # this is a swithch-modes-trigger
    prevent_initial_call=True
    # endregion
)
def switch_modes(signal):
    print('--------------- switch_modes triggered by signal', signal)
    # helper callback to toggle the visibility working and review modes (once run_gp finishes).
    if signal == "FINISHED":
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'flex'}


@app.callback(
    # region unfold me
    Output({'type': 'fit-selector', 'index': ALL}, 'value'),
    Input('select-all-btn', 'n_clicks'),
    Input('unselect-all-btn', 'n_clicks'),
    State({'type': 'fit-selector', 'index': ALL}, 'value'),
    prevent_initial_call=True
    # endregion
)
def bulk_toggle_fits(select_clicks, unselect_clicks, current_values):
    # This is one of the magic Multi-Selection Callbacks
    # Check which button was actually pressed
    ctx = callback_context
    if not ctx.triggered:
        return current_values

    trigger_id = ctx.triggered[0]['prop_id']

    # We return a list of booleans the same length as the number of checkboxes
    if 'unselect-all-btn' in trigger_id:
        return [False] * len(current_values)
    else:
        return [True] * len(current_values)


if __name__ == '__main__':
    # Debug mode enabled as requested
    app.run(debug=True,
            port=8052,
            dev_tools_ui=True,  # will pop up a small blue circle in the bottom right of the browser.
            # If a callback fails, it turns red and you can click it to see the full Python
            # error without even looking at your IDE.
            dev_tools_props_check=True)
