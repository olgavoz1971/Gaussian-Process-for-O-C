import dash
import diskcache
import pandas as pd
import json
from dash import dcc, html, Input, Output, State, ALL, DiskcacheManager
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import callback_context


import traceback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import base64
import io
import numpy as np

from gp import (GUESS_SIGMA, LEN_MIN,
                read_lc, load_intervals, add_flux, select_jd_interval, gp_peak_pipeline,
                NOISE_SCALE_DIVISOR,
                LENGTH_SCALE_INIT, LENGTH_SCALE_MIN, LENGTH_SCALE_MAX,
                WHITE_NOISE_LEVEL_INIT, WHITE_NOISE_LEVEL_MIN, WHITE_NOISE_LEVEL_MAX)
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


# Callback for Lightcurve Upload
@app.callback(
    Output('store-lc-data', 'data'),
    Output('upload-lc-text', 'children'),  # Targets the text inside the box
    Input('upload-lc', 'contents'),
    State('upload-lc', 'filename'),  # Grabs the filename
    prevent_initial_call=True
)
def upload_lc(contents, filename):
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


# Callback for Intervals Upload
@app.callback(
    # region fold me
    Output('store-intervals-data', 'data'),
    Output('upload-intervals-text', 'children'),
    Input('upload-intervals', 'contents'),
    State('upload-intervals', 'filename'),
    prevent_initial_call=True
    # endregion
    # Output('store-intervals-data', 'data'),
    # Input('intervals', 'contents'),
    # prevent_initial_call=True
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

        # --- Success UI ---
        return intervals_list, html.Div([
            html.I(className="bi bi-check-circle-fill me-2", style={"color": "#28a745"}),
            html.B(filename)
        ])
    except Exception as e:
        # 1. Log full traceback for terminal
        logging.error(f"Error processing interval file {filename}:")
        logging.error(traceback.format_exc())

        # 2. Terse UI for sidebar with hover details
        # We replace dots/spaces with hyphens for a valid HTML ID
        safe_id = f"err-int-{filename.replace('.', '-').replace(' ', '-')}"

        return dash.no_update, html.Div([
            html.I(className="bi bi-exclamation-octagon-fill me-2", style={"color": "#dc3545"}),
            html.Span(
                [html.B("Error: "), filename],
                id=safe_id,
                style={"color": "#dc3545", "fontSize": "0.85rem", "cursor": "help"}
            ),
            dbc.Tooltip(
                f"Interval Load Error: {str(e)}",
                target=safe_id,
                placement="right",
                style={"fontSize": "0.75rem"}
            )
        ])

    # content_type, content_string = contents.split(',')
    # decoded = base64.b64decode(content_string)
    #
    # # convert bytes -> text -> StringIO
    # text = decoded.decode('utf-8')
    # intervals_list = load_intervals(io.StringIO(text))
    # return intervals_list


from dash import callback_context


@app.callback(
    Output({'type': 'float-input', 'index': ALL}, 'value'),
    Output('guess-sigma', 'value'),
    Input('reset-btn', 'n_clicks'),
    State({'type': 'float-input', 'index': ALL}, 'id'),
    prevent_initial_call=True
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
    # return [str(params_float[val_id['index']]) for val_id in ids]


def create_float_input(label, default_val, tooltip_text, index, step=1.0):
    """Helper to create labeled float inputs with tooltips."""
    return html.Div([
        dbc.Label(label, id=f"label-{index}"),
        dbc.Tooltip(tooltip_text, target=f"label-{index}"),
        dbc.Input(id={'type': 'float-input', 'index': index},
                  type="number", value=default_val, step=step),
    ], className="mb-2")


def create_parameter_triple(main_label, main_tooltip, prefix, defaults: dict, step=0.001):
    """
    Creates a grouped set of 3 inputs (Min, Init, Max) with a common header.
    - prefix: the base string for the dictionary keys (e.g., 'white_noise_level')
    - defaults: the params_float dictionary
    """
    print(f'Creating parameter-triple {prefix}')
    print(f'{defaults=}')
    return html.Div([
        # Main Category Label with Tooltip
        html.Div([
            html.B(main_label, id=f"triple-label-{prefix}", style={"cursor": "help"}),
            dbc.Tooltip(main_tooltip, target=f"triple-label-{prefix}"),
        ], className="mt-3 mb-1"),

        # Row of 3 inputs
        dbc.Row([
            # MIN
            dbc.Col([
                dbc.Input(id={'type': 'float-input', 'index': f"{prefix}_min"},
                          type="number", value=defaults[f"{prefix}_min"], step=step, size="sm"),
                html.Small("Min", className="text-muted d-block text-center")
            ], width=4, className="pe-1"),

            # INIT
            dbc.Col([
                dbc.Input(id={'type': 'float-input', 'index': f"{prefix}_init"},
                          type="number", value=defaults[f"{prefix}_init"], step=step, size="sm",
                          style={"border-color": "#007bff"}),  # Highlight initial guess
                html.Small("Init", className="text-muted d-block text-center")
            ], width=4, className="px-1"),

            # MAX
            dbc.Col([
                dbc.Input(id={'type': 'float-input', 'index': f"{prefix}_max"},
                          type="number", value=defaults[f"{prefix}_max"], step=step, size="sm"),
                html.Small("Max", className="text-muted d-block text-center")
            ], width=4, className="ps-1"),
        ], className="g-0")  # No gutters for maximum compactness
    ], className="mb-2 p-2 border-bottom")


def make_label(key: str) -> str:
    return key.replace("_", " ").capitalize()


# Sidebar Layout (Input Panel)
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


sidebar = html.Div([
    # html.H4("Control"),  # , className="display-6"),
    # html.Hr(),
    html.H6("Legend"),
    LegendItem("black", "Data Points", mode='circle'),
    LegendItem("rgb(31, 119, 180)", "GP Mean", mode='line'),
    LegendItem("rgba(31, 119, 180, 0.25)", "GP ±1σ Confidence", mode='line'),
    LegendItem("magenta", "Peak Estimate & 1σ Range", mode='dashed'),
    LegendItem("orange", "Posterior Draws", mode='circle'),
    LegendItem("green", "Guess", mode='dashed'),

    html.Hr(),
    # Buttons
    dbc.Stack([
        dbc.Button("Run GP", id="run-btn", color="primary", n_clicks=0),
        dbc.Button("Cancel", id="cancel-btn", color="danger", outline=True, n_clicks=0),
    ], direction="horizontal", gap=2),

    # File Selectors
    dbc.Label("lightcurve:"),
    dcc.Upload(id='upload-lc',
               children=html.Div(['Drag or ', html.A('Select')], id='upload-lc-text'),
               style={'border': '1px dashed', 'padding': '5px', 'textAlign': 'center'}),

    dbc.Label("intervals:", className="mt-3"),
    dcc.Upload(id='upload-intervals',
               children=html.Div(['Drag or ', html.A('Select')], id='upload-intervals-text'),
               style={'border': '1px dashed', 'padding': '5px', 'textAlign': 'center'}),

    html.Hr(),
    dbc.Button("Reset Defaults", id="reset-btn", color="secondary", outline=True, size="sm", className="mt-3"),

    # Unpaired parameter:
    create_float_input(
        label="Noise Divisor",
        default_val=params_float["noise_scale_divisor"],
        tooltip_text="Scaling factor for assumed errors.",
        index="noise_scale_divisor"
    ),
    # 2. White Noise Triple
    create_parameter_triple(
        main_label="White Kernel Bounds",
        main_tooltip="Bounds and initial guess for the WhiteNoise kernel level.",
        prefix="white_noise_level",
        defaults=params_float,
        step=0.001
    ),
    # 3. Length Scale Triple (Assuming you add 'length_scale_min/max' to your dict)
    create_parameter_triple(
        main_label="RBF Length Scale",
        main_tooltip="Smoothness control. Small = wiggly, Large = smooth.",
        prefix="length_scale",
        defaults=params_float
    ),

    # 7 Float Entries in a Grid (2 columns)
    # dbc.Row([
    #     dbc.Col(create_float_input(label="Noise divisor", step=0.1,
    #                                default_val=params_float["noise_scale_divisor"],
    #                                tooltip_text="Scaling factor applied to estimated noise when GUESS_SIGMA=True. "
    #                                             "Larger value - smaller assumed errors -> more wiggly fit. "
    #                                             "Smaller value - larger errors → smoother fit",
    #                                index="noise_scale_divisor"), width=6),
    #
    #     dbc.Col(create_float_input(label="Length scale", step=0.001,
    #                                default_val=params_float["length_scale_init"],
    #                                tooltip_text="This defines how quickly the model is allowed to vary with time. "
    #                                "It controls smoothness of the fit: smaller values -- more flexible (wiggly) model;"
    #                                "larger values -- smoother model. Reasonable first guess: "
    #                                "Estimate a typical width of a visible feature; "
    #                                "set length_scale_init ~ feature_width / 2",
    #                                index="length_scale_init"), width=6),
    #
    #     dbc.Col(create_float_input(label="Sampling factor", step=0.1,
    #                                default_val=params_float["sampling_scale_factor"],
    #                                tooltip_text="Controls Lower bound of the Length scale. "
    #                                "Lower bound control prevents GP from fitting structures smaller "
    #                                "than data resolution: if too small - model overfits noise."
    #                                "Lower scale bound = sampling_scale * SAMPLING_SCALE_FACTOR",
    #                                index="sampling_scale_factor"), width=6),
    #
    #     dbc.Col(create_float_input(label="Length factor", step=0.1,
    #                                default_val=params_float["length_scale_factor"],
    #                                tooltip_text="Length scale factor",
    #                                index="length_scale_factor"), width=6),
    #
    #     dbc.Col(create_float_input(label="White noise", step=0.001,
    #                                default_val=params_float["white_noise_level_init"],
    #                                tooltip_text="White noise level init",
    #                                index="white_noise_level_init"), width=6),
    #
    #     dbc.Col(create_float_input(label="White noise min", step=0.001,
    #                                default_val=params_float["white_noise_level_min"],
    #                                tooltip_text="White noise level min",
    #                                index="white_noise_level_min"), width=6),
    #
    #     dbc.Col(create_float_input(label="White noise max", step=0.1,
    #                                default_val=params_float["white_noise_level_max"],
    #                                tooltip_text="White noise level max",
    #                                index="white_noise_level_max"), width=6),
    # ]),

    # Checkbox with Tooltip
    html.Div([
        dbc.Checkbox(id="guess-sigma", value=GUESS_SIGMA, className="form-check-input"),
        dbc.Label("Guess sigma", html_for="guess-sigma", id="guess-sigma-label", className="ms-2"),
        dbc.Tooltip("Check this to ignore provided errors and estimate them from data scatter",
                    target="guess-sigma-label"),
    ], className="mt-3 mb-3"),

    # ], style={"padding": "2rem", "backgroundColor": "#f8f9fa", "height": "100vh"})
], style={"padding": "10px"})

# Output Panel (Graphs in a Grid)
content = html.Div([
    html.H4("Peaks: normalised flux vs. JD"),
    html.Hr(),
    # Grid for graphs - 2 per row
    dbc.Row(id='graphs-container')
], style={"padding": "2rem"})

# Main App Layout
app.layout = dbc.Container([

    dcc.Store(id='store-lc-data'),
    dcc.Store(id='store-intervals-data'),

    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col(content, width=9),
    ])
], fluid=True)

# --- Callbacks (Logic) ---
DEBUG = False


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
        # region fold me
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
    # region fold me
    Output('graphs-container', 'children', allow_duplicate=True),
    Input('run-btn', 'n_clicks'),
    State('store-lc-data', 'data'),
    State('store-intervals-data', 'data'),
    State('guess-sigma', 'value'),
    State({'type': 'float-input', 'index': ALL}, 'id'),   # Get the IDs
    State({'type': 'float-input', 'index': ALL}, 'value'),   # Get the values
    background=True,
    cancel=[Input("cancel-btn", "n_clicks")],
    running=[
        (Output("run-btn", "disabled"), True, False),
        (Output("cancel-btn", "disabled"), False, True),
    ],
    progress=[Output("graphs-container", "children")],  # Updates UI during execution
    prevent_initial_call=True
    # endregion
)
def run_gp(set_progress, n_clicks, lc_json_string, intervals, guess_sigma, ids, float_values):
    p = {val_id['index']: float(val) for val_id, val in zip(ids, float_values)}
    # Add a standalone guess_sigma
    p['guess_sigma'] = guess_sigma

    # 1. Validation: Ensure files are loaded
    if DEBUG:
        from gp import FILENAME_IN, INTERVALS_FILE, load_intervals_from_file
        intervals = load_intervals_from_file(INTERVALS_FILE)
        df_lc = read_lc(FILENAME_IN)
        df_lc = add_flux(df_lc)
    else:
        if not lc_json_string or not intervals:
            return dbc.Alert("Please upload both lightcurve and intervals files.", color="warning")
        di = json.loads(lc_json_string)
        df_lc = pd.DataFrame(data=di['data'], columns=di['columns'])

    figs = []
    i = 0
    with open('maxima_gp.dat', 'a') as f:
        for piece in intervals:
            jd_min, jd_max = piece[0], piece[-1]
            jd_max_guess = piece[1] if len(piece) > 2 else None
            # print(f'Start with {jd_min} : {jd_max} piece')

            if len(select_jd_interval(df_lc, jd_min, jd_max)) < LEN_MIN:
                continue

            i += 1
            gp_res = gp_peak_pipeline(
                df_lc,
                jd_min,
                jd_max,
                params=p,
            )

            f.write(f'GP peak = {gp_res["jd_peak"]:.6f}  std = {gp_res["jd_peak_std"]:.6f}\n')
            # x = gp_res["gp"].X_train_
            # y_norm = gp_res["gp"].y_train_
            # alpha = gp_res["gp"].alpha

            fig = create_gp_plot(gp_res, jd_max_guess=jd_max_guess)

            # Extract optimized values (formatted for brevity)
            optimized_kernel = gp_res['gp'].kernel_
            optimized_params = optimized_kernel.get_params()
            opt_l = optimized_params.get('k1__k2__length_scale', 0.0)
            opt_w = optimized_params.get('k2__noise_level', 0.0)

            l_color = "danger" if (opt_l <= p['length_scale_min'] * 1.01 or
                                   opt_l >= p['length_scale_max'] * 0.99) else "info"
            w_color = "danger" if (opt_w <= p['white_noise_level_min'] * 1.01 or
                                   opt_w >= p['white_noise_level_max'] * 0.99) else "info"

            new_graph = dbc.Col(
                html.Div([
                    # Metadata Badge Row
                    html.Div([
                        dbc.Badge(f"Scale: {opt_l:.4f}", color=l_color, className="me-1"),
                        dbc.Badge(f"White Noise: {opt_w:.4f}", color=w_color, className="me-1"),
                        dbc.Badge(f"σ: {gp_res['jd_peak_std']:.4f}", color="secondary"),
                    ], style={"textAlign": "center", "marginBottom": "2px"}),

                    dcc.Graph(
                        figure=fig,
                        config={    # type: ignore
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d',
                                                       'select2d', 'zoomIn2d', 'zoomOut2d']
                            # 'displayModeBar': True,
                        },
                    ),
                ], style={"border": "1px solid #eee", "padding": "5px", "borderRadius": "5px"}),
                width=6,
                className="px-1 mb-2"   # "px-1" reduces horizontal padding between columns
            )
            # new_graph = dbc.Col(
            #     dcc.Graph(
            #         figure=fig,
            #         # ONLY leave 'Reset Axes' and 'Box Select'
            #         config={  # type: ignore
            #             'displaylogo': False,
            #             'modeBarButtonsToRemove': ['pan2d', 'lasso2d', ' autoScale2d',
            #                                        'select2d', 'zoomIn2d', 'zoomOut2d'],
            #             # 'displayModeBar': True,
            #         },
            #     ),
            #     width=6,
            #     className="px-1 mb-2"  # "px-1" reduces horizontal padding between columns
            # )
            figs.append(new_graph)

            # "Spit out" the current list of figures to the UI
            # This updates the 'progress' Output (graphs-container) immediately
            set_progress([figs])

    return figs


if __name__ == '__main__':
    # Debug mode enabled as requested
    app.run(debug=True,
            dev_tools_ui=True,  # will pop up a small blue circle in the bottom right of the browser.
            # If a callback fails, it turns red and you can click it to see the full Python
            # error without even looking at your IDE.
            dev_tools_props_check=True)
