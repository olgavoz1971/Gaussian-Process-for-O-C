import dash
import diskcache
import pandas as pd
import json
from dash import dcc, html, Input, Output, State, ALL, DiskcacheManager
import dash_bootstrap_components as dbc
# import plotly.express as px
import plotly.graph_objects as go

import base64
import io
import numpy as np

from gp import (GUESS_SIGMA, LEN_MIN,
                read_lc, load_intervals, add_flux, select_jd_interval, gp_peak_pipeline,
                NOISE_SCALE_DIVISOR, LENGTH_SCALE_INIT, SAMPLING_SCALE_FACTOR,
                WHITE_NOISE_LEVEL_INIT, WHITE_NOISE_LEVEL_MIN, WHITE_NOISE_LEVEL_MAX)

params_float = {
    "noise_scale_divisor": NOISE_SCALE_DIVISOR,
    "length_scale_init": LENGTH_SCALE_INIT,
    "sampling_scale_factor": SAMPLING_SCALE_FACTOR,
    "length_scale_factor": SAMPLING_SCALE_FACTOR,
    "white_noise_level_init": WHITE_NOISE_LEVEL_INIT,
    "white_noise_level_min": WHITE_NOISE_LEVEL_MIN,
    "white_noise_level_max": WHITE_NOISE_LEVEL_MAX
}
# Initialize Diskcache for background callbacks
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                background_callback_manager=background_callback_manager
                )


# Callback for Lightcurve Upload
@app.callback(
    Output('store-lc-data', 'data'),
    Input('upload-lc', 'contents'),
    prevent_initial_call=True
)
def upload_lc(contents):
    if contents is None:
        return dash.no_update

    # Decode the upload
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Use io.BytesIO to make the decoded bytes act like a file
    # Then pass it to your backend function
    df = read_lc(io.BytesIO(decoded))
    df = add_flux(df)

    # Serialise to JSON for dcc.Store
    lc = df.to_dict(orient='split', index=False)
    return json.dumps(lc)
    # return df.to_json(date_format='iso', orient='split')


# Callback for Intervals Upload
@app.callback(
    Output('store-intervals-data', 'data'),
    Input('intervals', 'contents'),
    prevent_initial_call=True
)
def upload_intervals(contents):
    if contents is None:
        return dash.no_update

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # convert bytes -> text -> StringIO
    text = decoded.decode('utf-8')
    intervals_list = load_intervals(io.StringIO(text))
    # print(f'{intervals_list=}')
    # Serialise the list of tuples
    return intervals_list


def create_float_input(label, default_val, tooltip_text, index):
    """Helper to create labeled float inputs with tooltips."""
    return html.Div([
        dbc.Label(label, id=f"label-{index}"),
        dbc.Tooltip(tooltip_text, target=f"label-{index}"),
        dbc.Input(id={'type': 'float-input', 'index': index},
                  type="number", value=default_val),
    ], className="mb-2")


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
    dbc.Label("lightcurve"),
    dcc.Upload(id='upload-lc', children=html.Div(['Drag or ', html.A('Select')]),
               style={'border': '1px dashed', 'padding': '10px', 'textAlign': 'center'}),

    dbc.Label("intervals", className="mt-3"),
    dcc.Upload(id='intervals', children=html.Div(['Drag or ', html.A('Select')]),
               style={'border': '1px dashed', 'padding': '10px', 'textAlign': 'center'}),

    html.Hr(),

    # 7 Float Entries in a Grid (2 columns)
    dbc.Row([
        dbc.Col(create_float_input(label="Noise divisor",
                                   default_val=params_float["noise_scale_divisor"],
                                   tooltip_text="Scaling factor applied to estimated noise when GUESS_SIGMA=True. "
                                                "Larger value - smaller assumed errors -> more wiggly fit. "
                                                "Smaller value - larger errors → smoother fit",
                                   index="noise_scale_divisor"), width=6),

        dbc.Col(create_float_input("Length scale",
                                   params_float["length_scale_init"],
                                   "This defines how quickly the model is allowed to vary with time. "
                                   "It controls smoothness of the fit: smaller values -- more flexible (wiggly) model;"
                                   "larger values -- smoother model. Reasonable first guess: "
                                   "Estimate a typical width of a visible feature; "
                                   "set length_scale_init ~ feature_width / 2",
                                   "length_scale_init"), width=6),

        dbc.Col(create_float_input("Sampling factor",
                                   params_float["sampling_scale_factor"],
                                   "Controls Lower bound of the Length scale. "
                                   "Lower bound control prevents GP from fitting structures smaller "
                                   "than data resolution: if too small - model overfits noise."
                                   "Lower scale bound = sampling_scale * SAMPLING_SCALE_FACTOR",
                                   "sampling_scale_factor"), width=6),

        dbc.Col(create_float_input("Length factor",
                                   params_float["length_scale_factor"],
                                   "Length scale factor",
                                   "length_scale_factor"), width=6),

        dbc.Col(create_float_input("White noise",
                                   params_float["white_noise_level_init"],
                                   "White noise level init",
                                   "white_noise_level_init"), width=6),

        dbc.Col(create_float_input("White noise min",
                                   params_float["white_noise_level_min"],
                                   "White noise level min",
                                   "white_noise_level_min"), width=6),

        dbc.Col(create_float_input("White noise max",
                                   params_float["white_noise_level_max"],
                                   "White noise level max",
                                   "white_noise_level_max"), width=6),
    ]),

    # Checkbox with Tooltip
    html.Div([
        dbc.Checkbox(id="guess-sigma", value=GUESS_SIGMA, className="form-check-input"),
        dbc.Label("Guess sigma", html_for="guess-sigma", id="guess-sigma-label", className="ms-2"),
        dbc.Tooltip("Check this to ignore provided errors and estimate them from data scatter",
                    target="guess-sigma-label"),
    ], className="mt-3 mb-3"),

], style={"padding": "2rem", "backgroundColor": "#f8f9fa", "height": "100vh"})

# Output Panel (Graphs in a Grid)
content = html.Div([
    html.H4("Peaks"),
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

    # 1. Data points with error bars
    gp = gp_res['gp']
    x = gp.X_train_.ravel()
    y = gp.y_train_
    noise_sigma_norm = gp_res['noise_sigma_norm']
    # print(f'{x=} {y=} {noise_sigma_norm=}')
    fig.add_trace(go.Scatter(
        # region fold me
        x=x, y=y,
        mode='markers',
        marker=dict(color='black', size=6),
        error_y=dict(type='data', array=np.full_like(y, noise_sigma_norm),
                     visible=True, thickness=1, width=2,
                     color='gray'),
        name='data (with estimated errors)'
        # endregion
    ))

    # 2. GP Mean Line
    x_grid = gp_res['jd_grid'].ravel()
    y_mean = gp_res['mean_grid'].ravel()
    y_std = gp_res['std_grid'].ravel()

    jd_peak = gp_res['jd_peak']
    jd_peak_std = gp_res['jd_peak_std']
    peaks_jd = gp_res['peaks_jd']
    mean_peak = gp_res['mean_peak']
    n_samples_uncert = gp_res['n_samples_uncert']

    fig.add_trace(go.Scatter(
        x=x_grid, y=y_mean,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)', width=2),
        name='GP mean',
        # showlegend=False,
    ))

    # 3. GP Confidence Interval (±1σ)
    # Plotly trick: Create a continuous boundary by reversing the lower bound
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_grid, x_grid[::-1]]),
        y=np.concatenate([y_mean + y_std, (y_mean - y_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.25)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        # showlegend=False,
        name='GP ±1σ'
    ))

    # 4. Posterior-sampled maxima (Alpha blending)
    fig.add_trace(go.Scatter(
        x=peaks_jd,
        y=np.full_like(peaks_jd, 0.98 * mean_peak),
        mode='markers',
        marker=dict(color='orange', size=8, opacity=0.1),
        name=f'Posterior peak draws (n={n_samples_uncert})',
        # showlegend=False,
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
        # margin=dict(l=40, r=10, t=40, b=40),
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
        title=dict(text=f'JD Peak: {jd_peak:.3f}', font=dict(size=14), y=0.95),
        template='plotly_white',
        height=400,  # Shorter height to see more rows at once
        hovermode='x unified'  # Saves space by showing one tooltip for all traces
    )

    return fig


@app.callback(
    # region fold me
    Output('graphs-container', 'children', allow_duplicate=True),
    Input('run-btn', 'n_clicks'),
    State('store-lc-data', 'data'),
    State('store-intervals-data', 'data'),
    State('guess-sigma', 'value'),
    State({'type': 'float-input', 'index': ALL}, 'value'),
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
def run_gp(set_progress, n_clicks, lc_json_string, intervals, guess_sigma, float_values):
    p = {
        "noise_scale_divisor": float_values[0],
        "length_scale_init": float_values[1],
        "sampling_scale_factor": float_values[2],
        "length_scale_factor": float_values[3],
        "white_noise_level_init": float_values[4],
        "white_noise_level_min": float_values[5],
        "white_noise_level_max": float_values[6],
        "guess_sigma": guess_sigma,
    }

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

            new_graph = dbc.Col(
                dcc.Graph(
                    figure=fig,
                    # ONLY leave 'Reset Axes' and 'Box Select'
                    config={  # type: ignore
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', ' autoScale2d',
                                                   'select2d',  'zoomIn2d', 'zoomOut2d'],
                        # 'displayModeBar': True,
                        },
                ),
                width=6,
                className="px-1 mb-2"  # "px-1" reduces horizontal padding between columns
            )
            figs.append(new_graph)

            # "Spit out" the current list of figures to the UI
            # This updates the 'progress' Output (graphs-container) immediately
            set_progress([figs])

    return figs


if __name__ == '__main__':
    # Debug mode enabled as requested
    app.run(debug=True)
