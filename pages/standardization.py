from chart_style import finish_chart

import numpy as np
import plotly.graph_objects as go

from dash import (
    html, dcc, Input, Output, State, ctx, no_update
)
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. Load and Standardize Data
# ==========================================

iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)

feature_names = list(iris.feature_names)

colors = ["#24b6a4", "#eaa054", "#8e80e6"]

feature_options = [
    {"label": name, "value": i}
    for i, name in enumerate(feature_names)
]


# ==========================================
# 2. Create Scatter Plot
# ==========================================

def create_figure(data, x_idx, y_idx, title,
                  fixed_ranges=None, axis_mode="original"):

    fig = go.Figure()

    for i, species in enumerate(iris.target_names):
        mask = iris.target == i

        fig.add_trace(go.Scatter(
            x=data[mask, x_idx],
            y=data[mask, y_idx],
            mode="markers",
            name=species,
            marker=dict(
                size=8,
                color=colors[i],
                opacity=0.8
            ),
            customdata=np.where(mask)[0].reshape(-1, 1),
            hovertemplate=(
                "Sample %{customdata[0]}<br>"
                "X: %{x:.3f}<br>"
                "Y: %{y:.3f}<extra></extra>"
            )
        ))

    # Standardized values are dimensionless: do not label them in cm.
    if axis_mode == "standardized":
        x_label = f"{feature_names[x_idx]} — standardized (z-score)"
        y_label = f"{feature_names[y_idx]} — standardized (z-score)"
    elif axis_mode == "transition":
        x_label = f"{feature_names[x_idx]} — transitioning to z-score"
        y_label = f"{feature_names[y_idx]} — transitioning to z-score"
    else:
        x_label = feature_names[x_idx]
        y_label = feature_names[y_idx]

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=450,
        margin=dict(l=45, r=15, t=55, b=45),
        font=dict(family="Segoe UI, Arial"),
        legend=dict(orientation="h", y=-0.25),
        xaxis_title=x_label,
        yaxis_title=y_label
    )

    if fixed_ranges:
        fig.update_xaxes(range=fixed_ranges[0])
        fig.update_yaxes(range=fixed_ranges[1])

    return finish_chart(fig)
# ==========================================
# 3. Calculate Fixed Axis Ranges
# ==========================================

def get_fixed_ranges(x_idx, y_idx):

    all_x = np.concatenate([
        X[:, x_idx],
        X_scaled[:, x_idx]
    ])

    all_y = np.concatenate([
        X[:, y_idx],
        X_scaled[:, y_idx]
    ])

    def padded_range(values):
        low = float(values.min())
        high = float(values.max())
        padding = max((high - low) * 0.1, 0.3)
        return [low - padding, high + padding]

    return padded_range(all_x), padded_range(all_y)


def build_stats(progress, x_idx, y_idx):

    alpha = (progress or 0) / 100

    # Educational interpolation
    current = (
        (1 - alpha) * X + alpha * X_scaled
    )

    mean = current.mean(axis=0)
    std = current.std(axis=0, ddof=0)
    original_mean = X.mean(axis=0)
    original_std = X.std(axis=0, ddof=0)
    scaled_mean = X_scaled.mean(axis=0)
    scaled_std = X_scaled.std(axis=0, ddof=0)

    def format_stat(value):
        # Suppress negative zero caused by floating-point rounding.
        return "0.000" if abs(float(value)) < 0.0005 else f"{value:.3f}"

    def metric_card(label, value, note, value_id):
        return html.Div(
            className="stat-card",
            children=[
                html.Div(label, className="stat-label"),
                html.Div(format_stat(value), id=value_id, className="stat-value"),
                html.Div(note, className="section-description")
            ]
        )

    stats = html.Div([
        html.H3(f"Progress: {progress}%", id="std-live-progress"),
        html.P(
            "The points move smoothly from the original measurements "
            "toward their standardized z-scores."
        ),
        html.Div(
            className="stats-grid",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                "gap": "16px",
                "marginBottom": "24px"
            },
            children=[
            metric_card("Current X Mean", mean[x_idx], "Live transformation", "std-live-x-mean"),
            metric_card("Current X Std", std[x_idx], "Live transformation", "std-live-x-std"),
            metric_card("Current Y Mean", mean[y_idx], "Live transformation", "std-live-y-mean"),
            metric_card("Current Y Std", std[y_idx], "Live transformation", "std-live-y-std")
        ]),
        html.H4("Original vs Standardized Statistics"),
        html.Div(
            style={"overflowX": "auto"},
            children=html.Table(
                style={"width": "100%", "borderCollapse": "collapse", "textAlign": "left"},
                children=[
                    html.Thead(html.Tr([
                        html.Th("Feature", style={"padding": "10px"}),
                        html.Th("Original Mean", style={"padding": "10px"}),
                        html.Th("Original Std", style={"padding": "10px"}),
                        html.Th("Standardized Mean", style={"padding": "10px"}),
                        html.Th("Standardized Std", style={"padding": "10px"})
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(feature_names[idx], style={"padding": "10px"}),
                            html.Td(format_stat(original_mean[idx]), style={"padding": "10px"}),
                            html.Td(format_stat(original_std[idx]), style={"padding": "10px"}),
                            html.Td(format_stat(scaled_mean[idx]), style={"padding": "10px"}),
                            html.Td(format_stat(scaled_std[idx]), style={"padding": "10px"})
                        ]) for idx in dict.fromkeys([x_idx, y_idx])
                    ])
                ]
            )
        ),
        html.P(
            "Note: The animation interpolates between the original and "
            "standardized coordinates for learning purposes. "
            "Only the 100% frame represents fully standardized data."
        )
    ])

    return stats


# ==========================================
# 4. Page Layout
# ==========================================

def layout():

    return html.Div([
        dcc.Store(id='std-animation-data', data={
            'original': X.tolist(), 'scaled': X_scaled.tolist(),
            'targets': iris.target.tolist(), 'names': feature_names,
            'base': create_figure(X, 0, 1, 'Transformation Progress: 0%',
                                  get_fixed_ranges(0, 1)).to_plotly_json()
        }),

        html.H1(
            "Standardization Simulator",
            className="page-title"
        ),

        html.P(
            "Watch the transformation from original "
            "feature values to standardized values.",
            className="page-description"
        ),

        # Feature Selection
        html.Div(
            className="chart-card",
            children=[

                html.Div(
                    "Select Features",
                    className="section-title"
                ),

                html.Div(
                    className="controls-row",
                    children=[

                        html.Div([
                            html.Label("X-axis Feature"),
                            dcc.Dropdown(
                                id="std-x",
                                options=feature_options,
                                value=0,
                                clearable=False
                            )
                        ], style={"flex": "1", "minWidth": "220px"}),

                        html.Div([
                            html.Label("Y-axis Feature"),
                            dcc.Dropdown(
                                id="std-y",
                                options=feature_options,
                                value=1,
                                clearable=False
                            )
                        ], style={"flex": "1", "minWidth": "220px"})
                    ]
                )
            ]
        ),

        # Live Animation
        html.Div(
            className="chart-card",
            children=[

                html.Div(
                    "Live Standardization Animation",
                    className="section-title"
                ),

                html.P(
                    "Press Play or move the slider to "
                    "observe how the data changes.",
                    className="section-description"
                ),

                dcc.Graph(
                    id="std-graph",
                    config={
                        "displaylogo": False,
                        "responsive": True
                    }
                ),

                html.Div(
                    className="controls-row",
                    children=[
                        html.Button(
                            "▶ Play",
                            id="std-play",
                            className="control-button"
                        ),
                        html.Button(
                            "⏸ Pause",
                            id="std-pause",
                            className="control-button secondary"
                        ),
                        html.Button(
                            "↺ Reset",
                            id="std-reset",
                            className="control-button secondary"
                        )
                    ]
                ),

                html.Label("Transformation Progress"),

                dcc.Slider(
                    id="std-slider",
                    min=0,
                    max=100,
                    step=5,
                    value=0,
                    marks={
                        0: "Original",
                        50: "50%",
                        100: "Standardized"
                    }
                ),

                dcc.Interval(
                    id="std-interval",
                    interval=100,
                    disabled=True
                ),

                html.Div(
                    id="std-stats",
                    children=build_stats(0, 0, 1),
                    className="info-box",
                    style={"marginTop": "30px"}
                )
            ]
        ),

        # Before and After Comparison
        html.Div(
            className="chart-card",
            children=[

                html.Div(
                    "Before vs After Standardization",
                    className="section-title"
                ),

                html.P(
                    "Compare the original data with "
                    "the fully standardized data.",
                    className="section-description"
                ),

                html.Div(
                    children=[
                        dcc.Graph(id="std-before"),
                        dcc.Graph(id="std-after")
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns":
                            "repeat(auto-fit, minmax(300px, 1fr))",
                        "gap": "15px"
                    }
                )
            ]
        ),

        # Mathematical Explanation
        html.Div(
            className="info-box",
            children=[

                html.H3("How Standardization Works"),

                dcc.Markdown(
                    r"""
                    $$z = \frac{x-\mu}{\sigma}$$

                    **x:** Original value

                    **μ:** Mean of the feature

                    **σ:** Standard deviation

                    **z:** Standardized value
                    """,
                    mathjax=True
                ),

                html.P(
                    "Standardization subtracts the mean "
                    "and divides by the standard deviation "
                    "for each feature independently."
                ),

                html.P(
                    "After standardization, each feature "
                    "has mean approximately 0 and "
                    "population standard deviation 1. "
                    "Z-scores have no physical units (such as cm)."
                )
            ]
        )
    ])


# ==========================================
# 5. Register Callbacks
# ==========================================

def register_callbacks(app):

    # Animation controls
    # Keep the animation clock and slider in the browser, independent of server latency.
    app.clientside_callback(
        """function(play, pause, reset, progress) {
            const context = dash_clientside.callback_context;
            const t = context.triggered_id || ((context.triggered && context.triggered[0]) ? context.triggered[0].prop_id.split('.')[0] : null);
            if (t === 'std-play') return Number(progress) >= 100;
            if (t === 'std-pause' || t === 'std-reset') return true;
            if (t === 'std-slider' && Number(progress) >= 100) return true;
            return dash_clientside.no_update;
        }""",
        Output('std-interval', 'disabled'),
        Input('std-play', 'n_clicks'), Input('std-pause', 'n_clicks'),
        Input('std-reset', 'n_clicks'), Input('std-slider', 'value'),
        prevent_initial_call=True
    )

    app.clientside_callback(
        """function(ticks, reset, current) {
            if (dash_clientside.callback_context.triggered.some(x => x.prop_id === 'std-reset.n_clicks')) return 0;
            return Math.round(Math.min(100, (Number(current) || 0) + 5) * 100) / 100;
        }""",
        Output('std-slider', 'value'),
        Input('std-interval', 'n_intervals'), Input('std-reset', 'n_clicks'),
        State('std-slider', 'value'), prevent_initial_call=True
    )

    # Render the moving plot entirely in the browser. No network request per frame.
    app.clientside_callback(
        """function(progress, xIdx, yIdx, d) {
            if (!d) return dash_clientside.no_update;
            const a = Math.max(0, Math.min(1, Number(progress || 0) / 100));
            const fig = structuredClone(d.base);
            const points = [[], [], []];
            for (let i = 0; i < d.original.length; i++) {
                const k = d.targets[i];
                points[k].push([
                    (1-a)*d.original[i][xIdx] + a*d.scaled[i][xIdx],
                    (1-a)*d.original[i][yIdx] + a*d.scaled[i][yIdx]
                ]);
            }
            for (let k = 0; k < 3; k++) {
                fig.data[k].x = points[k].map(v => v[0]);
                fig.data[k].y = points[k].map(v => v[1]);
            }
            const axisLabel = (idx) => d.names[idx] + (a === 0 ? '' : a === 1 ? ' — standardized (z-score)' : ' — transitioning to z-score');
            fig.layout.title.text = `Transformation Progress: ${progress}%`;
            fig.layout.xaxis.title.text = axisLabel(xIdx);
            fig.layout.yaxis.title.text = axisLabel(yIdx);
            const bounds = (idx) => {
                const vals = d.original.map(r => r[idx]).concat(d.scaled.map(r => r[idx]));
                const lo = Math.min(...vals), hi = Math.max(...vals);
                const pad = Math.max((hi-lo)*0.1, 0.3);
                return [lo-pad, hi+pad];
            };
            fig.layout.xaxis.range = bounds(xIdx);
            fig.layout.yaxis.range = bounds(yIdx);
            fig.layout.transition = {duration: 110, easing: 'linear'};
            if (fig && fig.layout && !fig.layout.scene) {
                fig.layout.xaxis = Object.assign({}, fig.layout.xaxis || {}, {fixedrange:true});
                fig.layout.yaxis = Object.assign({}, fig.layout.yaxis || {}, {fixedrange:true});
                fig.layout.legend = Object.assign({}, fig.layout.legend || {}, {orientation:'h', x:0.5, xanchor:'center', y:-0.20, yanchor:'top'});
                fig.layout.margin = Object.assign({}, fig.layout.margin || {}, {b:110});
            }
            return fig;
        }""",
        Output('std-graph', 'figure'),
        Input('std-slider', 'value'), Input('std-x', 'value'),
        Input('std-y', 'value'), State('std-animation-data', 'data')
    )

    @app.callback(
        Output("std-before", "figure"),
        Output("std-after", "figure"),
        Input("std-x", "value"), Input("std-y", "value")
    )
    def update_comparison(x_idx, y_idx):
        return (create_figure(X, x_idx, y_idx, 'Before: Original Data'),
                create_figure(X_scaled, x_idx, y_idx, 'After: Standardized Data', axis_mode='standardized'))

    # Lightweight statistics update alongside each browser-side animation frame.
    app.clientside_callback(
        """function(progress, xIdx, yIdx, d) {
            if (!d) return Array(5).fill(dash_clientside.no_update);
            const a = Math.max(0, Math.min(1, Number(progress || 0)/100));
            const fmt = v => Math.abs(v) < 0.0005 ? '0.000' : v.toFixed(3);
            const metric = idx => {
                let sum = 0, sq = 0, n = d.original.length;
                for (let i=0; i<n; i++) {
                    const v = (1-a)*d.original[i][idx] + a*d.scaled[i][idx];
                    sum += v; sq += v*v;
                }
                const mean = sum/n;
                return [fmt(mean), fmt(Math.sqrt(Math.max(0, sq/n-mean*mean)))];
            };
            const x = metric(Number(xIdx)), y = metric(Number(yIdx));
            return [`Progress: ${progress}%`, x[0], x[1], y[0], y[1]];
        }""",
        Output('std-live-progress','children'),
        Output('std-live-x-mean','children'), Output('std-live-x-std','children'),
        Output('std-live-y-mean','children'), Output('std-live-y-std','children'),
        Input('std-slider','value'), Input('std-x','value'), Input('std-y','value'),
        State('std-animation-data','data')
    )
