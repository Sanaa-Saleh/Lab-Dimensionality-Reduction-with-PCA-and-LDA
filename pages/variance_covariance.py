from chart_style import finish_chart

import numpy as np
import plotly.graph_objects as go

from dash import html, dcc, Input, Output, State, ctx, no_update
from sklearn.datasets import load_iris


# ==========================================
# 1. Load Iris Dataset
# ==========================================

iris = load_iris()
X = iris.data.copy()

feature_names = list(iris.feature_names)
means = X.mean(axis=0)

colors = [
    "#24b6a4",
    "#eaa054",
    "#8e80e6"
]

feature_options = [
    {"label": name, "value": i}
    for i, name in enumerate(feature_names)
]

MIN_SPREAD = 0.5
MAX_SPREAD = 3.0
STEP = 0.05


# ==========================================
# 2. Transform Data
# ==========================================

def transform_data(spread):
    """
    Scale each feature's deviations from its mean.
    The original Iris dataset remains unchanged.
    """
    return means + spread * (X - means)


# ==========================================
# 3. Fixed Axis Ranges
# ==========================================

def get_fixed_ranges(x_idx, y_idx):

    maximum_data = transform_data(MAX_SPREAD)

    def axis_range(values, center):
        low = min(float(values.min()), float(center))
        high = max(float(values.max()), float(center))
        padding = max((high - low) * 0.08, 0.2)
        return [low - padding, high + padding]

    return (
        axis_range(maximum_data[:, x_idx], means[x_idx]),
        axis_range(maximum_data[:, y_idx], means[y_idx])
    )


# ==========================================
# 4. Create Scatter Plot
# ==========================================

def create_scatter(data, x_idx, y_idx, spread):

    fig = go.Figure()

    for i, species in enumerate(iris.target_names):

        mask = iris.target == i

        fig.add_trace(
            go.Scatter(
                x=data[mask, x_idx],
                y=data[mask, y_idx],
                mode="markers",
                name=species,
                marker=dict(
                    color=colors[i],
                    size=9,
                    opacity=0.85
                ),
                customdata=np.where(mask)[0].reshape(-1, 1),
                hovertemplate=(
                    "Sample %{customdata[0]}<br>"
                    "X: %{x:.3f}<br>"
                    "Y: %{y:.3f}<extra></extra>"
                )
            )
        )

    # Mean marker
    fig.add_trace(
        go.Scatter(
            x=[means[x_idx]],
            y=[means[y_idx]],
            mode="markers",
            name="Mean",
            marker=dict(
                color="#172554",
                size=16,
                symbol="x"
            )
        )
    )

    fig.add_hline(
        y=means[y_idx],
        line_dash="dash",
        line_color="#94a3b8"
    )

    fig.add_vline(
        x=means[x_idx],
        line_dash="dash",
        line_color="#94a3b8"
    )

    x_range, y_range = get_fixed_ranges(x_idx, y_idx)

    fig.update_layout(
        title=f"Data Spread: {spread:.2f}×",
        template="plotly_white",
        height=520,
        margin=dict(l=50, r=20, t=60, b=45),
        font=dict(family="Segoe UI, Arial"),
        legend_title="Iris Species",
        transition=dict(duration=180, easing="linear"),
        uirevision="variance-covariance"
    )

    fig.update_xaxes(
        title=feature_names[x_idx],
        range=x_range,
        autorange=False
    )

    fig.update_yaxes(
        title=feature_names[y_idx],
        range=y_range,
        autorange=False
    )

    return finish_chart(fig)
# ==========================================
# 5. Covariance Heatmap
# ==========================================

def create_covariance_heatmap(data):

    covariance = np.cov(data, rowvar=False)

    labels = [
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ]

    fig = go.Figure(
        go.Heatmap(
            z=covariance.tolist(),
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            text=np.round(covariance, 3).tolist(),
            texttemplate="%{text}",
            colorbar=dict(orientation="h", x=0.5, xanchor="center", y=-0.28, yanchor="top", len=0.7, thickness=13),
            hovertemplate=(
                "%{y} vs %{x}<br>"
                "Covariance: %{z:.4f}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Live Covariance Matrix",
        template="plotly_white",
        height=450,
        margin=dict(l=95, r=35, t=65, b=120)
    )

    return finish_chart(fig)
# ==========================================
# 6. Relationship Experiment
# ==========================================

def relationship_layout():

    return html.Div(
        className="chart-card",
        children=[

            html.Div(
                "Positive & Negative Covariance",
                className="section-title"
            ),

            html.P(
                "Move the slider to change the linear "
                "relationship between two variables.",
                className="section-description"
            ),

            html.Label("Relationship Strength"),

            dcc.Slider(
                id="vc-relationship",
                className="vc-relationship-slider",
                min=-1,
                max=1,
                step=0.05,
                value=-0.7,
                marks={
                    -1: {"label": "Negative", "style": {"transform": "translateX(15px)"}},
                    0: "Zero",
                    1: {"label": "Positive", "style": {"transform": "translateX(-45px)"}}
                },
                updatemode="drag"
            ),

            dcc.Graph(id="vc-relationship-graph", config={"displaylogo": False, "displayModeBar": False}
            ),

            html.Div(
                className="stats-grid",
                children=[

                    html.Div(
                        className="stat-card",
                        children=[
                            html.Div(
                                "Covariance",
                                className="stat-label"
                            ),
                            html.Div(
                                id="vc-relationship-cov",
                                className="stat-value"
                            )
                        ]
                    ),

                    html.Div(
                        className="stat-card",
                        children=[
                            html.Div(
                                "Correlation",
                                className="stat-label"
                            ),
                            html.Div(
                                id="vc-relationship-corr",
                                className="stat-value"
                            )
                        ]
                    )
                ]
            ),

            html.Div(
                id="vc-relationship-explanation",
                className="info-box"
            )
        ]
    )


# ==========================================
# 7. Main Page Layout
# ==========================================

def layout():

    return html.Div([
        dcc.Store(id='vc-animation-data', data={
            'original': X.tolist(), 'means': means.tolist(),
            'targets': iris.target.tolist(),
            'base': create_scatter(X, 0, 1, 1).to_plotly_json(),
            'heatmap_base': create_covariance_heatmap(X).to_plotly_json(),
            'names': feature_names,
            'ranges': [get_fixed_ranges(i, i)[0] for i in range(4)]
        }),

        html.H1(
            "Variance & Covariance Simulator",
            className="page-title"
        ),

        html.P(
            "Explore data spread, variance, covariance, "
            "and correlation interactively.",
            className="page-description"
        ),

        # Feature Controls
        html.Div(
            className="chart-card",
            children=[

                html.Div(
                    "Interactive Data Controls",
                    className="section-title"
                ),

                html.Div(
                    className="controls-row",
                    children=[

                        html.Div([
                            html.Label("X-axis Feature"),
                            dcc.Dropdown(
                                id="vc-x",
                                options=feature_options,
                                value=0,
                                clearable=False
                            )
                        ], style={
                            "flex": "1",
                            "minWidth": "200px"
                        }),

                        html.Div([
                            html.Label("Y-axis Feature"),
                            dcc.Dropdown(
                                id="vc-y",
                                options=feature_options,
                                value=1,
                                clearable=False
                            )
                        ], style={
                            "flex": "1",
                            "minWidth": "200px"
                        })
                    ]
                ),

                html.Div(
                    style={"marginTop": "25px"},
                    children=[

                        html.Label("Spread Multiplier"),

                        dcc.Slider(
                            id="vc-spread",
                            min=MIN_SPREAD,
                            max=MAX_SPREAD,
                            step=STEP,
                            value=1,
                            marks={
                                0.5: "0.5×",
                                1: "Original",
                                2: "2×",
                                3: "3×"
                            },
                            updatemode="drag"
                        ),
                        html.Div(
                            id="vc-spread-label",
                            style={
                                "marginTop": "24px",
                                "fontWeight": "600",
                                "color": "#172554"
                            }
                        )
                    ]
                ),

                html.Div(
                    className="controls-row",
                    style={"marginTop": "16px"},
                    children=[

                        html.Button(
                            "▶ Play",
                            id="vc-play",
                            className="control-button"
                        ),

                        html.Button(
                            "⏸ Pause",
                            id="vc-pause",
                            className="control-button secondary"
                        ),

                        html.Button(
                            "↺ Reset",
                            id="vc-reset",
                            className="control-button secondary"
                        )
                    ]
                ),

                dcc.Interval(
                    id="vc-interval",
                    interval=100,
                    n_intervals=0,
                    disabled=True
                )
            ]
        ),

        # Statistics
        html.Div(
            className="vc-kpi-grid",
            children=[

                html.Div(
                    className="stat-card",
                    children=[
                        html.Div(
                            "Variance (X)",
                            className="stat-label"
                        ),
                        html.Div(
                            id="vc-var-x",
                            className="stat-value"
                        )
                    ]
                ),

                html.Div(
                    className="stat-card",
                    children=[
                        html.Div(
                            "Variance (Y)",
                            className="stat-label"
                        ),
                        html.Div(
                            id="vc-var-y",
                            className="stat-value"
                        )
                    ]
                ),

                html.Div(
                    className="stat-card",
                    children=[
                        html.Div(
                            "Covariance (X,Y)",
                            className="stat-label"
                        ),
                        html.Div(
                            id="vc-cov",
                            className="stat-value"
                        )
                    ]
                ),
                html.Div(
                    className="stat-card",
                    children=[
                        html.Div("Correlation (X,Y)", className="stat-label"),
                        html.Div(id="vc-corr", className="stat-value")
                    ]
                )
            ],

        ),

        html.Div(
            id="vc-spread-explanation",
            className="info-box",
            style={"marginBottom": "20px"}
        ),

        # Live Scatter
        html.Div(
            className="chart-card",
            children=[

                html.Div(
                    "Live Data Distribution",
                    className="section-title"
                ),

                html.P(
                    "Watch the points move toward or away "
                    "from the mean.",
                    className="section-description"
                ),

                dcc.Graph(id="vc-scatter",
                    style={"height":"520px", "width":"100%"},
                    config={
                        "displaylogo": False,
                        "responsive": True, "displayModeBar": False
                    }
                )
            ]
        ),

        # Covariance Matrix
        html.Div(
            className="chart-card",
            children=[

                html.Div(
                    "Covariance Matrix",
                    className="section-title"
                ),

                dcc.Graph(id="vc-heatmap",
                    style={"height":"490px", "width":"100%"},
                    config={"displaylogo": False, "displayModeBar": False}
                )
            ]
        ),

        # New Relationship Experiment
        relationship_layout(),

        # Mathematical Explanation
        html.Div(
            className="info-box",
            children=[

                html.H3("Variance"),

                dcc.Markdown(
                    r"""
                    $$s^2 =
                    \frac{\sum_{i=1}^{n}
                    (x_i-\bar{x})^2}{n-1}$$

                    Variance measures how far values
                    spread around their mean.
                    """,
                    mathjax=True
                ),

                html.H3("Covariance"),

                dcc.Markdown(
                    r"""
                    $$\operatorname{Cov}(X,Y)=
                    \frac{\sum_{i=1}^{n}
                    (x_i-\bar{x})(y_i-\bar{y})}{n-1}$$

                    - **Positive:** Variables tend to
                      increase together.
                    - **Negative:** One tends to decrease
                      as the other increases.
                    - **Near zero:** Little linear relationship.
                    """,
                    mathjax=True
                ),

                html.H3("Correlation"),

                dcc.Markdown(
                    r"""
                    $$r =
                    \frac{\operatorname{Cov}(X,Y)}
                    {s_X s_Y}$$

                    Correlation ranges from -1 to +1.
                    """,
                    mathjax=True
                )
            ]
        )
    ])


# ==========================================
# 8. Register All Callbacks
# ==========================================

def register_callbacks(app):

    # --------------------------------------
    # Play / Pause / Reset
    # --------------------------------------

    # Keep the animation clock and slider in the browser, independent of server latency.
    app.clientside_callback(
        """function(play, pause, reset, progress) {
            const context = dash_clientside.callback_context;
            const t = context.triggered_id || ((context.triggered && context.triggered[0]) ? context.triggered[0].prop_id.split('.')[0] : null);
            if (t === 'vc-play') return Number(progress) >= 3;
            if (t === 'vc-pause' || t === 'vc-reset') return true;
            if (t === 'vc-spread' && Number(progress) >= 3) return true;
            return dash_clientside.no_update;
        }""",
        Output('vc-interval', 'disabled'),
        Input('vc-play', 'n_clicks'), Input('vc-pause', 'n_clicks'),
        Input('vc-reset', 'n_clicks'), Input('vc-spread', 'value'),
        prevent_initial_call=True
    )

    app.clientside_callback(
        """function(ticks, reset, current) {
            if (dash_clientside.callback_context.triggered.some(x => x.prop_id === 'vc-reset.n_clicks')) return 1;
            return Math.round(Math.min(3, (Number(current) || 1) + 0.1) * 100) / 100;
        }""",
        Output('vc-spread', 'value'),
        Input('vc-interval', 'n_intervals'), Input('vc-reset', 'n_clicks'),
        State('vc-spread', 'value'), prevent_initial_call=True
    )

    app.clientside_callback(
        """function(xIdx, yIdx, spread, d) {
            if (!d) return dash_clientside.no_update;
            const fig = structuredClone(d.base);
            const groups = [[], [], []];
            const scale = Number(spread || 1);
            for (let i = 0; i < d.original.length; i++) {
                const r = d.original[i], k = d.targets[i];
                groups[k].push([d.means[xIdx]+scale*(r[xIdx]-d.means[xIdx]),
                                d.means[yIdx]+scale*(r[yIdx]-d.means[yIdx])]);
            }
            for (let k = 0; k < 3; k++) {
                fig.data[k].x = groups[k].map(v => v[0]);
                fig.data[k].y = groups[k].map(v => v[1]);
            }
            fig.data[3].x = [d.means[xIdx]];
            fig.data[3].y = [d.means[yIdx]];
            fig.layout.title.text = `Data Spread: ${scale.toFixed(2)}×`;
            fig.layout.xaxis.title.text = d.names[xIdx];
            fig.layout.yaxis.title.text = d.names[yIdx];
            fig.layout.xaxis.range = d.ranges[xIdx];
            fig.layout.yaxis.range = d.ranges[yIdx];
            fig.layout.shapes = (fig.layout.shapes || []).map(sh => {
                const copy = {...sh};
                if (copy.type === 'line' && copy.y0 === copy.y1) copy.y0 = copy.y1 = d.means[yIdx];
                if (copy.type === 'line' && copy.x0 === copy.x1) copy.x0 = copy.x1 = d.means[xIdx];
                return copy;
            });
            fig.layout.transition = {duration: 110, easing: 'linear'};
            if (fig && fig.layout && !fig.layout.scene) {
                fig.layout.xaxis = Object.assign({}, fig.layout.xaxis || {}, {fixedrange:true});
                fig.layout.yaxis = Object.assign({}, fig.layout.yaxis || {}, {fixedrange:true});
                fig.layout.legend = Object.assign({}, fig.layout.legend || {}, {orientation:'h', x:0.5, xanchor:'center', y:-0.20, yanchor:'top'});
                fig.layout.margin = Object.assign({}, fig.layout.margin || {}, {b:110});
            }
            return fig;
        }""",
        Output('vc-scatter', 'figure'),
        Input('vc-x', 'value'), Input('vc-y', 'value'), Input('vc-spread', 'value'),
        State('vc-animation-data', 'data')
    )

    # Values and explanatory text update locally on every animation frame.
    app.clientside_callback(
        """function(xIdx, yIdx, spread, d) {
            if (!d) return Array(6).fill(dash_clientside.no_update);
            const x=Number(xIdx), y=Number(yIdx), scale=Number(spread || 1);
            const n=d.original.length;
            let vx=0, vy=0, cv=0;
            const mx=d.means[x], my=d.means[y];
            for (const row of d.original) {
                const dx=row[x]-mx, dy=row[y]-my;
                vx+=dx*dx; vy+=dy*dy; cv+=dx*dy;
            }
            vx=vx/(n-1)*scale*scale;
            vy=vy/(n-1)*scale*scale;
            cv=cv/(n-1)*scale*scale;
            const corr=(vx>0 && vy>0) ? cv/Math.sqrt(vx*vy) : 0;
            const factor=scale*scale;
            const explanation=Math.abs(scale-1)<1e-9
              ? 'Original data (1.00×): variance and covariance are at their baseline values. Correlation describes the strength and direction of the linear relationship.'
              : `At ${scale.toFixed(2)}× spread, both variances and covariance are multiplied by ${factor.toFixed(2)} (spread²). Their magnitudes are ${factor.toFixed(2)}× the original values (${factor>1?'larger':'smaller'} in magnitude when compared with 1×). Correlation stays unchanged because both axes are scaled by the same positive factor.`;
            return [vx.toFixed(3),vy.toFixed(3),cv.toFixed(3),corr.toFixed(3),`Current spread: ${scale.toFixed(2)}×`,explanation];
        }""",
        Output('vc-var-x','children'), Output('vc-var-y','children'),
        Output('vc-cov','children'), Output('vc-corr','children'),
        Output('vc-spread-label','children'), Output('vc-spread-explanation','children'),
        Input('vc-x','value'), Input('vc-y','value'), Input('vc-spread','value'),
        State('vc-animation-data','data')
    )

    # The covariance matrix is updated in the browser, in sync with the scatter.
    app.clientside_callback(
        """function(spread, d) {
            if (!d) return dash_clientside.no_update;
            const factor=Math.pow(Number(spread || 1),2);
            const fig=structuredClone(d.heatmap_base);
            const base=fig.data[0].z;
            fig.data[0].z=base.map(row=>row.map(v=>v*factor));
            fig.data[0].text=fig.data[0].z.map(row=>row.map(v=>Number(v.toFixed(3))));
            if (fig && fig.layout && !fig.layout.scene) {
                fig.layout.xaxis = Object.assign({}, fig.layout.xaxis || {}, {fixedrange:true});
                fig.layout.yaxis = Object.assign({}, fig.layout.yaxis || {}, {fixedrange:true});
                fig.layout.legend = Object.assign({}, fig.layout.legend || {}, {orientation:'h', x:0.5, xanchor:'center', y:-0.20, yanchor:'top'});
                fig.layout.margin = Object.assign({}, fig.layout.margin || {}, {b:110});
            }
            return fig;
        }""",
        Output('vc-heatmap','figure'), Input('vc-spread','value'),
        State('vc-animation-data','data')
    )

    # --------------------------------------
    # Positive / Negative Relationship
    # --------------------------------------

    @app.callback(
        Output("vc-relationship-graph", "figure"),
        Output("vc-relationship-cov", "children"),
        Output("vc-relationship-corr", "children"),
        Output("vc-relationship-explanation", "children"),
        Input("vc-relationship", "value")
    )
    def update_relationship(strength):

        rng = np.random.default_rng(42)

        x = rng.normal(size=150)
        noise = rng.normal(size=150)

        # Normalize X
        x = x - x.mean()
        x = x / np.std(x, ddof=1)

        # Orthogonalize noise against X
        noise = noise - noise.mean()
        noise = noise - (
            np.dot(noise, x) / np.dot(x, x)
        ) * x

        noise = noise / np.std(noise, ddof=1)

        # Build controlled relationship
        y = (
            strength * x
            + np.sqrt(max(0, 1 - strength**2)) * noise
        )

        covariance = np.cov(x, y)[0, 1]
        correlation = np.corrcoef(x, y)[0, 1]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Samples",
                marker=dict(
                    color="#8e80e6",
                    size=9,
                    opacity=0.8
                ),
                hovertemplate=(
                    "X: %{x:.3f}<br>"
                    "Y: %{y:.3f}<extra></extra>"
                )
            )
        )

        fig.update_layout(
            title="Live Covariance Relationship",
            template="plotly_white",
            height=480,
            xaxis=dict(
                title="Variable X",
                range=[-4, 4],
                autorange=False
            ),
            yaxis=dict(
                title="Variable Y",
                range=[-4, 4],
                autorange=False
            ),
            margin=dict(l=45, r=20, t=60, b=45),
            transition=dict(
                duration=180,
                easing="linear"
            )
        )

        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        fig.update_layout(dragmode=False, uirevision="vc-relationship-fixed", transition=dict(duration=0))

        if strength > 0.05:
            explanation = (
                "Positive covariance: both variables "
                "tend to increase together."
            )
        elif strength < -0.05:
            explanation = (
                "Negative covariance: as X increases, "
                "Y tends to decrease."
            )
        else:
            explanation = (
                "Near-zero covariance: little linear "
                "relationship exists."
            )

        return (
            fig,
            f"{covariance:.3f}",
            f"{correlation:.3f}",
            explanation
        )
