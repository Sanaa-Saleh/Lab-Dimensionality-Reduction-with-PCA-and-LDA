"""Interactive PCA explained variance page for Dimensionality Lab."""
from chart_style import finish_chart

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
scaled = StandardScaler().fit_transform(iris.data)
pca = PCA(n_components=scaled.shape[1]).fit(scaled)
ratios = pca.explained_variance_ratio_
cumulative = np.cumsum(ratios)
labels = [f"PC{i}" for i in range(1, len(ratios) + 1)]
PURPLE = "#7657d9"
TEAL = "#24b6a4"


def stat_card(label, element_id):
    return html.Div(className="stat-card", children=[
        html.Div(label, className="stat-label"),
        html.Div(id=element_id, className="stat-value"),
    ])


def layout():
    return html.Div([
        html.H1("Explained Variance", className="page-title"),
        html.P(
            "Discover how much information each principal component preserves, "
            "and choose how many components to keep.",
            className="page-description",
        ),
        html.Div(className="chart-card", children=[
            html.Div("Interactive PCA Controls", className="section-title"),
            html.P("PCA is fitted to all four standardized Iris features.",
                   className="section-description"),
            html.Div(className="controls-row", children=[
                html.Div(style={"flex": "1", "minWidth": "240px"}, children=[
                    html.Label("Target Explained Variance"),
                    dcc.Slider(
                        id="ev-threshold", min=50, max=100, step=1, value=95,
                        marks={50: "50%", 70: "70%", 80: "80%", 90: "90%",
                               95: "95%", 100: "100%"},
                        updatemode="drag",
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ]),
            ]),
            html.Div(
                id="ev-target-status",
                className="info-box",
                style={"marginTop": "26px", "lineHeight": "1.7"},
            ),
        ]),
        html.Div(className="stats-grid", children=[
            stat_card("Components Needed for Target", "ev-needed"),
            stat_card("Variance Retained (Automatic)", "ev-retained"),
            stat_card("Variance Lost (Automatic)", "ev-lost"),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("Explained Variance Ratio", className="section-title"),
            html.P("Each bar shows the variance explained by one component. "
                   "Purple bars are automatically retained to meet the target.",
                   className="section-description"),
            dcc.Graph(id="ev-bars", config={"displaylogo": False}),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("Cumulative Explained Variance", className="section-title"),
            html.P("The dashed horizontal line is your target. "
                   "The highlighted point is the smallest number of components that reaches it.",
                   className="section-description"),
            dcc.Graph(id="ev-cumulative", config={"displaylogo": False}),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("Component-by-Component Breakdown", className="section-title"),
            html.Div(id="ev-breakdown"),
        ]),
        html.Div(className="info-box", children=[
            html.H3("What do these numbers mean?"),
            dcc.Markdown(
                r"""
**Explained Variance Ratio:** the share of total variance captured by a component.

$$\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

**Cumulative Explained Variance:** the sum of the ratios from PC1 through PC*k*.

$$\text{Cumulative Ratio}_k = \sum_{i=1}^{k}\text{Ratio}_i$$

The threshold selects the **smallest** number of components whose cumulative
explained variance meets or exceeds the target. Retaining variance does not
necessarily preserve all information needed for every prediction task.
                """,
                mathjax=True,
            ),
            html.P(id="ev-insight"),
        ]),
    ])


def make_bar_chart(components):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=ratios * 100,
        marker_color=[PURPLE if i < components else "#cbd5e1" for i in range(4)],
        text=[f"{v:.2f}%" for v in ratios * 100],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", height=390,
        margin=dict(l=45, r=20, t=40, b=40),
        yaxis_title="Explained variance (%)", yaxis_range=[0, 100],
        font=dict(family="Segoe UI, Arial"),
    )
    return finish_chart(fig)
def make_cumulative_chart(threshold, needed):
    x = list(range(1, 5))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=cumulative * 100,
        mode="lines+markers+text", name="Cumulative variance",
        line=dict(color=PURPLE, width=3), marker=dict(size=11),
        text=[f"{v:.1f}%" for v in cumulative * 100],
        textposition=["top center", "top center", "top center", "top center"],
        hovertemplate="%{x} component(s): %{y:.2f}%<extra></extra>",
    ))
    # Place the target label inside the plot, away from the final 100% marker.
    fig.add_hline(y=threshold, line_dash="dash", line_color="#eaa054")
    fig.add_annotation(
        x=4.26, y=threshold, xref="x", yref="y",
        text=f"Target {threshold:.0f}%", showarrow=False,
        xanchor="right", yanchor="top" if threshold >= 98 else "bottom",
        yshift=-8 if threshold >= 98 else 8,
        font=dict(color="#b45309", size=11),
        bgcolor="rgba(255,255,255,0.86)",
    )
    fig.add_trace(go.Scatter(
        x=[needed], y=[float(cumulative[needed - 1] * 100)],
        mode="markers", name="Minimum for target",
        marker=dict(color=TEAL, size=17, symbol="diamond"),
        hovertemplate="Minimum: %{x} component(s)<br>%{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", height=410,
        margin=dict(l=55, r=30, t=45, b=45),
        xaxis=dict(title="Number of retained components", tickmode="array", tickvals=x,
                   range=[0.7, 4.3]),
        yaxis=dict(title="Cumulative variance (%)", range=[0, 112], autorange=False),
        font=dict(family="Segoe UI, Arial"),
        legend=dict(orientation="h", y=-0.3),
    )
    return finish_chart(fig)
def register_callbacks(app):
    @app.callback(
        Output("ev-bars", "figure"),
        Output("ev-cumulative", "figure"),
        Output("ev-needed", "children"),
        Output("ev-retained", "children"),
        Output("ev-lost", "children"),
        Output("ev-breakdown", "children"),
        Output("ev-insight", "children"),
        Output("ev-target-status", "children"),
        Input("ev-threshold", "value"),
    )
    def update_explained_variance(threshold):
        threshold = float(threshold)
        needed = int(np.searchsorted(cumulative, threshold / 100, side="left") + 1)
        needed = min(needed, 4)
        components = needed
        retained = float(cumulative[components - 1] * 100)
        lost = max(0.0, 100.0 - retained)
        header_style = {"textAlign": "left", "padding": "10px", "borderBottom": "1px solid #e2e8f0"}
        cell_style = {"padding": "10px", "borderBottom": "1px solid #e2e8f0"}
        table = html.Table(style={"width": "100%", "borderCollapse": "collapse"}, children=[
            html.Thead(html.Tr([
                html.Th("Component", style=header_style),
                html.Th("Individual", style=header_style),
                html.Th("Cumulative", style=header_style),
                html.Th("Retained?", style=header_style),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(labels[i], style=cell_style),
                    html.Td(f"{ratios[i] * 100:.2f}%", style=cell_style),
                    html.Td(f"{cumulative[i] * 100:.2f}%", style=cell_style),
                    html.Td("Yes" if i < components else "No", style=cell_style),
                ]) for i in range(4)
            ]),
        ])
        insight = (
            f"To reach at least {threshold:.0f}% explained variance, the minimum "
            f"{needed} component(s) are automatically retained, preserving "
            f"{retained:.2f}% variance and losing {lost:.2f}%."
        )
        status = html.Div([
            html.Strong("✓ Target achieved automatically"),
            html.Div(
                f"The {threshold:.0f}% target requires {needed} component(s), "
                f"retaining {retained:.2f}% variance."
            ),
        ])
        return (
            make_bar_chart(components),
            make_cumulative_chart(threshold, needed),
            str(needed), f"{retained:.2f}%", f"{lost:.2f}%", table, insight, status,
        )
