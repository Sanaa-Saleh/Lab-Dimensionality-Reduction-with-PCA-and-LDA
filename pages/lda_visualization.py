"""Interactive LDA visualization page for Dimensionality Lab (Dash)."""
from chart_style import finish_chart
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target
lda = LinearDiscriminantAnalysis(n_components=2)
Z = lda.fit_transform(X, y)
ratios = lda.explained_variance_ratio_
COLORS = ["#24b6a4", "#eaa054", "#8e80e6"]
NAMES = list(iris.target_names)


def _figure(n_components, show_centroids):
    fig = go.Figure()
    for i, name in enumerate(NAMES):
        mask = y == i
        xvals = Z[mask, 0]
        if n_components == 1:
            # Fixed, deterministic display jitter: not a second LDA dimension.
            rng = np.random.default_rng(100 + i)
            yvals = rng.uniform(-0.15, 0.15, size=int(mask.sum()))
        else:
            yvals = Z[mask, 1]
        hover = (
            "Sample %{customdata}"
            "<br>LD1 score: %{x:.3f}"
            + ("<br>Display jitter: %{y:.3f} (visual only)" if n_components == 1
               else "<br>LD2 score: %{y:.3f}")
            + "<extra>" + name + "</extra>"
        )
        fig.add_trace(go.Scatter(
            x=xvals, y=yvals, mode="markers", name=name,
            marker=dict(color=COLORS[i], size=9, opacity=0.8),
            customdata=np.flatnonzero(mask) + 1,
            hovertemplate=hover,
        ))
        if show_centroids:
            fig.add_trace(go.Scatter(
                x=[Z[mask, 0].mean()],
                y=[0 if n_components == 1 else Z[mask, 1].mean()],
                mode="markers", name=f"{name} centroid",
                marker=dict(color=COLORS[i], size=17, symbol="x", line=dict(width=2)),
                showlegend=False,
                hovertemplate=(
                    f"{name} class centroid<br>Mean LD1: %{{x:.3f}}"
                    + ("<br>Display position: 0 (visual only)" if n_components == 1
                       else "<br>Mean LD2: %{y:.3f}")
                    + "<extra></extra>"
                ),
            ))
    if n_components == 1:
        fig.add_hline(y=0, line_color="#7755dd", line_width=2)
    fig.update_layout(
        title="LDA Projection — " + ("1D" if n_components == 1 else "2D"),
        template="plotly_white", height=520,
        margin=dict(l=50, r=30, t=65, b=50),
        xaxis_title="LD1 score",
        yaxis_title="Display jitter (not a feature)" if n_components == 1 else "LD2 score",
        legend_title=None,legend=dict( orientation='h',x=0.5,xanchor='center',y=-0.20,yanchor='top',font=dict(size=10),entrywidth=60,entrywidthmode='pixels'),uirevision="lda",
    )
    if n_components == 1:
        fig.update_yaxes(range=[-0.45, 0.45])
    return finish_chart(fig)
def _scatter_summary():
    overall = X.mean(axis=0)
    within = 0.0
    between = 0.0
    for cls in np.unique(y):
        Xi = X[y == cls]
        center = Xi.mean(axis=0)
        within += float(np.sum((Xi - center) ** 2))
        between += len(Xi) * float(np.sum((center - overall) ** 2))
    return within, between

WITHIN, BETWEEN = _scatter_summary()


def _scatter_figure():
    fig = go.Figure(go.Bar(
        x=["Within-class scatter", "Between-class scatter"],
        y=[WITHIN, BETWEEN], marker_color=["#eaa054", "#7755dd"],
        text=[f"{WITHIN:.1f}", f"{BETWEEN:.1f}"], textposition="outside",
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Scatter in Standardized Original Feature Space",
        template="plotly_white", height=360,
        margin=dict(l=45, r=20, t=65, b=65),
        yaxis_title="Sum of squared distances",
    )
    return finish_chart(fig)
def layout():
    return html.Div([
        html.H1("LDA Visualization", className="page-title"),
        html.P("Discover how supervised dimensionality reduction separates the three Iris classes.", className="page-description"),
        html.Div(className="stats-grid", children=[
            html.Div(className="stat-card", children=[html.Div("Original Features", className="stat-label"), html.Div("4", className="stat-value")]),
            html.Div(className="stat-card", children=[html.Div("Classes", className="stat-label"), html.Div("3", className="stat-value")]),
            html.Div(className="stat-card", children=[html.Div("Maximum LDA Dimensions", className="stat-label"), html.Div("2", className="stat-value")]),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("Interactive LDA Controls", className="section-title"),
            html.P("Choose one or two discriminant axes and inspect class separation.", className="section-description"),
            html.P("The × markers represent class centroids: the mean position of each Iris species in the selected LDA projection.", className="section-description"),
            html.Div(className="controls-row", children=[
                html.Div([
                    html.Label("Retained LDA Components"),
                    dcc.RadioItems(
                        id="lda-dimensions", options=[{"label": " 1D (LD1)", "value": 1}, {"label": " 2D (LD1 + LD2)", "value": 2}],
                        value=2, inline=True, inputStyle={"marginRight": "7px", "marginLeft": "15px"},
                    ),
                ], style={"flex": "1", "minWidth": "220px"}),
                html.Div([
                    dcc.Checklist(
                        id="lda-centroids", options=[{"label": " Show class centroids", "value": "yes"}],
                        value=["yes"], inputStyle={"marginRight": "8px"},
                    )
                ], style={"flex": "1", "minWidth": "220px"}),
            ]),
        ]),
        html.Div(className="stats-grid", children=[
            html.Div(className="stat-card", children=[html.Div("LD1 Discriminant Variance", className="stat-label"), html.Div(f"{ratios[0] * 100:.2f}%", className="stat-value")]),
            html.Div(className="stat-card", children=[html.Div("LD2 Discriminant Variance", className="stat-label"), html.Div(f"{ratios[1] * 100:.2f}%", className="stat-value")]),
            html.Div(className="stat-card", children=[html.Div("Retained Discriminant Variance", className="stat-label"), html.Div(id="lda-retained", className="stat-value")]),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("Live Class Separation", className="section-title"),
            html.P(id="lda-plot-note", className="section-description"),
            html.Div(id="lda-dimension-explanation", className="info-box", style={"marginBottom": "14px"}),
            dcc.Graph(id="lda-scatter", config={"displaylogo": False, "displayModeBar": False}),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("Within-Class vs Between-Class Scatter", className="section-title"),
            html.P("These two values summarize the standardized original data; they do not change when you select 1D or 2D. LDA seeks directions that favor separation between classes relative to variation within classes.", className="section-description"),
            dcc.Graph(figure=_scatter_figure(), config={"displaylogo": False, "displayModeBar": False}),
        ]),
        html.Div(className="info-box", children=[
            html.H3("What is LDA optimizing?"),
            dcc.Markdown(r"LDA uses class labels and seeks directions that maximize class separation relative to within-class spread. A common expression is $J(w)=\frac{w^T S_B w}{w^T S_W w}$.", mathjax=True),
            html.P("With three Iris classes, LDA can produce at most 3 − 1 = 2 discriminant dimensions. Unlike PCA, the percentages here describe discriminant variance, not the fraction of total original feature variance retained."),
            html.P("Important: Discriminant variance is not classification accuracy. The displayed percentages describe the relative discriminant information carried by the LDA axes, not how often a classifier predicts correctly."),
            html.P("The scatter chart above reports traces of within-class and between-class scatter matrices in standardized feature space; the optimization itself uses their directional ratio."),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("lda-scatter", "figure"),
        Output("lda-retained", "children"),
        Output("lda-plot-note", "children"),
        Output("lda-dimension-explanation", "children"),
        Input("lda-dimensions", "value"),
        Input("lda-centroids", "value"),
    )
    def update_lda(n_components, centroids):
        n_components = 1 if n_components == 1 else 2
        retained = 100 * float(ratios[:n_components].sum())
        note = ("1D LDA: vertical jitter is only for readability; each sample has one LD1 score."
                if n_components == 1 else
                "2D LDA: both LD1 and LD2 are real discriminant coordinates.")
        show_centroids = "yes" in (centroids or [])
        if n_components == 1:
            heading = "Understanding 1D LDA"
            variance_text = f"LD1 preserves {retained:.2f}% of the discriminant variance. The remaining LD2 variance is not shown."
            visualization_text = "Each sample has one real coordinate (LD1). Vertical jitter is added only for readability; it is not another feature."
            centroid_text = (
                "The × markers represent the mean LD1 position of each Iris species."
                if show_centroids else "Class centroids are currently hidden. Enable the checkbox to display their mean LD1 positions."
            )
        else:
            heading = "Understanding 2D LDA"
            variance_text = f"LD1 and LD2 together preserve {retained:.2f}% of the discriminant variance."
            visualization_text = "Both axes represent real LDA coordinates, showing class separation in two discriminant dimensions."
            centroid_text = (
                "The × markers represent the mean position of each Iris species in the LD1–LD2 space."
                if show_centroids else "Class centroids are currently hidden. Enable the checkbox to display their mean LD1–LD2 positions."
            )
        explanation = html.Div([
            html.Div(heading, style={"fontWeight": "700", "marginBottom": "10px"}),
            html.P([html.Strong("Variance Retained: "), variance_text], style={"margin": "0 0 8px"}),
            html.P([html.Strong("Visualization: "), visualization_text], style={"margin": "0 0 8px"}),
            html.P([html.Strong("Class Centroids: "), centroid_text], style={"margin": "0"}),
        ])
        return _figure(n_components, "yes" in (centroids or [])), f"{retained:.2f}%", note, explanation
