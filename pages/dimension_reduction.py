"""Interactive dimensionality reduction and inverse PCA reconstruction."""
from chart_style import finish_chart
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
model = PCA(n_components=4).fit(X)
scores = model.transform(X)
variance = model.explained_variance_ratio_
COLORS = ["#24b6a4", "#eaa054", "#8e80e6"]
FEATURES = ["Sepal length", "Sepal width", "Petal length", "Petal width"]


def reconstruct(k):
    """Inverse transform after retaining the first k PCA scores."""
    truncated = np.zeros_like(scores)
    truncated[:, :k] = scores[:, :k]
    return model.inverse_transform(truncated)


def card(label, identifier):
    return html.Div(className="stat-card", children=[
        html.Div(label, className="stat-label"),
        html.Div(id=identifier, className="stat-value"),
    ])


def layout():
    return html.Div([
        html.H1("Dimension Reduction", className="page-title"),
        html.P("Reduce four standardized Iris features to 1–4 principal components, "
               "then see what information can be reconstructed.", className="page-description"),
        html.Div(className="chart-card", children=[
            html.Div("Choose the Number of Components", className="section-title"),
            html.P("PCA is fitted once to all four standardized features. "
                   "Changing the slider retains only the first k components.",
                   className="section-description"),
            html.Label("Retained Principal Components"),
            # The value is displayed in our own label, without an extra numeric input.
            html.Div([
                dcc.Slider(id="dr-k", min=1, max=4, step=1, value=2,
                           marks={i: f"{i} PC{'s' if i > 1 else ''}" for i in range(1, 5)},
                           updatemode="drag", tooltip=None),
            ], className="dr-slider-wrapper"),
            html.P(id="dr-selected-components", className="section-description",
                   style={"marginTop": "18px", "marginBottom": "0"}),
        ]),
        html.Div(id="dr-learning-message", className="info-box",
                 style={"marginBottom": "20px"}),
        html.Div(className="stats-grid", children=[
            card("Dimensions: Before → After", "dr-dimensions"),
            card("Explained Variance Retained", "dr-variance"),
            card("Reconstruction RMSE", "dr-rmse"),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("Reduced Data Representation", className="section-title"),
            html.P(id="dr-plot-explanation", className="section-description"),
            dcc.Graph(id="dr-projection", config={"displaylogo": False, "responsive": True},
                      style={"width": "100%", "minHeight": "560px"}),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("Inverse Transform: Recovering the Features", className="section-title"),
            html.P("Choose an Iris sample and compare its original standardized values "
                   "with the approximation reconstructed from the retained components.",
                   className="section-description"),
            html.Div(className="controls-row", children=[
                html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                    html.Label("Sample"),
                    dcc.Dropdown(id="dr-sample", options=[
                        {"label": f"Sample {i + 1} — {iris.target_names[iris.target[i]]}",
                         "value": i} for i in range(len(X))], value=0, clearable=False),
                ]),
            ]),
            dcc.Graph(id="dr-reconstruction", config={"displaylogo": False}),
            html.P(id="dr-sample-error", className="section-description"),
        ]),
        html.Div(className="info-box", children=[
            html.H3("What does reconstruction error mean?"),
            dcc.Markdown(
                r"""PCA compresses the standardized data into **k** scores. The inverse
transform approximates the original four standardized features using only
those retained scores.

$$\hat{X}_k = Z_k W_k + \mu$$

**RMSE** measures the difference between all standardized original values
and their reconstructed values. Smaller is better; retaining all four
components gives (approximately) zero reconstruction error.

**Important:** Explained variance is not classification accuracy. A 1D
scatter may use vertical jitter for visibility only; jitter is not a
second PCA dimension. Four-dimensional data cannot be shown as four spatial axes in
one 2D/3D scatter plot. In the **4D view**, PC1, PC2 and PC3 form
the spatial axes while point color encodes PC4. The color bar shows the
PC4 score, not a species label.
""", mathjax=True),
        ]),
    ])


def make_projection(k):
    fig = go.Figure()
    if k == 1:
        # Vertical jitter is visual-only and never part of PCA scores.
        rng = np.random.default_rng(21)
        jitter = rng.uniform(-0.18, 0.18, len(X))
        for i, species in enumerate(iris.target_names):
            m = iris.target == i
            fig.add_trace(go.Scatter(x=scores[m, 0], y=jitter[m], mode="markers",
                                     name=species, marker=dict(color=COLORS[i], size=8, opacity=0.8),
                                     hovertemplate="PC1: %{x:.3f}<extra>" + species + "</extra>"))
        fig.update_yaxes(title="Display jitter (not a feature)", range=[-0.32, 0.32], zeroline=True)
        fig.update_xaxes(title="PC1 score")
        note = "1D PCA: each sample has one retained score. Vertical jitter only separates overlapping dots."
    elif k in (3, 4):
        for i, species in enumerate(iris.target_names):
            m = iris.target == i
            marker = (dict(color=scores[m, 3], coloraxis="coloraxis", size=5, opacity=0.85)
                      if k == 4 else dict(color=COLORS[i], size=5, opacity=0.85))
            fig.add_trace(go.Scatter3d(
                x=scores[m, 0], y=scores[m, 1], z=scores[m, 2],
                mode="markers", name=species, marker=marker,
                customdata=scores[m, 3],
                hovertemplate=("PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}"
                               + ("<br>PC4: %{customdata:.3f}" if k == 4 else "")
                               + f"<extra>{species}</extra>")))
        fig.update_layout(
            height=660,
            margin=dict(l=0, r=35, t=15, b=85),
            scene=dict(
                xaxis=dict(title="PC1", backgroundcolor="#ffffff", gridcolor="#e8eef8"),
                yaxis=dict(title="PC2", backgroundcolor="#ffffff", gridcolor="#e8eef8"),
                zaxis=dict(title="PC3", backgroundcolor="#ffffff", gridcolor="#e8eef8"),
                domain=dict(x=[0.02, 0.88], y=[0.08, 0.98]),
                aspectmode="data",
                camera=dict(eye=dict(x=1.45, y=1.55, z=1.1)),
            ),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12, yanchor="top"),
        )
        if k == 4:
            fig.update_layout(coloraxis=dict(
                colorscale="Viridis", cmin=float(scores[:, 3].min()),
                cmax=float(scores[:, 3].max()),
                colorbar=dict(title="PC4 score", orientation="h", len=0.65, thickness=14, x=0.5, xanchor="center", y=-0.15, yanchor="top"),
            ))
            note = ("4D PCA: PC1, PC2 and PC3 are the spatial axes, while point color "
                    "encodes PC4. Rotate the plot and use the color bar to read PC4 scores. "
                    "The legend identifies Iris species.")
        else:
            note = "3D PCA: rotate the plot to explore the first three retained components."
    else:
        for i, species in enumerate(iris.target_names):
            m = iris.target == i
            fig.add_trace(go.Scatter(x=scores[m, 0], y=scores[m, 1], mode="markers",
                                     name=species, marker=dict(color=COLORS[i], size=9, opacity=0.8),
                                     hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra>" + species + "</extra>"))
        fig.update_xaxes(title="PC1 score")
        fig.update_yaxes(title="PC2 score")
        note = "2D PCA: the displayed PC1 and PC2 coordinates are the complete reduced representation."
    fig.update_layout(template="plotly_white",
                      height=660 if k in (3, 4) else 510,
                      margin=dict(l=5, r=65 if k == 4 else 5, t=5, b=5) if k in (3, 4) else dict(l=40, r=25, t=35, b=45),
                      legend_title="Iris species",
                      font=dict(family="Segoe UI, Arial"), uirevision=f"dr-{k}")
    return finish_chart(fig), note
def make_reconstruction(k, sample, recovered):
    original = X[sample]
    approximation = recovered[sample]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Original (standardized)", x=FEATURES, y=original,
                         marker_color="#24b6a4"))
    fig.add_trace(go.Bar(name=f"Reconstructed ({k} PC{'s' if k > 1 else ''})",
                         x=FEATURES, y=approximation, marker_color="#7657d9"))
    fig.update_layout(template="plotly_white", barmode="group", height=430,
                      yaxis=dict(title="Standardized feature value", range=[-3.5, 3.5]),
                      margin=dict(l=50, r=20, t=35, b=100),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.3, yanchor="top"),
                      font=dict(family="Segoe UI, Arial"))
    return finish_chart(fig)
def register_callbacks(app):
    @app.callback(
        Output("dr-selected-components", "children"),
        Output("dr-learning-message", "children"),
        Output("dr-dimensions", "children"),
        Output("dr-variance", "children"),
        Output("dr-rmse", "children"),
        Output("dr-projection", "figure"),
        Output("dr-plot-explanation", "children"),
        Output("dr-reconstruction", "figure"),
        Output("dr-sample-error", "children"),
        Input("dr-k", "value"),
        Input("dr-sample", "value"),
    )
    def update_page(k, sample):
        k = int(k if k is not None else 2)
        sample = int(sample if sample is not None else 0)
        recovered = reconstruct(k)
        rmse = float(np.sqrt(np.mean((X - recovered) ** 2)))
        sample_rmse = float(np.sqrt(np.mean((X[sample] - recovered[sample]) ** 2)))
        fig, explanation = make_projection(k)
        explanations = {
            1: ("Keeping 1 component preserves the strongest direction of variation, "
                "but discards the most information. Reconstruction error is highest."),
            2: ("Keeping 2 components preserves 95.81% of the variance. "
                "Compared with 1 PC, reconstruction error decreases because PC2 restores additional information."),
            3: ("Keeping 3 components preserves 99.48% of the variance. "
                "PC3 restores more detail, so reconstruction RMSE decreases compared with 2 PCs."),
            4: ("All 4 components are retained: no dimensionality reduction occurs. "
                "Inverse PCA reconstructs the standardized features with approximately zero error."),
        }
        message = html.Div([
            html.Strong(f"What changes with {k} PC{'s' if k > 1 else ''}?"),
            html.P(explanations[k], style={"margin": "7px 0 0"}),
            html.P("More retained components cannot increase the overall PCA reconstruction RMSE "
                   "when the PCA model and data stay fixed.",
                   style={"margin": "6px 0 0"}),
        ])
        return (f"Selected: {k} principal component{'s' if k > 1 else ''}", message,
                f"4 → {k}", f"{100 * variance[:k].sum():.2f}%", f"{rmse:.4f}",
                fig, explanation, make_reconstruction(k, sample, recovered),
                f"Sample {sample + 1} reconstruction RMSE: {sample_rmse:.4f} (standardized units)")
