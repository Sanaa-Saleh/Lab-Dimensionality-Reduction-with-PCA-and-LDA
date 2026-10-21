import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Fixed baseline standardization ensures a fair comparison when an outlier is added.
iris = load_iris()
X_RAW = iris.data.copy()
Y = iris.target.copy()
SCALER = StandardScaler().fit(X_RAW)
X_BASE = SCALER.transform(X_RAW)
SPECIES = list(iris.target_names)
COLORS = ["#24b6a4", "#eaa054", "#8e80e6"]
PURPLE = "#7954d9"

BASE_PCA = PCA(n_components=2).fit(X_BASE)
BASE_LDA = LinearDiscriminantAnalysis(n_components=2).fit(X_BASE, Y)
BASE_PCA_SCORES = BASE_PCA.transform(X_BASE)
BASE_LDA_SCORES = BASE_LDA.transform(X_BASE)

# Keep the new observation in the Setosa class. Its direction is in standardized
# feature space, so its distance is comparable regardless of original units.
DIRECTION = np.array([1.0, -0.8, 1.0, -0.8])
DIRECTION = DIRECTION / np.linalg.norm(DIRECTION)
BASE_CLASS_CENTER = X_BASE[Y == 0].mean(axis=0)


def fit_experiment(enabled, distance):
    if not enabled:
        return X_BASE, Y, None
    point = BASE_CLASS_CENTER + float(distance) * DIRECTION
    data = np.vstack([X_BASE, point])
    labels = np.append(Y, 0)
    return data, labels, point


def orient_scores(scores, base_scores, model, base_model, kind):
    """Flip axes consistently so before/after charts are easier to compare."""
    scores = scores.copy()
    if kind == "PCA":
        for i in range(2):
            if np.dot(model.components_[i], base_model.components_[i]) < 0:
                scores[:, i] *= -1
    else:
        for i in range(2):
            if np.dot(model.scalings_[:, i], base_model.scalings_[:, i]) < 0:
                scores[:, i] *= -1
    return scores


def plot_projection(scores, labels, title, prefix, outlier_index=None, x_range=None, y_range=None, dimensions=2):
    fig = go.Figure()
    if dimensions == 1:
        rng = np.random.default_rng(112)
        display_y = rng.uniform(-0.14, 0.14, size=len(scores))
        if outlier_index is not None:
            display_y[outlier_index] = 0.0
    else:
        display_y = scores[:, 1]
    for class_id, name in enumerate(SPECIES):
        mask = labels == class_id
        if outlier_index is not None:
            mask[outlier_index] = False
        fig.add_trace(go.Scatter(
            x=scores[mask, 0], y=display_y[mask], mode="markers",
            name=name, legendgroup=name, marker=dict(size=8, color=COLORS[class_id], opacity=0.78),
            hovertemplate=(f"{name}<br>{prefix}1: %{{x:.3f}}" + ("<br>Display jitter: %{y:.3f} (visual only)" if dimensions == 1 else f"<br>{prefix}2: %{{y:.3f}}") + "<extra></extra>"),
        ))
    if outlier_index is not None:
        fig.add_trace(go.Scatter(
            x=[scores[outlier_index, 0]], y=[display_y[outlier_index]],
            mode="markers", name="Added outlier",
            marker=dict(size=18, color="#ed4d63", symbol="star", line=dict(width=2, color="#9f1239")),
            hovertemplate="Added outlier<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title=title, template="plotly_white", height=420,
        margin=dict(l=35, r=20, t=55, b=40), font=dict(family="Segoe UI, Arial"),
        legend_title="Iris species", xaxis_title=f"{prefix}1 score",
        yaxis_title="Display jitter (not a feature)" if dimensions == 1 else f"{prefix}2 score",
        uirevision="outlier-experiment",
    )
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if dimensions == 1:
        fig.add_hline(y=0, line_color=PURPLE, line_width=1.5)
        fig.update_yaxes(range=[-0.35, 0.35])
    elif y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


def comparison_ranges(before, after):
    stacked = np.vstack([before, after])
    low = stacked.min(axis=0)
    high = stacked.max(axis=0)
    padding = np.maximum((high - low) * 0.12, 0.4)
    return [low[0] - padding[0], high[0] + padding[0]], [low[1] - padding[1], high[1] + padding[1]]


def separation_ratio(scores, labels):
    """Between-class scatter / within-class scatter in the chosen projection."""
    center = scores.mean(axis=0)
    within = 0.0
    between = 0.0
    for class_id in np.unique(labels):
        group = scores[labels == class_id]
        group_center = group.mean(axis=0)
        within += float(np.sum((group - group_center) ** 2))
        between += len(group) * float(np.sum((group_center - center) ** 2))
    return between / within if within > 0 else float("nan")


def info_card(label, component_id):
    return html.Div(className="stat-card", children=[
        html.Div(label, className="stat-label"),
        html.Div(id=component_id, className="stat-value"),
    ])


def layout():
    return html.Div([
        html.H1("Outlier Experiment", className="page-title"),
        html.P(
            "Add one extreme Iris observation and watch PCA and LDA refit. "
            "The same baseline standardization is used throughout.",
            className="page-description",
        ),
        html.Div(className="chart-card", children=[
            html.Div("Interactive Outlier Controls", className="section-title"),
            html.P(
                "The red star is a synthetic Setosa observation. "
                "Move the slider to change its distance from the Setosa center.",
                className="section-description",
            ),
            html.Div(className="controls-row", children=[
                html.Div([
                    dcc.Checklist(
                        id="outlier-enabled",
                        options=[{"label": " Add outlier", "value": "on"}],
                        value=[],
                        style={"marginTop": "10px"},
                    ),
                ], style={"minWidth": "160px"}),
                html.Div([
                    html.Label("Outlier Distance (standardized units)"),
                    dcc.Slider(
                        id="outlier-distance", min=2, max=20, step=0.5, value=8,
                        marks={2: "2", 5: "5", 10: "10", 15: "15", 20: "20"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"flex": "1", "minWidth": "250px"}),
                html.Button("↺ Reset", id="outlier-reset", n_clicks=0, className="control-button"),
            ]),
            html.Div([
                html.Label("Projection View", style={"marginRight": "15px"}),
                dcc.RadioItems(
                    id="outlier-projection-view",
                    options=[{"label": " 1D (PC1 / LD1)", "value": 1},
                             {"label": " 2D (PC1 + PC2 / LD1 + LD2)", "value": 2}],
                    value=2, inline=True,
                    inputStyle={"marginRight": "7px", "marginLeft": "14px"},
                ),
            ], style={"marginTop": "18px"}),
        ]),
        html.Div(className="stats-grid", children=[
            info_card("Samples", "outlier-count"),
            info_card("PCA PC1 Variance Retained", "outlier-pca-variance"),
            info_card("PC1 Direction Change", "outlier-pca-angle"),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("PCA — Before vs After", className="section-title"),
            html.P(
                "PCA ignores class labels. Both panels share the same axis scales. "
                "Axes are refitted after adding the outlier.",
                className="section-description",
            ),
            html.Div(className="controls-row", children=[
                html.Div(dcc.Graph(id="outlier-pca-before"), style={"flex": "1", "minWidth": "350px"}),
                html.Div(dcc.Graph(id="outlier-pca-after"), style={"flex": "1", "minWidth": "350px"}),
            ]),
            html.Div(className="stats-grid", children=[
                info_card("PCA PC1 Variance — Before", "outlier-pca-before-variance"),
                info_card("PCA PC1 Variance — After", "outlier-pca-after-variance"),
                info_card("PC1 Variance Change", "outlier-pca-variance-change"),
            ]),
            html.P(
                "PC1 explained variance is the percentage of total feature variance represented by the first principal component. "
                "The change is measured in percentage points, not classification accuracy.",
                className="section-description",
            ),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("LDA — Before vs After", className="section-title"),
            html.P(
                "LDA uses class labels. The synthetic observation is labeled Setosa, "
                "so it can affect within-class scatter and the discriminant directions.",
                className="section-description",
            ),
            html.Div(className="controls-row", children=[
                html.Div(dcc.Graph(id="outlier-lda-before"), style={"flex": "1", "minWidth": "350px"}),
                html.Div(dcc.Graph(id="outlier-lda-after"), style={"flex": "1", "minWidth": "350px"}),
            ]),
            html.Div(className="stats-grid", children=[
                info_card("LDA Separation Ratio — Before", "outlier-lda-separation-before"),
                info_card("LDA Separation Ratio — After", "outlier-lda-separation-after"),
                info_card("Separation Change (%)", "outlier-lda-separation-change"),
            ]),
            html.P(
                "Separation ratio = between-class scatter / within-class scatter in the selected LDA projection. "
                "Higher values indicate stronger relative class separation; this is not classification accuracy.",
                className="section-description",
            ),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("How PCA Explained Variance Changes", className="section-title"),
            html.P(
                "The chart compares explained variance ratios of PC1 and PC2. "
                "A large outlier may make one direction dominate.",
                className="section-description",
            ),
            dcc.Graph(id="outlier-variance-chart"),
        ]),
        html.Div(id="outlier-understanding", className="info-box",
                 style={"marginBottom": "18px"}),
        html.Div(className="info-box", children=[
            html.H3("What is happening?"),
            html.P(
                "An outlier is a point unusually far from most observations. "
                "PCA can rotate toward it because PCA maximizes overall variance."
            ),
            html.P(
                "LDA also can be sensitive to outliers. Its response depends on "
                "the outlier's assigned class and how it changes class means and scatter."
            ),
            html.P(
                "The PC1 direction change is the unsigned angle between the original "
                "and refitted PC1 axes (0° to 90°). PCA axis signs have no meaning."
            ),
            html.P(
                "This is an educational sensitivity experiment, not a test of "
                "classification accuracy. We keep the original scaler fixed so "
                "standardization does not hide the outlier's influence."
            ),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("outlier-enabled", "value"),
        Output("outlier-distance", "value"),
        Input("outlier-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_experiment(_clicks):
        return [], 8

    @app.callback(
        Output("outlier-count", "children"),
        Output("outlier-pca-variance", "children"),
        Output("outlier-pca-angle", "children"),
        Output("outlier-pca-before-variance", "children"),
        Output("outlier-pca-after-variance", "children"),
        Output("outlier-pca-variance-change", "children"),
        Output("outlier-pca-before", "figure"),
        Output("outlier-pca-after", "figure"),
        Output("outlier-lda-before", "figure"),
        Output("outlier-lda-after", "figure"),
        Output("outlier-variance-chart", "figure"),
        Output("outlier-lda-separation-before", "children"),
        Output("outlier-lda-separation-after", "children"),
        Output("outlier-lda-separation-change", "children"),
        Output("outlier-understanding", "children"),
        Input("outlier-enabled", "value"),
        Input("outlier-distance", "value"),
        Input("outlier-projection-view", "value"),
    )
    def update_experiment(enabled_values, distance, dimensions):
        dimensions = 1 if dimensions == 1 else 2
        enabled = "on" in (enabled_values or [])
        data, labels, point = fit_experiment(enabled, distance or 8)
        outlier_idx = len(data) - 1 if enabled else None

        pca = PCA(n_components=2).fit(data)
        lda = LinearDiscriminantAnalysis(n_components=2).fit(data, labels)

        pca_scores = orient_scores(
            pca.transform(data), BASE_PCA_SCORES, pca, BASE_PCA, "PCA"
        )
        lda_scores = orient_scores(
            lda.transform(data), BASE_LDA_SCORES, lda, BASE_LDA, "LDA"
        )

        # The reference observations are projected into the refitted axes for
        # the "after" plots. Before plots use the original fitted axes.
        pca_x, pca_y = comparison_ranges(BASE_PCA_SCORES, pca_scores)
        lda_x, lda_y = comparison_ranges(BASE_LDA_SCORES, lda_scores)

        dimension_suffix = "1D" if dimensions == 1 else "2D"
        pca_before = plot_projection(
            BASE_PCA_SCORES, Y, f"Original PCA — {dimension_suffix}", "PC", x_range=pca_x, y_range=pca_y, dimensions=dimensions
        )
        pca_after = plot_projection(
            pca_scores, labels, "PCA with Outlier" if enabled else "PCA (unchanged)",
            "PC", outlier_index=outlier_idx, x_range=pca_x, y_range=pca_y, dimensions=dimensions
        )
        lda_before = plot_projection(
            BASE_LDA_SCORES, Y, f"Original LDA — {dimension_suffix}", "LD", x_range=lda_x, y_range=lda_y, dimensions=dimensions
        )
        lda_after = plot_projection(
            lda_scores, labels, "LDA with Outlier" if enabled else "LDA (unchanged)",
            "LD", outlier_index=outlier_idx, x_range=lda_x, y_range=lda_y, dimensions=dimensions
        )

        cosine = np.clip(
            abs(float(np.dot(BASE_PCA.components_[0], pca.components_[0]))),
            0.0, 1.0
        )
        angle = np.degrees(np.arccos(cosine))

        variance_fig = go.Figure()
        names = ["PC1", "PC2"]
        variance_fig.add_trace(go.Bar(
            name="Before", x=names, y=BASE_PCA.explained_variance_ratio_ * 100,
            marker_color="#24b6a4", text=[f"{v:.2f}%" for v in BASE_PCA.explained_variance_ratio_ * 100],
            textposition="outside",
        ))
        variance_fig.add_trace(go.Bar(
            name="After", x=names, y=pca.explained_variance_ratio_ * 100,
            marker_color=PURPLE, text=[f"{v:.2f}%" for v in pca.explained_variance_ratio_ * 100],
            textposition="outside",
        ))
        variance_fig.update_layout(
            barmode="group", template="plotly_white", height=380,
            margin=dict(l=45, r=20, t=40, b=40),
            yaxis_title="Explained variance (%)",
            yaxis=dict(range=[0, 110]),
            legend_title="Experiment",
        )

        before_ratio = separation_ratio(BASE_LDA_SCORES[:, :dimensions], Y)
        after_ratio = separation_ratio(lda_scores[:, :dimensions], labels)
        change = (after_ratio - before_ratio) / before_ratio * 100
        change_label = ("No change" if abs(change) < 0.005 else
                        "Increase" if change > 0 else "Decrease")
        explanation = html.Div([
            html.H3(f"Understanding the Outlier — {dimension_suffix}"),
            html.P([html.Strong("Experiment: "),
                f"{len(data)} observations are displayed. " +
                (f"A synthetic Setosa point is added {float(distance):g} standardized units from its class center."
                 if enabled else "No outlier is added; the original models are unchanged.")]),
            html.P([html.Strong("PCA: "),
                f"PC1 retains {pca.explained_variance_ratio_[0]*100:.2f}% of variance, "
                f"and its direction changes by {angle:.2f}° relative to baseline."]),
            html.P([html.Strong("LDA: "),
                f"The projected separation ratio changes from {before_ratio:.3f} to "
                f"{after_ratio:.3f} ({change:+.2f}%). This measures relative class "
                "separation, not prediction accuracy."]),
            html.P([html.Strong("Visualization: "),
                "Only PC1 and LD1 are real coordinates in the 1D view. "
                "Vertical jitter is for readability and is excluded from calculations."
                if dimensions == 1 else
                "Both axes show real projected coordinates in the 2D view. "
                "The before and after plots for each algorithm share axis scales."]),
        ])

        return (
            str(len(data)),
            f"{pca.explained_variance_ratio_[0] * 100:.2f}%",
            f"{angle:.2f}°",
            f"{BASE_PCA.explained_variance_ratio_[0]*100:.2f}%",
            f"{pca.explained_variance_ratio_[0]*100:.2f}%",
            (lambda d: "0.00 pp (No change)" if abs(d) < 0.005 else
             f"{d:+.2f} pp ({'Increase' if d > 0 else 'Decrease'})")(
                 (pca.explained_variance_ratio_[0] - BASE_PCA.explained_variance_ratio_[0]) * 100),
            pca_before, pca_after, lda_before, lda_after, variance_fig,
            f"{before_ratio:.3f}", f"{after_ratio:.3f}",
            f"{change:+.2f}% ({change_label})" if change_label != "No change" else "0.00% (No change)",
            explanation,
        )
