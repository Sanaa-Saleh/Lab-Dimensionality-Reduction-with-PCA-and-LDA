from chart_style import finish_chart

import numpy as np
import plotly.graph_objects as go

from dash import html, dcc, Input, Output
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ==========================================
# 1. Load and Prepare Dataset
# ==========================================

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

feature_names = cancer.feature_names
class_names = cancer.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA fitted to all 30 standardized features
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# LDA: two classes allow only one component
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratio)

components_95 = int(
    np.searchsorted(cumulative_variance, 0.95) + 1
)

COLORS = {
    0: "#eaa054",
    1: "#24b6a4"
}

PURPLE = "#7654d8"


# ==========================================
# 2. Shared Chart Styling
# ==========================================

def style_figure(fig, title, height=480, legend_title="Diagnosis"):

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        margin=dict(l=45, r=25, t=65, b=45),
        font=dict(family="Segoe UI, Arial"),
        legend_title_text=legend_title,
        uirevision="breast-cancer"
    )

    return finish_chart(fig)
# ==========================================
# 3. PCA Visualization
# ==========================================

def create_pca_figure():

    fig = go.Figure()

    for class_id, name in enumerate(class_names):

        mask = y == class_id

        fig.add_trace(
            go.Scatter(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                mode="markers",
                name=name.capitalize(),
                marker=dict(
                    color=COLORS[class_id],
                    size=8,
                    opacity=0.75
                ),
                hovertemplate=(
                    "PC1: %{x:.3f}"
                    "<br>PC2: %{y:.3f}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_xaxes(title="PC1 Score")
    fig.update_yaxes(title="PC2 Score")

    return style_figure(
        fig,
        "PCA Projection — 30 Features to 2D"
    )


# ==========================================
# 4. LDA Visualization
# ==========================================

def create_lda_figure():

    fig = go.Figure()

    rng = np.random.default_rng(42)
    jitter = rng.uniform(
        -0.16, 0.16, size=len(y)
    )

    for class_id, name in enumerate(class_names):

        mask = y == class_id

        fig.add_trace(
            go.Scatter(
                x=X_lda[mask, 0],
                y=jitter[mask],
                mode="markers",
                name=name.capitalize(),
                marker=dict(
                    color=COLORS[class_id],
                    size=8,
                    opacity=0.75
                ),
                hovertemplate=(
                    "LD1: %{x:.3f}"
                    "<extra></extra>"
                )
            )
        )

    fig.add_hline(
        y=0,
        line_color=PURPLE,
        line_width=2
    )

    fig.update_xaxes(title="LD1 Score")
    fig.update_yaxes(
        title="Display Jitter (not a feature)",
        range=[-0.3, 0.3]
    )

    return style_figure(
        fig,
        "LDA Projection — 30 Features to 1D"
    )


# ==========================================
# 5. Explained Variance Chart
# ==========================================

def create_variance_figure(target=95):

    target_fraction = target / 100

    needed = int(
        np.searchsorted(
            cumulative_variance,
            target_fraction
        ) + 1
    )

    components = np.arange(
        1, len(cumulative_variance) + 1
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=components,
            y=cumulative_variance * 100,
            mode="lines+markers",
            name="Cumulative Variance",
            line=dict(
                color=PURPLE,
                width=3
            ),
            marker=dict(size=5),
            hovertemplate=(
                "Components: %{x}"
                "<br>Variance: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )

    fig.add_hline(
        y=target,
        line_dash="dash",
        line_color="#eaa054",
        annotation_text=f"Target {target}%"
    )

    fig.add_trace(
        go.Scatter(
            x=[needed],
            y=[
                cumulative_variance[needed - 1] * 100
            ],
            mode="markers",
            name="Minimum Components",
            marker=dict(
                color="#24b6a4",
                size=15,
                symbol="diamond"
            )
        )
    )

    fig.update_xaxes(
        title="Number of Principal Components",
        dtick=2,
        range=[0.5, 30.5]
    )

    fig.update_yaxes(
        title="Cumulative Explained Variance (%)",
        range=[0, 105]
    )

    fig.update_layout(
        legend_title_text="PCA Variance"
    )

    return style_figure(
        fig,
        "Cumulative Explained Variance",
        legend_title="PCA Variance"
    ), needed

# ==========================================
# 6. Page Layout
# ==========================================

def layout():

    return html.Div([

        html.H1(
            "Breast Cancer Dataset",
            className="page-title"
        ),

        html.P(
            "Explore PCA and LDA on a real dataset "
            "with 30 numerical features and two "
            "diagnostic classes.",
            className="page-description"
        ),

        # Dataset statistics
        html.Div(
            className="stats-grid",
            children=[

                html.Div(
                    className="stat-card",
                    children=[
                        html.Div(
                            "Total Samples",
                            className="stat-label"
                        ),
                        html.Div(
                            str(X.shape[0]),
                            className="stat-value"
                        )
                    ]
                ),

                html.Div(
                    className="stat-card",
                    children=[
                        html.Div(
                            "Original Features",
                            className="stat-label"
                        ),
                        html.Div(
                            str(X.shape[1]),
                            className="stat-value"
                        )
                    ]
                ),

                html.Div(
                    className="stat-card",
                    children=[
                        html.Div(
                            "Target Classes",
                            className="stat-label"
                        ),
                        html.Div(
                            str(len(class_names)),
                            className="stat-value"
                        )
                    ]
                )
            ]
        ),

        # PCA and LDA comparison
        html.Div(
            className="chart-card",
            children=[

                html.Div(
                    "PCA vs LDA Visualization",
                    className="section-title"
                ),

                html.Div(
                    "PCA maximizes overall variance. "
                    "LDA uses diagnosis labels to "
                    "find directions that separate classes.",
                    className="section-description"
                ),

                dcc.Graph(
                    id="bc-pca-graph",
                    figure=create_pca_figure()
                ),

                dcc.Graph(
                    id="bc-lda-graph",
                    figure=create_lda_figure()
                ),

                html.Div(
                    className="info-box",
                    style={"marginTop": "16px"},
                    children=[
                        html.H3("Understanding PCA vs LDA"),
                        html.P([
                            html.Strong("PCA: "),
                            "The 2D plot uses PC1 and PC2, directions selected to preserve "
                            "overall variation in the 30 standardized features. "
                            "The diagnosis labels are used only to color the points."
                        ]),
                        html.P([
                            html.Strong("LDA: "),
                            "The horizontal LD1 coordinate is learned using the diagnosis "
                            "labels to emphasize separation between the two classes. "
                            "The vertical jitter is only for display; it is not a second feature."
                        ]),
                        html.P([
                            html.Strong("Class colors: "),
                            "Orange represents Malignant and teal represents Benign. "
                            "These are actual dataset labels, not model predictions."
                        ]),
                        html.P(
                            "Visual separation does not establish diagnostic accuracy. "
                            "These are projections of the same 569 samples."
                        )
                    ]
                )
            ]
        ),

        # Explained variance experiment
        html.Div(
            className="chart-card",
            children=[

                html.Div(
                    "Interactive Explained Variance",
                    className="section-title"
                ),

                html.Div(
                    "Change the target to discover "
                    "how many principal components "
                    "are required.",
                    className="section-description"
                ),

                html.Label(
                    "Target Explained Variance"
                ),

                dcc.Slider(
                    id="bc-target",
                    min=50,
                    max=99,
                    step=1,
                    value=95,
                    marks={
                        50: "50%",
                        70: "70%",
                        80: "80%",
                        90: "90%",
                        95: "95%",
                        99: "99%"
                    },
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True
                    }
                ),

                html.Div(
                    className="stats-grid",
                    style={"marginTop": "35px"},
                    children=[

                        html.Div(
                            className="stat-card",
                            children=[
                                html.Div(
                                    "Components Needed",
                                    className="stat-label"
                                ),
                                html.Div(
                                    id="bc-components",
                                    className="stat-value"
                                )
                            ]
                        ),

                        html.Div(
                            className="stat-card",
                            children=[
                                html.Div(
                                    "Dimensions Reduced",
                                    className="stat-label"
                                ),
                                html.Div(
                                    id="bc-reduction",
                                    className="stat-value"
                                )
                            ]
                        ),

                        html.Div(
                            className="stat-card",
                            children=[
                                html.Div(
                                    "Variance Retained",
                                    className="stat-label"
                                ),
                                html.Div(
                                    id="bc-retained",
                                    className="stat-value"
                                )
                            ]
                        )
                    ]
                ),

                dcc.Graph(id="bc-variance-graph"),
                html.Div(
                    id="bc-variance-explanation",
                    className="info-box",
                    style={"marginTop": "16px"}
                )
            ]
        ),

        # Explanation
        html.Div(
            className="info-box",
            children=[

                html.H3("Understanding the Results"),

                html.P(
                    "The Breast Cancer Wisconsin "
                    "Diagnostic dataset contains "
                    "569 samples and 30 numerical features."
                ),

                html.P(
                    "PCA is unsupervised. It reduces "
                    "dimensions while preserving as "
                    "much total variance as possible."
                ),

                html.P(
                    "LDA is supervised. It uses the "
                    "diagnosis labels to find directions "
                    "that separate the two classes."
                ),

                html.P(
                    "Because this dataset contains "
                    "two classes, LDA can produce "
                    "at most one discriminant component."
                ),

                html.P(
                    "The plots are educational "
                    "visualizations, not measurements "
                    "of diagnostic accuracy."
                )
            ]
        )

    ])


# ==========================================
# 7. Interactive Callback
# ==========================================

def register_callbacks(app):

    @app.callback(
        Output("bc-variance-graph", "figure"),
        Output("bc-components", "children"),
        Output("bc-reduction", "children"),
        Output("bc-retained", "children"),
        Output("bc-variance-explanation", "children"),
        Input("bc-target", "value")
    )
    def update_variance(target):

        fig, needed = create_variance_figure(
            target
        )

        retained = (
            cumulative_variance[needed - 1] * 100
        )

        explanation = [
            html.H3(f"Understanding the {target}% Variance Target"),
            html.P([
                html.Strong("Components needed: "),
                f"The smallest number of principal components reaching at least "
                f"{target}% explained variance is {needed}."
            ]),
            html.P([
                html.Strong("Dimension reduction: "),
                f"PCA reduces the representation from 30 original features to "
                f"{needed} principal components, removing {30 - needed} dimensions."
            ]),
            html.P([
                html.Strong("Actual variance retained: "),
                f"The selected {needed} components explain {retained:.2f}% of total "
                f"standardized feature variance. This can be slightly higher "
                f"than the {target}% target because components are added one at a time."
            ]),
            html.P(
                "The purple curve shows cumulative explained variance; the "
                "orange dashed line marks the target, and the teal diamond "
                "marks the minimum number of components needed. "
                "Explained variance is not classification accuracy."
            )
        ]

        return (
            fig,
            str(needed),
            f"30 → {needed}",
            f"{retained:.2f}%",
            explanation
        )
