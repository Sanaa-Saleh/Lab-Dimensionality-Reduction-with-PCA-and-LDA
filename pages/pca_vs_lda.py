import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = load_iris()
labels = iris.target
names = iris.target_names
palette = ['#24b6a4', '#eaa054', '#8e80e6']
scaled = StandardScaler().fit_transform(iris.data)
pca_model = PCA(n_components=2)
lda_model = LinearDiscriminantAnalysis(n_components=2)
pca_scores = pca_model.fit_transform(scaled)
lda_scores = lda_model.fit_transform(scaled, labels)

# Fixed, deterministic jitter for display only (never used for calculations).
rng = np.random.default_rng(17)
jitter = rng.uniform(-0.13, 0.13, len(labels))


def separation_ratio(scores):
    """Between-class scatter / within-class scatter in the displayed projection."""
    grand = scores.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in range(len(names)):
        group = scores[labels == c]
        center = group.mean(axis=0)
        between += len(group) * float(np.sum((center - grand) ** 2))
        within += float(np.sum((group - center) ** 2))
    return between / within if within > 0 else float('inf')


def plot_projection(scores, title, dim, centroids):
    fig = go.Figure()
    for i, species in enumerate(names):
        mask = labels == i
        xs = scores[mask, 0]
        ys = scores[mask, 1] if dim == '2D' else jitter[mask]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers', name=species,
            marker=dict(color=palette[i], size=9, opacity=0.82),
            hovertemplate=(
                'Sample: %{customdata}<br>Axis 1: %{x:.3f}'
                + ('<br>Axis 2: %{y:.3f}' if dim == '2D' else '')
                + '<extra>%s</extra>' % species
            ),
            customdata=np.flatnonzero(mask) + 1,
        ))
        if centroids:
            fig.add_trace(go.Scatter(
                x=[float(xs.mean())],
                y=[float(scores[mask, 1].mean()) if dim == '2D' else 0.0],
                mode='markers', name=f'{species} centroid', showlegend=False,
                marker=dict(symbol='x', size=16, color=palette[i], line=dict(width=2, color='#172554')),
                hovertemplate=f'{species} centroid<extra></extra>',
            ))
    if dim == '1D':
        fig.add_hline(y=0, line_color='#7452d9', line_width=2)
    fig.update_layout(
        title=title, template='plotly_white', height=510,
        margin=dict(l=35, r=15, t=65, b=50),
        xaxis_title='PC1 score' if title.startswith('PCA') else 'LD1 score',
        yaxis_title=(('PC2 score' if title.startswith('PCA') else 'LD2 score')
                     if dim == '2D' else 'Display jitter (not a feature)'),
        legend_title='Iris species',
        font=dict(family='Segoe UI, Arial'),
        uirevision=f'{title}-{dim}',
    )
    if dim == '1D':
        fig.update_yaxes(range=[-0.3, 0.3])
    return fig


def metric_card(label, identifier):
    return html.Div(className='stat-card', children=[
        html.Div(label, className='stat-label'),
        html.Div(id=identifier, className='stat-value'),
    ])


def layout():
    return html.Div([
        html.H1('PCA vs LDA', className='page-title'),
        html.P('Compare variance preservation and supervised class separation on the same standardized Iris dataset.',
               className='page-description'),
        html.Div(className='chart-card', children=[
            html.Div('Interactive Comparison Controls', className='section-title'),
            html.P('Both algorithms are fitted to the same 150 standardized Iris samples.',
                   className='section-description'),
            html.Div(className='controls-row', children=[
                html.Div([
                    html.Label('Projection Dimensions'),
                    dcc.RadioItems(
                        id='compare-dimension',
                        options=[{'label': '1D (PC1 / LD1)', 'value': '1D'},
                                 {'label': '2D (PC1 + PC2 / LD1 + LD2)', 'value': '2D'}],
                        value='2D', inline=True,
                        labelStyle={'marginRight': '22px', 'display': 'inline-block', 'marginTop': '10px'},
                    ),
                ], style={'flex': '2', 'minWidth': '280px'}),
                html.Div([
                    dcc.Checklist(
                        id='compare-centroids',
                        options=[{'label': ' Show class centroids', 'value': 'show'}],
                        value=['show'],
                        style={'marginTop': '25px'},
                    ),
                ], style={'flex': '1', 'minWidth': '190px'}),
            ]),
        ]),
        html.Div(className='stats-grid', children=[
            metric_card('PCA: Projected Separation Ratio', 'compare-pca-separation'),
            metric_card('LDA: Projected Separation Ratio', 'compare-lda-separation'),
            metric_card('PCA: Original Variance Retained', 'compare-pca-variance'),
        ]),
        html.Div(className='chart-card', children=[
            html.Div('Side-by-Side Projections', className='section-title'),
            html.P('The same samples appear in both charts. The horizontal scales are algorithm-specific; compare class overlap, not absolute coordinates.',
                   className='section-description'),
            html.Div([
                html.Div(dcc.Graph(id='compare-pca-graph', config={'displaylogo': False}),
                         style={'flex': '1', 'minWidth': '340px'}),
                html.Div(dcc.Graph(id='compare-lda-graph', config={'displaylogo': False}),
                         style={'flex': '1', 'minWidth': '340px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px'}),
        ]),
        html.Div(className='info-box', style={'marginBottom': '20px'}, children=[
            html.H3(id='compare-explanation-title'),
            html.P(id='compare-explanation-pca'),
            html.P(id='compare-explanation-lda'),
            html.P(id='compare-explanation-centroids'),
        ]),
        html.Div(className='chart-card', children=[
            html.Div('Projected Class Separation', className='section-title'),
            html.P('Between-class scatter divided by within-class scatter in each displayed projection. Higher means better separation relative to within-class spread; this is not classification accuracy.',
                   className='section-description'),
            dcc.Graph(id='compare-ratio-chart', config={'displaylogo': False}),
        ]),
        html.Div(className='info-box', children=[
            html.H3('What is the difference?'),
            html.Div([
                html.Div([
                    html.H4('PCA — Principal Component Analysis'),
                    html.P('Unsupervised: does not use Iris species labels.'),
                    html.P('Goal: preserve as much overall feature variance as possible.'),
                    html.P('Iris: up to 4 principal components.'),
                ], style={'flex': '1', 'minWidth': '260px'}),
                html.Div([
                    html.H4('LDA — Linear Discriminant Analysis'),
                    html.P('Supervised: uses Iris species labels.'),
                    html.P('Goal: separate class means relative to within-class variation.'),
                    html.P('Iris: at most 2 discriminant components (3 classes − 1).'),
                ], style={'flex': '1', 'minWidth': '260px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '24px'}),
            html.P('Important: PCA explained variance measures the fraction of original standardized feature variance retained. LDA discriminant variance ratios measure something different and should not be compared directly.'),
            html.P('The projected separation ratio is between-class scatter divided by within-class scatter in the selected projection. A higher value means stronger relative separation under this metric, not higher classification accuracy.'),
            html.P('In 1D, vertical jitter is only a visual aid and is excluded from all calculations.'),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output('compare-pca-graph', 'figure'),
        Output('compare-lda-graph', 'figure'),
        Output('compare-ratio-chart', 'figure'),
        Output('compare-pca-separation', 'children'),
        Output('compare-lda-separation', 'children'),
        Output('compare-pca-variance', 'children'),
        Output('compare-explanation-title', 'children'),
        Output('compare-explanation-pca', 'children'),
        Output('compare-explanation-lda', 'children'),
        Output('compare-explanation-centroids', 'children'),
        Input('compare-dimension', 'value'),
        Input('compare-centroids', 'value'),
    )
    def update_comparison(dimension, centroid_values):
        dim = dimension if dimension in ('1D', '2D') else '2D'
        n = 1 if dim == '1D' else 2
        show = 'show' in (centroid_values or [])
        pca_slice = pca_scores[:, :n]
        lda_slice = lda_scores[:, :n]
        pca_ratio = separation_ratio(pca_slice)
        lda_ratio = separation_ratio(lda_slice)
        retained = float(pca_model.explained_variance_ratio_[:n].sum() * 100)

        bars = go.Figure(go.Bar(
            x=['PCA', 'LDA'], y=[pca_ratio, lda_ratio],
            marker_color=['#24b6a4', '#7654db'],
            text=[f'{pca_ratio:.3f}', f'{lda_ratio:.3f}'],
            textposition='outside',
        ))
        bars.update_layout(
            template='plotly_white', height=340,
            margin=dict(l=35, r=20, t=35, b=40),
            yaxis_title='Between / Within scatter ratio',
            font=dict(family='Segoe UI, Arial'),
        )
        bars.update_yaxes(range=[0, max(pca_ratio, lda_ratio) * 1.2])
        pca_note = (
            f'PCA: PC1 retains {retained:.2f}% of the original standardized feature variance. '
            'Only the PC1 coordinate is real; vertical jitter is for readability.'
            if dim == '1D' else
            f'PCA: PC1 and PC2 together retain {retained:.2f}% of the original standardized feature variance. '
            'Both plot axes are real PCA coordinates.'
        )
        lda_retained = float(lda_model.explained_variance_ratio_[:n].sum() * 100)
        lda_note = (
            f'LDA: LD1 represents {lda_retained:.2f}% of the discriminant variance. '
            'The vertical jitter is not a second feature.'
            if dim == '1D' else
            f'LDA: LD1 and LD2 together represent {lda_retained:.2f}% of the discriminant variance. '
            'Both axes are real LDA coordinates.'
        )
        centroid_note = (
            'Class Centroids: The × markers show the mean position of each Iris species '
            + ('along PC1 or LD1; their vertical position is only for display.'
               if dim == '1D' else 'in each two-dimensional projection.')
            if show else 'Class Centroids: Hidden in both charts. Enable the checkbox to display them.'
        )
        return (
            plot_projection(pca_scores, 'PCA Projection — ' + dim, dim, show),
            plot_projection(lda_scores, 'LDA Projection — ' + dim, dim, show),
            bars,
            f'{pca_ratio:.3f}',
            f'{lda_ratio:.3f}',
            f'{retained:.2f}%',
            f'Understanding {dim} PCA vs LDA',
            pca_note,
            lda_note,
            centroid_note,
        )
