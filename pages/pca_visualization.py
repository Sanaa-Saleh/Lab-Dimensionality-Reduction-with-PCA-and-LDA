import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, ctx, no_update
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
pca = PCA(n_components=2).fit(X)
S = pca.transform(X)
C = pca.components_
colors = ['#24b6a4', '#eaa054', '#8e80e6']
STEP = 0.025

def make_figure(progress):
    # Work in the PC1-PC2 plane, so distances and angles are faithful to PCA.
    original = S.copy()
    projected = S.copy()
    projected[:, 1] = 0
    points = original + progress * (projected - original)
    x_bound = max(3.5, float(np.max(np.abs(S[:, 0]))) * 1.12)
    y_bound = max(3.5, float(np.max(np.abs(S[:, 1]))) * 1.12)
    fig = go.Figure()
    for i, name in enumerate(iris.target_names):
        mask = iris.target == i
        fig.add_trace(go.Scatter(
            x=points[mask, 0], y=points[mask, 1], mode='markers',
            name=name, marker=dict(color=colors[i], size=9, opacity=0.85),
            customdata=np.where(mask)[0].reshape(-1, 1),
            hovertemplate='Sample %{customdata[0]}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
        ))
    fig.add_hline(y=0, line_color='#7c3aed', line_width=3,
                  annotation_text='PC1 — projection target', annotation_position='top right')
    fig.add_vline(x=0, line_color='#64748b', line_dash='dash')
    fig.update_layout(
        title=f'Projection onto PC1 — {progress * 100:.0f}% complete',
        template='plotly_white', height=520,
        margin=dict(l=50, r=25, t=70, b=50),
        xaxis=dict(title='PC1 score', range=[-x_bound, x_bound], autorange=False, fixedrange=True),
        yaxis=dict(title='PC2 score', range=[-y_bound, y_bound], autorange=False, fixedrange=True),
        transition=dict(duration=0),
        uirevision='pca-projection-fixed'
    )
    return fig

FEATURE_LABELS = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
FEATURE_COLORS = ['#6366f1', '#ef553b', '#10b981', '#f59e0b']


def make_loadings(selected='all'):
    """Feature component weights, not sample scores or PCA axes."""
    fig = go.Figure()
    for j, label in enumerate(FEATURE_LABELS):
        active = selected == 'all' or selected == str(j)
        weight1, weight2 = float(C[0, j]), float(C[1, j])
        fig.add_trace(go.Scatter(
            x=[0, weight1], y=[0, weight2],
            mode='lines+markers', name=label,
            line=dict(color=FEATURE_COLORS[j], width=5 if selected == str(j) else 3),
            marker=dict(color=FEATURE_COLORS[j], size=[5, 12 if selected == str(j) else 8]),
            opacity=1 if active else 0.13,
            customdata=[[weight1, weight2], [weight1, weight2]],
            hovertemplate=(
                f'<b>{label}</b><br>'
                'PC1 weight: %{customdata[0]:+.4f}<br>'
                'PC2 weight: %{customdata[1]:+.4f}<extra></extra>'
            )
        ))
    fig.add_hline(y=0, line_color='#cbd5e1', layer='below')
    fig.add_vline(x=0, line_color='#cbd5e1', layer='below')
    fig.update_layout(
        title='Feature Weights in PC1 and PC2',
        template='plotly_white', height=490,
        showlegend=True, legend=dict(orientation='h', y=-0.27, x=0),
        # Independent, fixed ranges let the chart use the full card width.
        # Include headroom above Sepal Width's PC2 loading (~0.923).
        xaxis=dict(title='PC1 component weight', range=[-0.75, 0.75],
                   autorange=False, fixedrange=True, zeroline=False),
        yaxis=dict(title='PC2 component weight', range=[-0.20, 1.12],
                   autorange=False, fixedrange=True, zeroline=False),
        margin=dict(l=55, r=30, t=65, b=110),
        hovermode='closest'
    )
    return fig


def feature_details(selected):
    if selected == 'all':
        return ('All four features', '—', '—',
                'Choose one feature to highlight its vector and inspect its signed PC1 and PC2 weights.')
    j = int(selected)
    w1, w2 = float(C[0, j]), float(C[1, j])
    return (FEATURE_LABELS[j], f'{w1:+.4f}', f'{w2:+.4f}',
            'Positive and negative signs indicate direction along each principal component. '
            'The size of a weight indicates its contribution to that component; '
            'these are component coefficients, not percentages.')

def layout():
    return html.Div([
        html.H1('PCA Visualization', className='page-title'),
        html.P('See how PCA finds directions of maximum variance and projects data onto PC1.',
               className='page-description'),
        html.Div(className='stats-grid', children=[
            html.Div(className='stat-card', children=[
                html.Div('PC1 Explained Variance', className='stat-label'),
                html.Div(f'{pca.explained_variance_ratio_[0] * 100:.2f}%', className='stat-value')
            ]),
            html.Div(className='stat-card', children=[
                html.Div('PC2 Explained Variance', className='stat-label'),
                html.Div(f'{pca.explained_variance_ratio_[1] * 100:.2f}%', className='stat-value')
            ]),
            html.Div(className='stat-card', children=[
                html.Div('PC1 + PC2', className='stat-label'),
                html.Div(f'{sum(pca.explained_variance_ratio_) * 100:.2f}%', className='stat-value')
            ])
        ]),
        html.Div(className='chart-card', children=[
            html.Div('Projection Controls', className='section-title'),
            html.P('Move the slider or press Play to project the points onto the PC1 axis.',
                   className='section-description'),
            dcc.Slider(id='pca-progress', min=0, max=1, step=STEP, value=0,
                       marks={0: 'Original PC1–PC2', 0.5: 'Halfway', 1: 'Projected onto PC1'},
                       updatemode='drag'),
            html.Div(id='pca-progress-label', style={'marginTop': '22px', 'fontWeight': '600', 'color': '#334155'}),
            html.Div(id='pca-progress-explanation', className='info-box',
                     style={'marginTop': '12px', 'marginBottom': '12px'}),
            html.Div(className='controls-row', style={'marginTop': '20px'}, children=[
                html.Button('▶ Play', id='pca-play', className='control-button'),
                html.Button('⏸ Pause', id='pca-pause', className='control-button secondary'),
                html.Button('↺ Reset', id='pca-reset', className='control-button secondary')
            ]),
            dcc.Interval(id='pca-interval', interval=200, n_intervals=0, disabled=True)
        ]),
        dcc.Store(id='pca-animation-data', data={
            'base': __import__('json').loads(make_figure(0).to_json()),
            'scores': S.tolist(),
            'targets': iris.target.tolist()
        }),
        html.Div(className='chart-card', children=[
            html.Div('Live PCA Projection', className='section-title'),
            dcc.Graph(id='pca-projection-graph', figure=make_figure(0), config={'displaylogo': False})
        ]),
        html.Div(className='chart-card', children=[
            html.Div('How Original Features Form the PCs', className='section-title'),
            html.P('These vectors show feature weights in PC1 and PC2. The PCA axes themselves live in four-dimensional standardized feature space.',
                   className='section-description'),
            html.Div(className='controls-row', children=[
                html.Div([
                    html.Label('Highlight Feature'),
                    dcc.Dropdown(
                        id='pca-feature-select',
                        options=[{'label': 'All Features', 'value': 'all'}] + [
                            {'label': name, 'value': str(i)}
                            for i, name in enumerate(FEATURE_LABELS)
                        ],
                        value='all', clearable=False
                    )
                ], style={'flex': '1', 'minWidth': '230px', 'maxWidth': '430px'})
            ]),
            dcc.Graph(id='pca-loadings-graph', config={'displaylogo': False}),
            html.Div(className='stats-grid', children=[
                html.Div(className='stat-card', children=[
                    html.Div('Selected Feature', className='stat-label'),
                    html.Div(id='pca-feature-name', className='stat-value',
                             style={'fontSize': '19px'})
                ]),
                html.Div(className='stat-card', children=[
                    html.Div('PC1 Weight', className='stat-label'),
                    html.Div(id='pca-pc1-weight', className='stat-value')
                ]),
                html.Div(className='stat-card', children=[
                    html.Div('PC2 Weight', className='stat-label'),
                    html.Div(id='pca-pc2-weight', className='stat-value')
                ])
            ]),
            html.Div(id='pca-feature-explanation', className='info-box',
                     style={'marginTop': '18px'})
        ]),
        html.Div(className='info-box', children=[
            html.H3('What is happening?'),
            dcc.Markdown(r'''
1. **Standardize** the four Iris features so each has mean 0 and standard deviation 1.
2. **Fit PCA** to find orthogonal directions of maximum variance in four-dimensional space.
3. **Transform** each flower into PCA scores. This chart displays the **PC1–PC2 score plane**.
4. **Project onto PC1:** the PC2 coordinate gradually moves to zero while PC1 stays unchanged.

$$\mathbf{z}=\mathbf{X}_{\mathrm{standardized}}\mathbf{W}^T$$

$$\mathbf{z}_{\mathrm{PC1}}=(z_1,0)$$

**Important:** This animation demonstrates the projection **within the PC1–PC2 plane**. The original dataset has four dimensions; PC3 and PC4 are not shown in this plot. The explained-variance percentages are calculated from all four standardized features.
            ''', mathjax=True)
        ])
    ])

def register_callbacks(app):
    # Browser-side controls prevent Play from waiting for Render round trips.
    app.clientside_callback(
        """function(play, pause, reset, progress) {
            const context = dash_clientside.callback_context;
            const t = context.triggered_id || ((context.triggered && context.triggered[0]) ? context.triggered[0].prop_id.split('.')[0] : null);
            if (t === 'pca-play') return Number(progress) >= 1;
            if (t === 'pca-pause' || t === 'pca-reset') return true;
            if (t === 'pca-progress' && Number(progress) >= 1) return true;
            return dash_clientside.no_update;
        }""",
        Output('pca-interval', 'disabled'),
        Input('pca-play', 'n_clicks'), Input('pca-pause', 'n_clicks'),
        Input('pca-reset', 'n_clicks'), Input('pca-progress', 'value'),
        prevent_initial_call=True
    )

    app.clientside_callback(
        """function(ticks, reset, current) {
            if (dash_clientside.callback_context.triggered.some(x => x.prop_id === 'pca-reset.n_clicks')) return 0;
            return Math.round(Math.min(1, (Number(current) || 0) + 0.05) * 1000) / 1000;
        }""",
        Output('pca-progress', 'value'),
        Input('pca-interval', 'n_intervals'), Input('pca-reset', 'n_clicks'),
        State('pca-progress', 'value'), prevent_initial_call=True
    )

    app.clientside_callback(
        """function(progress, d) {
            if (!d) return dash_clientside.no_update;
            const a = Math.max(0, Math.min(1, Number(progress || 0)));
            const fig = structuredClone(d.base);
            const groups = [[], [], []];
            for (let i = 0; i < d.scores.length; i++) {
                groups[d.targets[i]].push([d.scores[i][0], (1-a)*d.scores[i][1]]);
            }
            for (let k = 0; k < 3; k++) {
                fig.data[k].x = groups[k].map(v => v[0]);
                fig.data[k].y = groups[k].map(v => v[1]);
            }
            fig.layout.title.text = `Projection onto PC1 — ${(a*100).toFixed(0)}% complete`;
            fig.layout.transition = {duration: 110, easing: 'linear'};
            return fig;
        }""",
        Output('pca-projection-graph', 'figure'),
        Input('pca-progress', 'value'), State('pca-animation-data', 'data')
    )

    @app.callback(
        Output('pca-progress-label', 'children'),
        Output('pca-progress-explanation', 'children'),
        Input('pca-progress', 'value')
    )
    def update_graph(progress):
        progress = float(progress if progress is not None else 0)
        progress = max(0.0, min(1.0, progress))
        label = f'Projection progress: {progress * 100:.1f}%'
        if progress <= 0:
            explanation = (
                'Original PC1–PC2 view: each point keeps both its PC1 and PC2 scores. '
                'The plot shows two principal components of the standardized Iris data.'
            )
        elif progress >= 1:
            explanation = (
                'Projection complete: every point now lies on the PC1 axis (PC2 = 0), '
                'while its PC1 score stays unchanged. This is a visual demonstration of '
                'reducing the displayed two-dimensional score plane to one dimension.'
            )
        else:
            remaining = (1 - progress) * 100
            explanation = (
                f'Projection in progress: PC1 stays unchanged, while each point retains '
                f'{remaining:.1f}% of its original PC2 coordinate. '
                'This gradual movement is an educational animation, not a separate PCA fit.'
            )
        return label, explanation

    @app.callback(
        Output('pca-loadings-graph', 'figure'),
        Output('pca-feature-name', 'children'),
        Output('pca-pc1-weight', 'children'),
        Output('pca-pc2-weight', 'children'),
        Output('pca-feature-explanation', 'children'),
        Input('pca-feature-select', 'value')
    )
    def update_feature_highlight(selected):
        selected = selected if selected in ['all', '0', '1', '2', '3'] else 'all'
        name, w1, w2, explanation = feature_details(selected)
        return make_loadings(selected), name, w1, w2, explanation
