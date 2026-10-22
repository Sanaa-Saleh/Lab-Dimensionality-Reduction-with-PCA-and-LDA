from dash import Dash, html, dcc, Input, Output
from sklearn.datasets import load_iris
import pandas as pd
import plotly.express as px

from pages import standardization
from pages import variance_covariance
from pages import pca_visualization
from pages import explained_variance
from pages import dimension_reduction
from pages import lda_visualization
from pages import pca_vs_lda
from pages import outlier_experiment
from pages import breast_cancer


# ==========================================
# 1. Load Iris Dataset
# ==========================================
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = [iris.target_names[i] for i in iris.target]
feature_options = [{'label': col, 'value': col} for col in iris.feature_names]
colors = ['#24b6a4', '#eaa054', '#8e80e6']


# ==========================================
# 2. Create Dash Application
# ==========================================
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    prevent_initial_callbacks='initial_duplicate',
    meta_tags=[{'name':'viewport','content':'width=device-width, initial-scale=1'}]
)

server = app.server

standardization.register_callbacks(app)
variance_covariance.register_callbacks(app)
pca_visualization.register_callbacks(app)
explained_variance.register_callbacks(app)
dimension_reduction.register_callbacks(app)
lda_visualization.register_callbacks(app)
pca_vs_lda.register_callbacks(app)
outlier_experiment.register_callbacks(app)
breast_cancer.register_callbacks(app)


# ==========================================
# 3. Sidebar Navigation
# ==========================================
def navigation_item(label, page):
    return dcc.Link(
        label,
        href=f'/{page}',
        className='nav-item',
        style={'display': 'block', 'textDecoration': 'none'}
    )


sidebar = html.Div(
    className='sidebar',
    children=[
        html.H2('Lab6 Dimensionality ', className='sidebar-title'),
        html.P('Reduction with PCA and  LDA', className='sidebar-subtitle'),
        html.P('EXPLORATION', className='sidebar-heading'),
        navigation_item('Dataset Explorer', ''),
        navigation_item('Standardization', 'standardization'),
        navigation_item('Variance & Covariance', 'variance-covariance'),
        html.P('ALGORITHMS', className='sidebar-heading'),
        navigation_item('PCA Visualization', 'pca-visualization'),
        navigation_item('Explained Variance', 'explained-variance'),
        navigation_item('Dimension Reduction', 'dimension-reduction'),
        navigation_item('LDA Visualization', 'lda-visualization'),
        navigation_item('PCA vs LDA', 'pca-vs-lda'),
        html.P('EXPERIMENTS', className='sidebar-heading'),
        navigation_item('Outlier Experiment', 'outlier-experiment'),
        navigation_item('Breast Cancer Dataset', 'breast-cancer'),

        html.Div(
            children=[
                html.Div(
                    style={
                        'height': '1px',
                        'backgroundColor': '#465371',
                        'marginBottom': '14px'
                    }
                ),

            html.P(
                'Designed & Developed by',
                style={
                    'color': '#a7b4d0',
                    'fontSize': '12px',
                    'textAlign': 'center',
                    'margin': '0 0 6px'
                }
            ),

            html.P(
                'Sanaa Saleh Albatati',
                style={
                    'color': '#ffffff',
                    'fontSize': '14px',
                    'fontWeight': '600',
                    'textAlign': 'center',
                    'margin': '0'
                }
            )
        ],
        style={
            'marginTop': '32px',
            'paddingBottom': '10px'
        }
    )
        
    ]
)


# ==========================================
# 4. Dataset Explorer Layout
# ==========================================
def dataset_layout():
    return html.Div([
        html.H1('Dataset Explorer', className='page-title'),
        html.P(
            'Explore the original Iris dataset in interactive 2D and 3D visualizations.',
            className='page-description'
        ),
        html.Div(className='stats-grid', children=[
            html.Div(className='stat-card', children=[
                html.Div('Total Samples', className='stat-label'),
                html.Div(str(df.shape[0]), className='stat-value')
            ]),
            html.Div(className='stat-card', children=[
                html.Div('Original Features', className='stat-label'),
                html.Div(str(iris.data.shape[1]), className='stat-value')
            ]),
            html.Div(className='stat-card', children=[
                html.Div('Target Classes', className='stat-label'),
                html.Div(str(len(iris.target_names)), className='stat-value')
            ])
        ]),
        html.Div(className='chart-card', children=[
            html.Div('Original Data Distribution', className='section-title'),
            html.Div(
                'Select features and switch between 2D and 3D to explore the Iris dataset.',
                className='section-description'
            ),
            html.Div(className='controls-row', children=[
                html.Div(children=[
                    html.Label('Visualization Mode'),
                    dcc.RadioItems(
                        id='plot-mode',
                        options=[{'label': '2D', 'value': '2D'}, {'label': '3D', 'value': '3D'}],
                        value='2D',
                        inline=True,
                        labelStyle={'marginRight': '15px'},
                        style={'marginTop': '10px'}
                    )
                ], style={'minWidth': '140px'}),
                html.Div(children=[
                    html.Label('X-axis'),
                    dcc.Dropdown(
                        id='x-feature', options=feature_options,
                        value=iris.feature_names[0], clearable=False
                    )
                ], style={'minWidth': '180px', 'flex': '1'}),
                html.Div(children=[
                    html.Label('Y-axis'),
                    dcc.Dropdown(
                        id='y-feature', options=feature_options,
                        value=iris.feature_names[1], clearable=False
                    )
                ], style={'minWidth': '180px', 'flex': '1'}),
                html.Div(children=[
                    html.Label('Z-axis (3D only)'),
                    dcc.Dropdown(
                        id='z-feature', options=feature_options,
                        value=iris.feature_names[2], clearable=False
                    )
                ], id='z-feature-container', style={'minWidth': '180px', 'flex': '1'})
            ]),
            dcc.Graph(id='iris-scatter', config={'displaylogo': False, 'responsive': True, 'scrollZoom': False, 'displayModeBar': False}
            )
        ]),
        html.Div(className='info-box', children=[
            html.Strong('Understanding the Dataset'),
            html.P(
                'The Iris dataset contains 150 samples, four numerical features, and three flower species.'
            ),
            html.P(
                'Switch between 2D and 3D and select different features to explore the data.'
            )
        ])
    ])


# ==========================================
# 5. Main Application Layout
# ==========================================
app.layout = html.Div(
    className='app-container',
    children=[
        dcc.Location(id='url'),
        html.Div([html.Button('☰', id='menu-button', n_clicks=0, className='menu-button', **{'aria-label':'Toggle navigation'}),
                  html.Div(html.Span(id='current-page-name', className='current-page-name'), className='header-label')], className='mobile-header'),
        sidebar,
        html.Div(id='menu-overlay', className='menu-overlay', n_clicks=0),
        html.Div(id='page-content', className='main-content')
    ]
)


# ==========================================
# 6. Page Navigation Callback
# ==========================================
app.clientside_callback(
    """function(clicks, path, overlay) {
        const trigger = dash_clientside.callback_context.triggered_id || '';
        const mobile = window.matchMedia('(max-width:700px)').matches;
        if (trigger === 'menu-button') {
            if (mobile) document.body.classList.toggle('menu-open');
            else {
                document.body.classList.toggle('sidebar-collapsed');
                sessionStorage.setItem('dimlab-sidebar-collapsed', document.body.classList.contains('sidebar-collapsed') ? '1' : '0');
            }
        } else {
            document.body.classList.remove('menu-open');
            if (!mobile) document.body.classList.toggle('sidebar-collapsed', sessionStorage.getItem('dimlab-sidebar-collapsed') === '1');
        }
        requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
        return mobile && document.body.classList.contains('menu-open') ? '✕' : '☰';
    }""",
    Output('menu-button','children'), Input('menu-button','n_clicks'),
    Input('url','pathname'), Input('menu-overlay','n_clicks')
)

app.clientside_callback(
    """function(path) {
        const names = {'/':'Dataset Explorer','/standardization':'Standardization',
          '/variance-covariance':'Variance & Covariance','/pca-visualization':'PCA Visualization',
          '/explained-variance':'Explained Variance','/dimension-reduction':'Dimension Reduction',
          '/lda-visualization':'LDA Visualization','/pca-vs-lda':'PCA vs LDA',
          '/outlier-experiment':'Outlier Experiment','/breast-cancer':'Breast Cancer Dataset'};
        const active = path || '/';
        document.querySelectorAll('.sidebar .nav-item').forEach(link => {
            const url = new URL(link.href, window.location.href);
            const on = url.pathname === active;
            link.classList.toggle('active', on);
            if (on) link.setAttribute('aria-current','page');
            else link.removeAttribute('aria-current');
        });
        return names[active] || 'Dataset Explorer';
    }""", Output('current-page-name','children'), Input('url','pathname')
)

@app.callback(Output('page-content' , 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/standardization':
        return standardization.layout()
    if pathname == '/variance-covariance':
        return variance_covariance.layout()
    if pathname == '/pca-visualization':
        return pca_visualization.layout()
    if pathname == '/explained-variance':
        return explained_variance.layout()
    if pathname == '/dimension-reduction':
        return dimension_reduction.layout()
    if pathname == '/lda-visualization':
        return lda_visualization.layout()
    if pathname == '/pca-vs-lda':
        return pca_vs_lda.layout()
    if pathname == '/outlier-experiment':
        return outlier_experiment.layout()
    if pathname == '/breast-cancer':
        return breast_cancer.layout()
    return dataset_layout()
    


# ==========================================
# 7. Dataset Explorer Callback
# ==========================================
@app.callback(
    Output('iris-scatter', 'figure'),
    Input('plot-mode', 'value'),
    Input('x-feature', 'value'),
    Input('y-feature', 'value'),
    Input('z-feature', 'value')
)
def update_scatter(mode, x_feature, y_feature, z_feature):
    if mode == '3D':
        fig = px.scatter_3d(
            df, x=x_feature, y=y_feature, z=z_feature,
            color='Species', hover_data=iris.feature_names,
            color_discrete_sequence=colors,
            title='Iris Dataset - 3D Visualization'
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8))
    else:
        fig = px.scatter(
            df, x=x_feature, y=y_feature,
            color='Species', hover_data=iris.feature_names,
            color_discrete_sequence=colors,
            title='Iris Dataset - 2D Visualization'
        )
        fig.update_traces(marker=dict(size=9, opacity=0.8))

    fig.update_layout(
        template='plotly_white', height=550,
        font=dict(family='Segoe UI, Arial'),
        legend_title_text='Iris Species',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.25, yanchor='top'),
        margin=dict(l=40,r=25,t=65,b=95)
    )
    if mode != '3D':
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
    return fig


# ==========================================
# 8. Show / Hide Z-axis
# ==========================================
@app.callback(Output('z-feature-container', 'style'), Input('plot-mode', 'value'))
def toggle_z_axis(mode):
    if mode == '3D':
        return {'minWidth': '180px', 'flex': '1'}
    return {'display': 'none'}


# ==========================================
# 9. Run Application
# ==========================================
if __name__ == '__main__':
    app.run(debug=False)
