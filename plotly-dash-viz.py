from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the data
# df = pd.read_csv('rb_wr_predicted_fantasy.csv')
# df = df[df["season"] == 2024]

df = pd.read_csv('fantasy_prediction_data.csv')

df["player_id"] = df["player_id"].fillna(df["kicker_player_id"])
df["player_name"] = df["player_name"].fillna(df["kicker_player_name"])

# Initialize the app
app = Dash(__name__, suppress_callback_exceptions=True)

# Define the layout
app.layout = html.Div([
    # Your components go here
    html.H1('Fantasy Prediction Viz'),
    # dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    dcc.Tabs(id="select-view", value='week-view', children=[
        dcc.Tab(label='Weekly View', value='week-view'),
        dcc.Tab(label='Season View', value='season-view'),
    ]),
    html.Div(id='view-content'),
    
    
    # Add more components...
])

@callback(
    Output('view-content', 'children'),
    Input('select-view', 'value')
)
def render_content(tab):
    if tab == 'week-view':
        return (
            html.H2('Best players per week'),
            html.Div(id='weekly-selectors', children=[
                html.Div([
                    html.P('Season'),
                    dcc.Dropdown(options=df['season'].unique(), value='2024', id='season-dropdown', style={'width': '100px'}),
                ], style={'display': 'flex', 'flex-direction': 'row', 'gap': '5px'}),
                html.Div([
                    html.P('Week'),
                    dcc.Dropdown(options=df['week'].unique(), value='10', id='week-dropdown', style={'width': '100px'}),
                ], style={'display': 'flex', 'flex-direction': 'row', 'gap': '5px'}),
                html.Div([
                    html.P('Position'),
                    dcc.Dropdown(options=["All", "QB", "RB/WR/TE", "K"], value='All', id='position-dropdown', style={'width': '100px'}),
                ], style={'display': 'flex', 'flex-direction': 'row', 'gap': '5px'}),
                html.Div([
                    html.P('Order by'),
                    dcc.Dropdown(options=['fantasy_points', 'predicted_fantasy'], value='predicted_fantasy', id='order-by-dropdown', style={'width': '200px'})
                ], style={'display': 'flex', 'flex-direction': 'row', 'gap': '5px'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'gap': '20px'}),
            dcc.Graph(figure={}, id='scatter')
        )

@callback(
    Output(component_id='scatter', component_property='figure'),
    [Input(component_id='season-dropdown', component_property='value'),
     Input(component_id='week-dropdown', component_property='value'),
     Input(component_id='order-by-dropdown', component_property='value')]
)
def update_graphs(season_chosen, week_chosen, order_by_chosen):
    df_viz = (df[(df['week'] == int(week_chosen)) & (df['season'] == int(season_chosen))]
              .sort_values(by=order_by_chosen, ascending=False)
              .head(32))
    
    # Create two separate traces
    fig = go.Figure()
    
    # Add predicted_fantasy with error bars
    fig.add_trace(go.Scatter(
        x=df_viz['player_name'],
        y=df_viz['predicted_fantasy'],
        name='predicted_fantasy',
        mode='markers',
        # error_y=dict(
        #     type='data',
        #     array=df_viz['predicted_fantasy'] * 0.3,  # 70% error margin
        #     visible=True
        # )
    ))

    # Add actual fantasy points (no error bars)
    fig.add_trace(go.Scatter(
        x=df_viz['player_name'],
        y=df_viz['fantasy_points'],
        name='Actual Points',
        mode='markers'
    ))
    
    # Update layout
    fig.update_layout(
        title='Actual vs Predicted Fantasy Points',
        xaxis_title="Player Name",
        yaxis_title="Fantasy Points",
        legend_title="Point Type",
        showlegend=True,
        xaxis={'tickangle': 45}
    )
    
    return fig
    
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

