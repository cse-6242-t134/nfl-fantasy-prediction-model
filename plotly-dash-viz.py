from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the data
df = pd.read_csv('rb_wr_predictions.csv')
# df = df[df["season"] == 2024]

# Initialize the app
app = Dash(__name__)

# Define the layout
app.layout = html.Div([
    # Your components go here
    html.H1('Fantasy Prediction Viz'),
    # dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    html.H2('Best players per week'),
    html.P('Season'),
    dcc.Dropdown(options=df['season'].unique(), value='2024', id='season-dropdown'),
    html.P('Week'),
    dcc.Dropdown(options=df['week'].unique(), value='10', id='week-dropdown'),
    html.P('Order by'),
    dcc.Dropdown(options=['fantasy_points', 'predictions'], value='predictions', id='order-by-dropdown'),
    # dcc.Graph(figure={}, id='bar'),
    dcc.Graph(figure={}, id='scatter')
    
    # Add more components...
])

# @callback(
#     # [Output(component_id='bar', component_property='figure'),
#     #  Output(component_id='scatter', component_property='figure')],
#     Output(component_id='scatter', component_property='figure'),
#     [Input(component_id='season-dropdown', component_property='value'),
#      Input(component_id='week-dropdown', component_property='value')]
# )
# def update_graphs(season_chosen, week_chosen):
#     df_viz = (df[(df['week'] == int(week_chosen)) & (df['season'] == int(season_chosen))]
#               .sort_values(by='predictions', ascending=False)
#               .head(32))
#     df_viz["error_plus"] = df_viz["predictions"] * .7
#     df_viz["error_minus"] = df_viz["predictions"] * .7
#     # fig1 = px.bar(df_viz, x='player_name', y='fantasy_points')
#     fig2 = px.scatter(df_viz, x='player_name', y=['fantasy_points', 'predictions'],
#                       error_y='error_plus', error_y_minus='error_minus')
#     # return fig1, fig2
#     return fig2

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
    
    # Add predictions with error bars
    fig.add_trace(go.Scatter(
        x=df_viz['player_name'],
        y=df_viz['predictions'],
        name='Predictions',
        mode='markers',
        # error_y=dict(
        #     type='data',
        #     array=df_viz['predictions'] * 0.7,  # 70% error margin
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

