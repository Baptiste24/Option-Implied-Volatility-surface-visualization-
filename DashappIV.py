#Install if needed

# pip install dash
# pip install pip install yahoo-fin
# pip install dash-bootstrap-components

from yahoo_fin import options
from yahoo_fin import stock_info as si
from datetime import datetime
from datetime import date as dt
from scipy.interpolate import griddata
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotly.offline import init_notebook_mode, iplot
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


# Create a Dash app
app = dash.Dash(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.QUARTZ])

"""
## App layout with LUX theme
app.layout = html.Div(
    [
        html.H1("Option Implied Volatility Surface"),
        html.Div(
            [
                html.Label("Enter a stock ticker:"),
                dcc.Input(id="ticker-input", type="text"),
                html.Button("Submit", id="submit-button"),
            ],
            style={"marginBottom": 20},
        ),
        html.Div(
            dcc.Graph(id="option-surface", config={'displayModeBar': False}),
            style={"display": "flex", "justify-content": "center", "background-color": "transparent", "width": "100vh", "height": "84vh"},
            
        ),
    ],
    style={"textAlign": "center", "maxWidth": "800px", "margin": "0 auto", "backgroundColor": "transparent"},
)


@app.callback(
    Output("option-surface", "figure"),
    Input("submit-button", "n_clicks"),
    Input("ticker-input", "value"),
)
"""

app.layout = html.Div(
    [
        html.H1("Option Implied Volatility Surface"),
        html.Div(
            [
                html.Label("Enter a stock ticker:"),
                dcc.Input(id="ticker-input", type="text"),
                html.Button("Submit", id="submit-button"),
            ],
            style={"marginBottom": 20},
        ),
        html.Div(
            dcc.Graph(id="option-surface"),
            style={"width": "100vh", "height": "70vh"},
            className="chart-container",
        ),
        html.Div(
            [
                html.Label("Implied Volatility plane in %:"),
                dcc.Slider(
                    id="z-threshold-slider",
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.5,
                    marks={i / 10: str(i / 10) for i in range(0, 11)},
                ),
            ],
            style={"width": "90%", "margin": "20px auto"},
        ),
    ],
    style={"textAlign": "center", "maxWidth": "800px", "margin": "0 auto"},
)

@app.callback(
    Output("option-surface", "figure"),
    Input("submit-button", "n_clicks"),
    Input("ticker-input", "value"),
    Input("z-threshold-slider", "value"),
)

#def update_plot(ticker):
def update_option_surface(n_clicks, ticker, z_threshold):
    if not n_clicks or not ticker:
        # Return an empty figure if the button has not been clicked or ticker is empty
        return go.Figure()
    
    # Get the option data for the specified ticker
    option_data_dict = {}
    dte = options.get_expiration_dates(ticker)  # list of dates
    
    for date in dte:
        option_data = options.get_options_chain(ticker, date)
        option_data_dict[date] = option_data
    
    # Create an empty DataFrame
    merged_df = pd.DataFrame()
    
    # Merge option data for each date into the DataFrame
    for date, data in option_data_dict.items():
        option_df = pd.concat([pd.DataFrame(data["calls"]), pd.DataFrame(data["puts"])])
        option_df["Date"] = date
        
        # Append to the merged DataFrame while avoiding duplicates
        merged_df = pd.concat([merged_df, option_df]).drop_duplicates()
    
    # Reset the index of the merged DataFrame
    merged_df = merged_df.reset_index(drop=True)
    
    # Get the underlying price
    underlying_price = si.get_live_price(ticker)
    
    # Calculate the arrays
    X = np.array(merged_df["Strike"] / underlying_price)
    
    # Convert today's date to a pandas Timestamp object
    today = pd.Timestamp(dt.today())
    
    # Calculate the difference in days between the "Date" column and today's date
    Y = np.array((pd.to_datetime(merged_df["Date"]) - today).dt.days)
    
    # Convert "impliedVolatility" column to numeric values
    # Convert "impliedVolatility" column to numeric values
    merged_df["Implied Volatility"] = merged_df["Implied Volatility"].apply(lambda x: float(x.replace(',', '').rstrip('%')) / 100)
    
    Z = np.array(merged_df["Implied Volatility"])
    
    # Interpolation
    # Define the grid on which you want to interpolate the data
    xi, yi = np.meshgrid(np.linspace(X.min(), X.max(), 100), np.linspace(Y.min(), Y.max(), 100))
    
    # Interpolate the data using griddata
    zi = griddata((X.ravel(), Y.ravel()), Z.ravel(), (xi, yi), method='cubic', fill_value=np.nanmin(Z))
    
    # Create the 3D surface plot using Plotly
    surface_trace = go.Surface(x=xi, y=yi, z=zi, colorscale='Jet')
    
      # Add the dynamic horizontal plane
    threshold_plane_trace = go.Surface(
        x=xi,
        y=yi,
        z=np.ones_like(zi) * z_threshold,
        opacity = 0.7,
        colorscale='Thermal',
        showscale=False,
            )



    data = [
        surface_trace,
        threshold_plane_trace,
        go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(color='black', size=3), name='Actual Data Points')
    ]

    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(x=0.9, y=0.9),
        scene=dict(
            xaxis=dict(title='Strike Price / Underlying Price'),
            yaxis=dict(title='Days to Expiration'),
            zaxis=dict(title='Implied Volatility'),
            camera=dict(eye=dict(x=-1.7, y=-1.7, z=0.5))           
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        

    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(showlegend=False)

    # Display the interactive 3D surface within the Jupyter Notebook
    # Return the figure object
    return fig

    

if __name__ =="__main__":
    app.run_server(debug=True)