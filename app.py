import dash 
from dash import dcc
from dash import html
from datetime import datetime as dt

app = dash.Dash(__name__)
server = app.server 

# Placeholder divs for item1 and item2
item1 = html.Div("Item 1 placeholder")
item2 = html.Div("Item 2 placeholder")

# Layout setup
app.layout = html.Div([
    html.Div(
        [
            html.P("Welcome to the Stock Dash App!", className="start"),
            html.Div([
                # Heading for stock code input
                html.H3("Input stock code", className="stock-code-heading"),
                # Stock code input 
                dcc.Input(id='stock-code', placeholder='Enter stock code...', type='text'),
                html.Button('Submit', id='input')
            ], style={'margin-bottom': '10px'}),  # Add margin to create space between components
            # Date range selector
            html.Div([
                dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=dt(1995, 8, 5),
                    max_date_allowed=dt.today(),
                    initial_visible_month=dt.today(),
                    end_date=dt.today().date()
                )
            ]),
            # Number of days of forecast input
            dcc.Input(id='forecast-days', placeholder='Enter forecast days...', type='number', style={'margin-top': '10px'}),
            # Buttons
            html.Button('Get Stock Price', className="hello", n_clicks=0),
            html.Button('Get Indicators', className="hello", n_clicks=0),
            html.Button('Get Forecast', className="hello", n_clicks=0)
        ],
        className="nav"
    ),
    html.Div(
        [
            html.Div(
                [  # Logo, Company Name
                    html.Img(src='company_logo.png', className='company-logo'),
                    html.H1('Company Name', className='company-name')
                ],
                className="header"
            ),
            html.Div(
                id="description", className="description-ticker"
            ),
            html.Div(
                [
                    # Stock price plot
                    dcc.Graph(id='stock-price-plot')
                ],
                id="graphs-content"
            ),
            html.Div(
                [
                    # Indicator plot
                    dcc.Graph(id='indicator-plot')
                ],
                id="main-content"
            ),
            html.Div(
                [
                    # Forecast plot
                    dcc.Graph(id='forecast-plot')
                ],
                id="forecast-content"
            )
        ],
        className="container"
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
