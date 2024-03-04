import dash 
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
import yfinance as yf

app = dash.Dash(__name__, external_stylesheets=['styles.css']) 
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
                html.H2("Input stock code", className="stock-code-heading"),
                # Stock code input 
                dcc.Input(id='stock-code', placeholder='Enter stock code...', type='text'),
                html.Button('Submit', id='submit-button', n_clicks=0)
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
            html.Button('Get Stock Price', id='stock-price-button', n_clicks=0),
            html.Button('Get Indicators', id='indicators-button', n_clicks=0),
            html.Button('Get Forecast', id='forecast-button', n_clicks=0)
        ],
        className="nav"
    ),
    html.Div(
        [
            html.Div(
                [  # Logo, Company Name
                    html.Img(src='company_logo.png', className='company-logo'),
                    html.H1(id='company-name', className='company-name')
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

@app.callback(
    Output('description', 'children'),
    Output('company-name', 'children'),
    Output('stock-price-plot', 'figure'),
    Output('indicator-plot', 'figure'),
    Output('forecast-plot', 'figure'),
    Input('submit-button', 'n_clicks'),
    State('stock-code', 'value'),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
    State('forecast-days', 'value')
)
def update_data(n_clicks, stock_code, start_date, end_date, forecast_days):
    if n_clicks > 0 and stock_code:
        # Fetch company data using yfinance library
        ticker = yf.Ticker(stock_code)
        company_data = ticker.info

        # Extract company name and description
        company_name = company_data.get('longName', 'Company Name')
        description = company_data.get('longBusinessSummary', 'Company Description')

        # Fetch stock data for the given date range
        stock_data = ticker.history(start=start_date, end=end_date)

        # Placeholder data for stock price plot
        stock_price_plot = {
            'data': [
                {'x': stock_data.index, 'y': stock_data['Close'], 'type': 'line', 'name': 'Stock Price'}
            ],
            'layout': {
                'title': 'Stock Price Plot'
            }
        }

        # Placeholder data for indicator plot
        indicator_plot = {
            'data': [
                {'x': [1, 2, 3, 4], 'y': [10, 15, 13, 17], 'type': 'bar', 'name': 'Indicator Data'}
            ],
            'layout': {
                'title': 'Indicator Plot'
            }
        }

        # Placeholder data for forecast plot
        forecast_plot = {
            'data': [
                {'x': [1, 2, 3, 4], 'y': [8, 6, 5, 9], 'type': 'line', 'name': 'Forecast Data'}
            ],
            'layout': {
                'title': 'Forecast Plot'
            }
        }

        return description, company_name, stock_price_plot, indicator_plot, forecast_plot
    else:
        return '', '', {}, {}, {}

if __name__ == '__main__':
    app.run_server(debug=True)
