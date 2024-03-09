
# Stock Price Analyzer with Prediction

This project is a single-page web application built with Dash (Python framework) that displays company information, stock plots, and predicted stock prices.



## Project Goal


- Enter a stock code.
- View company logo, registered name, and description.
- Visualize historical stock price data using a plot.
- Input a future date and get a predicted stock price based on a machine learning model.

## Getting Started

This project requires the following Python libraries:

- dash
- dash-html-components
- dash-core-components
- yfinance (for retrieving stock data)
- plotly (for generating visualizations)
- A machine learning library of your choice (e.g., scikit-learn, tensorflow) for prediction

    1. Install Libraries: Use pip install to install the required libraries.
    2. Run the App: Navigate to the project directory in your terminal and run python app.py.
    3. Access the App: Open http://127.0.0.1:8050/ in your web browser.

## Project Structure

The project consists of a single Python file (app.py) that defines the Dash app layout, functionalities, and callbacks.

- Layout: The layout is built using components from dash_html_components and dash_core_components. It includes elements for user input, company information display, and a container for the stock plot and prediction output.
- Data Fetching: A function utilizes yfinance to retrieve stock data based on the user-provided code. It handles potential errors and returns company information and historical prices.
- Stock Plots: A function generates a plotly chart for the retrieved historical stock prices.
- Stock Price Prediction: A machine learning model predicts the stock price for the user-specified date.
- Callbacks: Dash callbacks connect user interactions with data updates and plot generation. Entering a stock code and date triggers a callback that fetches data, generates plots, and updates the prediction based on the model.

## Customization

- **Machine Learning Model**: We have used the Support Vector Regression (SVR) module from the sklearn library to fetch stock prices for the last 60 days and spliting the dataset into 9:1 ratio for training and testing respectively.
- Used the rbf kernel in GridSearchCV for tuning the hyperparameters.
- **Styling**: Include a separate CSS file or use inline styles with Dash components to enhance the app's visual appearance.

## Further Enhancements

- Implement functionalities for searching companies by name.
- Integrate technical indicators for stock analysis.
- Allow users to visualize different timeframes for stock data.
