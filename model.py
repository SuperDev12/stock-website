from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import numpy as np
import joblib

# Function to fetch stock prices for the last 60 days
def fetch_stock_prices(stock_code):
    try:
        # Fetch historical stock price data
        stock_data = yf.download(stock_code, period="60d", group_by='ticker')
        return stock_data['Close'].values.reshape(-1, 1)
    except Exception as e:
        print(f"Error fetching data for {stock_code}: {e}")
        return None

# Function to split dataset into training and testing sets
def split_data(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        return X_train, X_test, y_train, y_test
    except ValueError as ve:
        print(f"Error splitting data: {ve}")
        return None, None, None, None

# Function to train SVR model with hyperparameter tuning
def train_svr(X_train, y_train):
    try:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
        }
        svr = SVR(kernel='rbf')
        grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
        grid_search.fit(X_train, y_train)
        best_svr_model = grid_search.best_estimator_
        return best_svr_model
    except Exception as e:
        print(f"Error training SVR model: {e}")
        return None

# Function to evaluate model's performance
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return mse, mae
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None, None

def main():
    # List of stock tickers representing companies in the American stock market
    stock_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'JPM', 'V', 'PG', 'INTC']

    for stock_code in stock_tickers:
        # Fetch stock prices for the last 60 days
        X = fetch_stock_prices(stock_code)
        if X is None:
            continue
        
        y = np.arange(len(X)).reshape(-1, 1)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(X, y)
        if X_train is None:
            continue

        # Train SVR model with hyperparameter tuning
        svr_model = train_svr(X_train, y_train)
        if svr_model is None:
            continue

        # Evaluate model's performance
        mse, mae = evaluate_model(svr_model, X_test, y_test)
        if mse is None or mae is None:
            continue

        print(f"Stock: {stock_code}, Mean Squared Error: {mse}, Mean Absolute Error: {mae}")

        # Save the trained model
        joblib.dump(svr_model, f'{stock_code}_svr_model.pkl')

if __name__ == "__main__":
    main()