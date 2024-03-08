import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# Fetch historical stock price data
def fetch_stock_data(stock_code, start_date, end_date):
    ticker = yf.Ticker(stock_code)
    stock_data = ticker.history(start=start_date, end=end_date)
    return stock_data

# Prepare data for SVR
def prepare_data(stock_data):
    X = np.array(range(len(stock_data))).reshape(-1, 1)  # Use index as feature
    y = stock_data['Close'].values
    return X, y

# Split data into training and testing sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Train SVR model with hyperparameter tuning
def train_svr(X_train_scaled, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf', 'poly']
    }
    svr = SVR()
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_train_scaled, y_train)
    best_svr_model = grid_search.best_estimator_
    return best_svr_model

# Evaluate model
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Main function
def main():
    # Fetch historical data
    stock_code = 'AAPL'  # Example: Apple Inc.
    start_date = '2020-01-01'
    end_date = '2021-01-01'
    stock_data = fetch_stock_data(stock_code, start_date, end_date)
    
    # Prepare data
    X, y = prepare_data(stock_data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Train SVR model with hyperparameter tuning
    svr_model = train_svr(X_train_scaled, y_train)
    
    # Save the trained SVR model
    joblib.dump(svr_model,'model.py')
    
    # Evaluate model
    mae, mse, rmse = evaluate_model(svr_model, X_test_scaled, y_test)
    
    # Print evaluation metrics
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)

if __name__ == "__main__":
    main()