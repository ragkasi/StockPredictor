# Call the yahoo finance API to download daily stock index prices
import yfinance as yf
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response, render_template

app = Flask(__name__)

# Home page with input form
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    # Get the ticker symbol from the request
    ticker = request.form['ticker']
    # Return the text and plot
    # Initialize ticker class to download price history for symbol "VOO" (S&P 500 Vanguard ETF)
    sp500 = yf.Ticker(ticker)
    # Query historical prices (from when the index was created)
    sp500 = sp500.history(period = "max")
    # Plot the closing price against the index
    sp500.plot.line(y="Close", use_index=True)
    # Save plot to a string in base64 format
    img = io.BytesIO()
    plt.savefig("plot.png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    # Return the plot URL
    return jsonify({'plot_url': plot_url})

@app.route('/generate_text', methods=['POST'])
def generate_text():
    # Get the ticker symbol from the request
    ticker = request.form['ticker']
    # Return the text and plot
    # Initialize ticker class to download price history for symbol "VOO" (S&P 500 Vanguard ETF)
    sp500 = yf.Ticker(ticker)
    # Query historical prices (from when the index was created)
    sp500 = sp500.history(period = "max")
    # Do not need Dividends, Capital Gains, and Stock Splits because this is an index fund
    del sp500["Dividends"]
    del sp500["Stock Splits"]
    # Take the close call and shift all the prices back by one day
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    # Target: WHat we try to predict with ML
    # Is tomorrow's price greater than today's price?
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    # Remove all data from before 1990 to reduce interference of fundamental shifts
    sp500 = sp500.loc["1990-01-01":].copy()
    # This model works by training a number of individual decision trees with randomized params
    # Results are averaged. Random forests are resistant to overfitting. Also runs quickly and
    # picks up nonlinear trends.
    from sklearn.ensemble import RandomForestClassifier
    # n_estimators:number of individual decision trees to train. Higher, the better.
    # min_samples_split: protects against overfitting, which can happen if decision tree
    # is built too deeply. Higher it is, the less accurate the model will be.
    # random_state: if we run the same model twice, the random numbers generated will be
    # ina predictable sequence.
    model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)
    # Split data into train and test set. Can't use cross validation with time set, since we will be
    # using future data to predict the past. This leakage leaks data into the model.
    # All rows except last 100 rows into training set. Last 100 rows in test set.
    train = sp500.iloc[:-100]
    test = sp500.iloc[-100:]
    # Do not use tomorrow or target column. Model can't know the future.
    predictors = ["Close", "Volume", "Open","High","Low"]
    # Train the model
    model.fit(train[predictors], train["Target"])
    # Measure how accurate the model is
    from sklearn.metrics import precision_score
    # When we said the target was 1, did the stock actually go up?
    preds = model.predict(test[predictors])
    # predictions in numpy array. Turn into pandas series.
    preds = pd.Series(preds, index = test.index)
    # Create more predictors. Rolling mean close prices in the last 2 days, last week,
    # last 3 months, last year, and last four years
    horizons = [2,5,60,250,1000]
    new_predictors = []
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        # Create close ratio columns and add to dataset
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"]/rolling_averages["Close"]
        # Look at the number of days in a time frame that the stock price went up
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_column, trend_column]

    sp500 = sp500.dropna()
    # Change some model params
    model = RandomForestClassifier(n_estimators = 200, min_samples_split=50, random_state=1)
    predictions = backtest(sp500, model, new_predictors)
    predictions["Predictions"].value_counts()
    score = precision_score(predictions["Target"], predictions["Predictions"]).item()
    # Generate text output
    text_output = f"Information for {ticker}: {score}"
    return jsonify({'text': text_output})

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    # Set custom threshold. Model goes up if 60% chance or more that it will go up.
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

# Want to have a certain amount of data to train models. Train model with 10 years of data.
def backtest(data,model,predictors,start=2500,step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

if __name__ == '__main__':
    app.run(debug=True)