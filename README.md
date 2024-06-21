# Stock Predictor

## Data
The dataset used for training and analyzing various investment models includes historical stock market data from 10 different stocks. Additionally, it includes historical trading information from the S&P 500 index, known for having lower volatility compared to individual stocks. The dataset spans the past 10 years and encompasses six key variables: trading date, opening price, closing price, highest trading price, lowest trading price, and trading volume. 

To increase the accuracy of the predictive models, additional calculated columns have been incorporated, providing additional insight for making predictions. These columns include:
- **Date Components:** Day, Month, Year
- **Price Changes:** Percent Change, Montly Average Percent Change, Weekly Average Percent Change
- **Comparison Metrics:** Month Ago Price, Week Ago Price, Daily Price Variance Percentage
- **Time Indicators:** Quarter End Month
- **Financial Metrics:** Profit, Daily Multiplier

The Daily Price Variance is calculated as the percentage difference between the highest and lowest stock prices observed within a trading day. The Profit column is derived from the relationship between the day's opening and closing prices. As the target variable for the created machine learning algorithms, it is adjusted by one day, allowing the previous day's trading data to predict potential profitability for the following trading day. Similarly, the Daily Multiplier serves a comparable role in determining investment amounts predicted by the model. It is derived from the percentage change in stock price for the next trading day, as this is the day the predictions are generated. 

## Training Models

The machine learning algorithms will implement diverse investment strategies based on unique data features. These include: 'Close/Last', 'Volume', 'Open', 'High', 'Low', 'Pct_Change', '30_day_avg_pct_change', '5_day_avg_pct_change', 'month_ago_price', 'week_ago_price', 'Daily_Variance', and 'is_quarter_end'.

During data preprocessing, special consideration was given to ensure the models were trained without temporal bais. The dataset was split into training and testing subsets without regard to chronological order. This enables the models to generalize well across different subsets of dates, removing bias from time-dependent trends. Removing time-dependent bias empowers the models to make predictions effectively on any given trading date.

## Evaluating Models

The evaluation of the created models involves testing with various investment strategies. Each strategy has a varying degree of risk and popularity. To remain consistent, each strategy utilizes the same underlying machine learning model, trained and evaluated consistently on identical datasets to ensure fair comparisons of their success and failures. 

Regardless of the chosen investment strategy, the model initiates trading with $10,000 in capital. Depending on the market conditions and the investment strategy selected, the model may opt to buy, short sell, or maintain its position. Here is how each scenario is handled:
- **Buying:** The model invests a portion of its capital, applying a daily multiplier based on the forecasted potential returns
- **Short Selling:** In scenarios where the model predicts a decline, it shorts the stock, applying the inverse of the daily multiplier to the investment amount
- **Neither:** If the model decides not to trade on a given day, the capital earns interest equivalent to 2.5% annual rate, simulating holding the money in a bank or CD. 

The evaluation spans approximately 2.5 years (25% of the datasets) of historical trading data, comparing the model's performance against a benchmark strategy of long-term holding. This process outputs averaged results over multiple iterations specified during the execution of the `batch_test` method in the Stocks.py file. 

For long term analysis, as seen in `LongTermResults.pdf`, the machine learning algorithm makes predictions on nearly 20 years of stock market data with an initial investment of $10,000. This analysis is specifically focused on Investment Method 4 and Investment Method 6 with a 50% confidence level, as these strategies were proven most effective when compared to long-term buy-and-hold strategies.

## Investment Strategies

To comprehensively evaluate the model's performance, it is crucial to employ unique investment strategies that vary in risk. The strategies utilized in this study are detailed below:
- **Strategy 1:** Do not invest any capital; instead, earn a compounded annual interest of 2.5%
- **Strategy 2:** Hold investments long-term regardless of the model's predictions, utilizing the daily multiplier
- **Strategy 3:** Short-sell investments long-term, leveraging the inverse of the daily multiplier regardless of the model's predictions
- **Strategy 4:** Buy stocks when the model predicts profitability above a specified confidence threshold; otherwise, do nothing
- **Strategy 5:** Short-sell stocks when the model predicts profitability above a specified confidence threshold; otherwise, do nothing
- **Strategy 6:** Buy stocks when the model predicts profitability above a specified confidence threshold; otherwise, short-sell stocks
- **Strategy 7:** Short-sell stocks when the model predicts profitability above a specified confidence threshold; otherwise, buy stocks
- **Strategy 8:** Allocate capital based on the model's confidence probability: buy stocks with a portion of the funds and short-sell stocks with the remainder
- **Strategy 9:** Allocate capital based on the model's confidence probability: short-sell stocks with a portion of the funds and buy stocks with the remainder

## Running the Analysis

To execute the machine learning stock analysis, follow these steps:

- Open a new Windows PowerShell instance
- Navigate to the directory containing the `Stocks.py` file
- Execute the command `python Stocks.py [number of training/testing iterations]`

This command runs the Stocks.py file with the specified number of training and testing iterations passed as parameters to the methods. The terminal will display progress updates on the analysis as it proceeds.

To create the Machine Learning Stock Analysis document, navigate to the `Machine Learning Stock Analysis.Rmd` file in the directory. Open this is R studio, click run all, and click knit to generate an html version of this analysis.
