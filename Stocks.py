# Implementing all the necessary Imports and Libraries needed for this project
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sb 
import random

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
  
import argparse
import warnings 
warnings.filterwarnings('ignore')

# Formats the data to be easier for the machine learning algorithms to analyze and adds in additional features to help with predictions
def format_data(path):

    # Read in the CSV File for the historical stock data
    df = pd.read_csv(path)

    # Format the Dates
    splitted = df['Date'].str.split('/', expand=True) 
    df['day'] = splitted[1].astype('int') 
    df['month'] = splitted[0].astype('int') 
    df['year'] = splitted[2].astype('int') 

    # Format the Numerical Values to remove $ and ,
    df['Close/Last'] = df['Close/Last'].replace('[\\$,]', '', regex=True).astype(float)
    df['Open'] = df['Open'].replace('[\\$,]', '', regex=True).astype(float)
    df['High'] = df['High'].replace('[\\$,]', '', regex=True).astype(float)
    df['Low'] = df['Low'].replace('[\\$,]', '', regex=True).astype(float)

    # Add in calculated columns that may help with training the model

    # Add in percentage changes and rolling averages
    df['Pct_Change'] = ((df['Close/Last']/df['Open'])-1)*100
    df['30_day_avg_pct_change'] = df['Pct_Change'].rolling(window=30).mean().shift(-30)
    df['30_day_avg_pct_change'].fillna(0, inplace=True)
    df['5_day_avg_pct_change'] = df['Pct_Change'].rolling(window=5).mean().shift(-5)
    df['5_day_avg_pct_change'].fillna(0, inplace=True)

    # Add in historical price data
    df['month_ago_price'] = df['Close/Last'].shift(-30)
    df['month_ago_price'] = df['month_ago_price'].fillna(0)
    df['week_ago_price'] = df['Close/Last'].shift(-5)
    df['week_ago_price'] = df['week_ago_price'].fillna(0)

    # Add in daily stock price fluctuation metric
    df['Daily_Variance'] = ((df['High'] - df['Low']) / df['Low']) * 100

    # Add in if the month has an earnings report
    df['is_quarter_end'] = np.where(df['month']%3==0,1,0) 
    df.head()

    # Return TRUE or FALSE if the stock was profitable on that trading day
    df['Profit'] = df['Pct_Change'] >= 0
    df['Profit'] = df['Profit'].shift(1)
    df['Profit'] = df['Profit'].fillna(False)

    # Create the multiplier column and shifts it to reflect future data
    df['Multiplier'] = 1 + df['Pct_Change'] / 100
    df['Multiplier'] = df['Multiplier'].shift(1)

    # Removes the first and last thirty rows
    df = df[1:-30]

    # Remove the Date column
    df.drop(columns=['Date'], inplace=True)

    # Return the new dataframe
    return df

# This method creates multiple graphs with valuable information about the chosen Stock
def graph(list_of_multipliers, initial_investment, output_file):
    
    # Creating Stock List for Graph Labeling
    sl = ['AAPL', 'AMD', 'AMZN', 'CSCO', 'META', 'MSFT', 'NFLX', 'QCOM', 'SBUX', 'TSLA', 'SPX']

    # Colors for the combined plot
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(sl))]

    # Dictionary to store combined data
    combined_data = {}

    # Create a PdfPages object to save multiple plots in a single PDF file
    with PdfPages(output_file) as pdf:

        # Iterate through each list of multipliers
        for idx, multipliers in enumerate(list_of_multipliers):

            # Lists to store the days and investment values
            days = []
            investment_values = []

            # Calculate the investment value for each day
            current_investment = initial_investment
            for day, multiplier in enumerate(multipliers, start=1):
                current_investment *= multiplier
                days.append(day)
                investment_values.append(current_investment)

            # Create a DataFrame
            df = pd.DataFrame({
                'Days': days,
                'Investment': investment_values
            })

            # Store data for combined plot
            combined_data[sl[idx]] = df

            # Plot the line graph
            plt.figure(figsize=(10, 6))
            plt.plot(df['Days'], df['Investment'])
            plt.title(f'Investment Over Time ({sl[idx]})')
            plt.xlabel('Days')
            plt.ylabel('Investment Value')
            plt.grid(True)

            # Save the plot to the PDF file
            pdf.savefig()
            plt.close()

        # Create the combined plot
        plt.figure(figsize=(10, 6))
        for idx, (stock, df) in enumerate(combined_data.items()):
            plt.plot(df['Days'], df['Investment'], label=stock, color=colors[idx])

        plt.title('Combined Investment Over Time')
        plt.xlabel('Days')
        plt.ylabel('Investment Value')
        plt.legend()
        plt.grid(True)

        # Save the combined plot to the PDF file
        pdf.savefig()
        plt.close()

    # Return the file path
    return output_file

# Function to format y-axis tick labels without scientific notation
def format_y_ticks(value, pos):
    return f'{value:.0f}'  # Format the value without any decimal places

# This method creates multiple graphs with valuable information about the chosen Stock
def graph2(list_of_multipliers, initial_investment, output_file):
    
    # Creating Stock List for Graph Labeling
    sl = ['AAPL', 'AMD', 'AMZN', 'CSCO', 'META', 'MSFT', 'NFLX', 'QCOM', 'SBUX', 'TSLA', 'SPX']

    # Colors for the plots
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(list_of_multipliers[0]))]  # Use the length of the first inner list

    # Dictionary to store data for individual plots
    combined_data = {stock: [] for stock in sl}

    # Create a PdfPages object to save multiple plots in a single PDF file
    with PdfPages(output_file) as pdf:

        # Iterate through each stock
        for stock_idx, stock in enumerate(sl):
            plt.figure(figsize=(10, 6))

            # Iterate through each method for the current stock
            for method_idx, multipliers in enumerate(list_of_multipliers[stock_idx]):
                
                # Lists to store the days and investment values for the current method
                days = []
                investment_values = []

                # Calculate the investment value for each day
                current_investment = initial_investment
                for day, multiplier in enumerate(multipliers, start=1):
                    current_investment *= multiplier
                    days.append(day)
                    investment_values.append(current_investment)

                # Create a DataFrame for the current method
                df = pd.DataFrame({
                    'Days': days,
                    'Investment': investment_values
                })

                # Store data for combined plot
                combined_data[stock].append(df)

                # Determine label based on method index
                if method_idx == 0:
                    label = 'Buy-And-Hold'
                elif method_idx == 1:
                    label = 'Investment Method 4'
                elif method_idx == 2:
                    label = 'Investment Method 6'
                else:
                    label = f'Method {method_idx + 1}'

                # Plot the line graph for the current method with updated label
                plt.plot(df['Days'], df['Investment'], label=label, color=colors[method_idx])

            plt.title(f'Investment Over Time ({stock})')
            plt.xlabel('Days')
            plt.ylabel('Investment Value')
            plt.legend()
            plt.grid(True)

            # Format y-axis tick labels to avoid scientific notation
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_y_ticks))

            # Save the plot to the PDF file
            pdf.savefig()
            plt.close()

    # Return the file path
    return output_file

# Fits the model according to the random_state specified in the calling of the function
def fit(df, random_state=None):
    
    # Define the columns to base the prediction off of
    xcols = df[['Close/Last', 'Volume', 'Open', 'High', 'Low',
       'Pct_Change', '30_day_avg_pct_change', '5_day_avg_pct_change',
       'month_ago_price', 'week_ago_price', 'Daily_Variance', 'is_quarter_end']] 
    ycol = df['Profit'] 

    # Format the prediction columns correctly
    scaler = StandardScaler() 
    xcols_scaled = scaler.fit_transform(xcols)
    
    # Train the model
    train_x, test_x, train_y, test_y, train_idx, test_idx = train_test_split(xcols_scaled, ycol, ycol.index, test_size=0.25, random_state=random_state)

    # Define the model
    model = LogisticRegression()

    # Fit and evaluate the model
    model.fit(train_x, train_y)

    # Return the model
    return model, test_idx, test_x

def method1(df, investment_logistic_amt, investment_amt, rs, n):
    # Create the overall average variables to store totals and return
    il = 0
    i = 0
    mult = []

    # Test, Train, and Evaluate the model n times
    for a in range(n):
        
        # Re-train the model and reset the investment values for each iteration
        model, test_idx, test_x = fit(df, random_state=rs[a])
        investment = investment_amt
        investment_logistic = investment_logistic_amt
        iteration_mult = []

        # Iterate through each day and update the investment for the baseline
        for idx in reversed(test_idx):
            investment *= df.loc[idx, 'Multiplier']
            iteration_mult.append(df.loc[idx, 'Multiplier'])

        # Append the iteration's multipliers to mult
        mult.append(iteration_mult)

        # Evaluate the investment with investment strategy 1
        for idx in reversed(test_idx):
            investment_logistic = investment_logistic * 1.00009917204
        
        # Add to the total variables for each iteration
        il += investment_logistic
        i += investment
    
    # Calculate the average multiplier for each day over all iterations
    s_mult = []
    for day_idx in range(len(mult[0])):  
        day_total = sum(mult[a][day_idx] for a in range(n))  
        day_average = day_total / n  
        s_mult.append(day_average)

    # Formatting the output so that it can be used in the creation of a dataframe and returning it
    results = ['', 1, '', round(il/n, 2), round(i/n, 2)]
    return s_mult, results

# Method 2: Hold Long Term
def method2(df, investment_logistic_amt, investment_amt, rs, n):

    # Create the overall average variables to store totals and return
    il = 0
    i = 0

    # Test, Train, and Evaluate the model n times
    for a in range(n):
        
        # Re-train the model and reset the investment values for each iteration
        model, test_idx, test_x = fit(df, random_state=rs[a])
        investment = investment_amt
        investment_logistic = investment_logistic_amt

        # Iterate through each day and update the investment for the baseline
        for idx in reversed(test_idx):
            investment *= df.loc[idx, 'Multiplier']

        # Evaluate the investment with investment strategy 2
        for idx in reversed(test_idx):
            investment_logistic *= df.loc[idx, 'Multiplier']

        # Add to the total variables for each iteration
        il += investment_logistic
        i += investment

    # Formatting the output so that it can be used in the creation of a dataframe and returning it
    results = ['', 2, '', round(il/n, 2), round(i/n,2)]
    return results

# Method 3: Short Long Term
def method3(df, investment_logistic_amt, investment_amt, rs, n):

    # Create the overall average variables to store totals and return
    il = 0
    i = 0

    # Test, Train, and Evaluate the model n times
    for a in range(n):

        # Re-train the model and reset the investment values for each iteration
        model, test_idx, test_x = fit(df, random_state=rs[a])
        investment = investment_amt
        investment_logistic = investment_logistic_amt

        # Iterate through each day and update the investment for the baseline
        for idx in reversed(test_idx):
            investment *= df.loc[idx, 'Multiplier']

        # Evaluate the investment with investment strategy 3
        for idx in reversed(test_idx):
            investment_logistic *= (1 / df.loc[idx, 'Multiplier'])

        # Add to the total variables for each iteration
        il += investment_logistic
        i += investment

    # Formatting the output so that it can be used in the creation of a dataframe and returning it
    results = ['', 3, '', round(il/n, 2), round(i/n,2)]
    return results

# Method 4: Buy the stock when there is a certain level of confidence that there will be PROFIT
def method4(df, investment_logistic_amt, investment_amt, rs, n):
 
    # Creating the Results dictionary and confidence levels
    res = {}
    results = []
    conf = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    for c in conf:
        res[c] = [0, 0]

    # Test, Train, and Evaluate the model n times
    for a in range(n):
        
        # Re-train the model and reset the investment values for each iteration
        investment = investment_amt
        investment_logistic = investment_logistic_amt
        model, test_idx, test_x = fit(df, random_state=rs[a])

        # Iterate through each day and update the investment for the baseline
        for idx in reversed(test_idx):
            investment *= df.loc[idx, 'Multiplier']

        # Evaluate the investment based on Logistic Regression model's predictions
        test_probabilities = model.predict_proba(test_x)[:, 1]

        # Creating a Loop to go through different levels of confidence and add results to result dictionary
        for c in conf:
            temp_investment_logistic = investment_logistic
            
            for idx, prob in zip(reversed(test_idx), reversed(test_probabilities)):
                
                # Calculate investments based on probabilities
                if prob >= c:
                    temp_investment_logistic = temp_investment_logistic * (df.loc[idx, 'Multiplier'])
                else:
                    temp_investment_logistic = temp_investment_logistic * 1.00009917204
            
            # Store the results for this confidence level
            r = [temp_investment_logistic, investment]
            
            # Add to the totals for each iteration
            res[c][0] += r[0]
            res[c][1] += r[1]
    
    # Calculate averages for each confidence level
    for c in conf:
        res[c][0] = round(res[c][0]/n, 2)
        res[c][1] = round(res[c][1]/n, 2)
        results.append(['', 4, c, res[c][0], res[c][1]])
    
    # Formatting the output so that it can be used in the creation of a dataframe and returning it
    return results

# Method 4.1: Buy the stock when there is a certain level of confidence that there will be PROFIT
def method4_with_investment_update(df, investment_logistic_amt, investment_amt, rs, n):
    
    # Initialize results dictionary and confidence level
    res = {}
    results = []
    conf = 0.5  # Fixed confidence level of 0.5
    
    # Initialize lists for storing multipliers
    baseline_multipliers = []
    method_multipliers = []

    # Test, Train, and Evaluate the model n times
    for a in range(n):
        
        # Re-train the model and reset the investment values for each iteration
        if a == 0:
            # First iteration uses the provided investment amounts
            initial_investment_logistic = investment_logistic_amt
            initial_investment = investment_amt
        else:
            # Subsequent iterations use the results from the previous iteration
            initial_investment_logistic = res[conf][0]
            initial_investment = res[conf][1]

        investment_logistic = initial_investment_logistic
        investment = initial_investment
        
        model, test_idx, test_x = fit(df, random_state=rs[a])

        # Iterate through each day and update the investment for the baseline
        for idx in reversed(test_idx):
            baseline_multipliers.append(df.loc[idx, 'Multiplier'])
            investment *= df.loc[idx, 'Multiplier']

        # Evaluate the investment based on Logistic Regression model's predictions
        test_probabilities = model.predict_proba(test_x)[:, 1]

        # Calculate investments based on probabilities and store daily multipliers
        for idx, prob in zip(reversed(test_idx), reversed(test_probabilities)):
            if prob >= conf:
                method_multipliers.append(df.loc[idx, 'Multiplier'])
                investment_logistic *= df.loc[idx, 'Multiplier']
            else:
                method_multipliers.append(1.00009917204)  # Adjust this multiplier as needed
                investment_logistic *= 1.00009917204

        # Store the results for this confidence level
        res[conf] = [investment_logistic, investment]

        # Append results formatted for output
        results.append(['', 4, conf, round(res[conf][0], 2), round(res[conf][1], 2)])
    
    # Return lists of multipliers and results separately
    return baseline_multipliers, method_multipliers, results

# Method 5: Short the stock when there is a certain level of confidence that there will be a PROFIT
def method5(df, investment_logistic_amt, investment_amt, rs, n):
 
    # Creating the Results dictionary and confidence levels
    res = {}
    results = []
    conf = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    for c in conf:
        res[c] = [0, 0]

    # Test, Train, and Evaluate the model n times
    for a in range(n):

        # Re-train the model and reset the investment values for each iteration
        investment = investment_amt
        investment_logistic = investment_logistic_amt
        model, test_idx, test_x = fit(df, random_state=rs[a])

        # Iterate through each day and update the investment for the baseline
        for idx in reversed(test_idx):
            investment *= df.loc[idx, 'Multiplier']

        # Evaluate the investment based on Logistic Regression model's predictions
        test_probabilities = model.predict_proba(test_x)[:, 1]

        # Creating a Loop to go through different levels of confidence and add results to result dictionary
        for c in conf:
            temp_investment_logistic = investment_logistic
            
            for idx, prob in zip(reversed(test_idx), reversed(test_probabilities)):
                
                # Calculate investments based on probabilities
                if prob >= c:
                    temp_investment_logistic = temp_investment_logistic * (1/df.loc[idx, 'Multiplier'])
                else:
                    temp_investment_logistic = temp_investment_logistic * 1.00009917204
            
            # Store the results for this confidence level
            r = [temp_investment_logistic, investment]

            # Add to the totals for each iteration
            res[c][0] += r[0]
            res[c][1] += r[1]
    
    # Calculate averages for each confidence level
    for c in conf:
        res[c][0] = round(res[c][0]/n, 2)
        res[c][1] = round(res[c][1]/n, 2)
        results.append(['', 5, c, res[c][0], res[c][1]])
    
    # Formatting the output so that it can be used in the creation of a dataframe and returning it
    return results

# Strategy 6 - Invest when model is positive, short when model thinks it will go down
def method6(df, investment_logistic_amt, investment_amt, rs, n):
    
    # Creating the Results dictionary and confidence levels
    res = {}
    results = []
    conf = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    for c in conf:
        res[c] = [0, 0]

    # Test, Train, and Evaluate the model n times
    for a in range(n):
        
        # Re-train the model and reset the investment values for each iteration
        investment = investment_amt
        investment_logistic = investment_logistic_amt
        model, test_idx, test_x = fit(df, random_state=rs[a])

        # Iterate through each day and update the investment for the baseline
        for idx in reversed(test_idx):
            investment *= df.loc[idx, 'Multiplier']

        # Evaluate the investment based on Logistic Regression model's predictions
        test_probabilities = model.predict_proba(test_x)[:, 1]

        # Creating a Loop to go through different levels of confidence and add results to result dictionary
        for c in conf:
            temp_investment_logistic = investment_logistic
            
            for idx, prob in zip(reversed(test_idx), reversed(test_probabilities)):
                
                # Calculate investments based on probabilities
                if prob >= c:
                    temp_investment_logistic = temp_investment_logistic * df.loc[idx, 'Multiplier']
                else:
                    temp_investment_logistic = temp_investment_logistic * (1/df.loc[idx, 'Multiplier'])
            
            # Store the results for this confidence level
            r = [temp_investment_logistic, investment]

            # Add to the totals for each iteration
            res[c][0] += r[0]
            res[c][1] += r[1]
    
    # Calculate averages for each confidence level
    for c in conf:
        res[c][0] = round(res[c][0]/n, 2)
        res[c][1] = round(res[c][1]/n, 2)
        results.append(['', 6, c, res[c][0], res[c][1]])
    
    # Formatting the output so that it can be used in the creation of a dataframe and returning it
    return results

# Strategy 6.1 - Invest when model is positive, short when model thinks it will go down
def method6_with_investment_update(df, investment_logistic_amt, investment_amt, rs, n):
    
    # Initialize results dictionary and confidence level
    res = {}
    results = []
    conf = 0.5  # Fixed confidence level of 0.5
    
    # Initialize lists for storing multipliers
    method_multipliers = []

    # Test, Train, and Evaluate the model n times
    for a in range(n):
        
        # Re-train the model and reset the investment values for each iteration
        if a == 0:
            # First iteration uses the provided investment amounts
            initial_investment_logistic = investment_logistic_amt
            initial_investment = investment_amt
        else:
            # Subsequent iterations use the results from the previous iteration
            initial_investment_logistic = res[conf][0]
            initial_investment = res[conf][1]

        investment_logistic = initial_investment_logistic
        investment = initial_investment
        
        model, test_idx, test_x = fit(df, random_state=rs[a])

        # Iterate through each day and update the investment for the baseline
        for idx in reversed(test_idx):
            investment *= df.loc[idx, 'Multiplier']

        # Evaluate the investment based on Logistic Regression model's predictions
        test_probabilities = model.predict_proba(test_x)[:, 1]

        # Calculate investments based on probabilities and store daily multipliers
        for idx, prob in zip(reversed(test_idx), reversed(test_probabilities)):
            if prob >= conf:
                method_multipliers.append(df.loc[idx, 'Multiplier'])
                investment_logistic *= df.loc[idx, 'Multiplier']
            else:
                method_multipliers.append(1 / df.loc[idx, 'Multiplier'])
                investment_logistic *= (1 / df.loc[idx, 'Multiplier'])

        # Store the results for this confidence level
        res[conf] = [investment_logistic, investment]

        # Append results formatted for output
        results.append(['', 6, conf, round(res[conf][0], 2), round(res[conf][1], 2)])
    
    # Return lists of multipliers and results separately
    return method_multipliers, results

# Strategy 7 - Short when model is positive, buy when model thinks it will go down
def method7(df, investment_logistic_amt, investment_amt, rs, n):
    
    # Creating the Results dictionary and confidence levels
    res = {}
    results = []
    conf = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    for c in conf:
        res[c] = [0, 0]

    # Test, Train, and Evaluate the model n times
    for a in range(n):

        # Re-train the model and reset the investment values for each iteration
        investment = investment_amt
        investment_logistic = investment_logistic_amt
        model, test_idx, test_x = fit(df, random_state=rs[a])

        # Iterate through each day and update the investment for the baseline
        for idx in reversed(test_idx):
            investment *= df.loc[idx, 'Multiplier']

        # Evaluate the investment based on Logistic Regression model's predictions
        test_probabilities = model.predict_proba(test_x)[:, 1]

        # Creating a Loop to go through different levels of confidence and add results to result dictionary
        for c in conf:
            temp_investment_logistic = investment_logistic
            
            for idx, prob in zip(reversed(test_idx), reversed(test_probabilities)):
                
                # Calculate investments based on probabilities
                if prob <= c:
                    temp_investment_logistic = temp_investment_logistic * df.loc[idx, 'Multiplier']
                else:
                    temp_investment_logistic = temp_investment_logistic * (1/df.loc[idx, 'Multiplier'])
            
            # Store the results for this confidence level
            r = [temp_investment_logistic, investment]

            # Add to the totals for each iteration
            res[c][0] += r[0]
            res[c][1] += r[1]
    
    # Calculate averages for each confidence level
    for c in conf:
        res[c][0] = round(res[c][0]/n, 2)
        res[c][1] = round(res[c][1]/n, 2)
        results.append(['', 7, c, res[c][0], res[c][1]])
    
    # Formatting the output so that it can be used in the creation of a dataframe and returning it
    return results

# Strategy 8 - Buy percentage based on confidence, short with the rest of the money
def method8(df, investment_logistic_amt, investment_amt, rs, n):

    # Create the overall average variables to store totals and return
    il = 0
    i = 0

    # Test, Train, and Evaluate the model n times
    for a in range(n):

        # Re-train the model and reset the investment values for each iteration
        model, test_idx, test_x = fit(df, random_state=rs[a])
        investment = investment_amt
        investment_logistic = investment_logistic_amt
        
        # Iterate through each day and update the investment for the baseline
        for idx in reversed(test_idx):
            investment *= df.loc[idx, 'Multiplier']

        # Evaluate the investment based on Logistic Regression model's predictions
        test_probabilities = model.predict_proba(test_x)[:, 1]
    
        # Calculate the investment based on predicted probabilities
        for idx, prob in zip(reversed(test_idx), reversed(test_probabilities)):
            
            # Calculate investments based on probabilities
            investment_profit = investment_logistic * (prob * df.loc[idx, 'Multiplier'])
            investment_short = investment_logistic * ((1 - prob) * (1 / df.loc[idx, 'Multiplier']))

        # Combine investments
        investment_logistic = investment_profit + investment_short

        # Add to the total variables for each iteration
        il += investment_logistic
        i += investment

    # Formatting the output so that it can be used in the creation of a dataframe and returning it
    results = ['', 8, '', round(il/n, 2), round(i/n,2)]
    return results

#Strategy 9 - Short percentage based on confidence, buy with the rest of the money
def method9(df, investment_logistic_amt, investment_amt, rs, n):
    
    # Create the overall average variables to store totals and return
    il = 0
    i = 0

    # Test, Train, and Evaluate the model n times
    for a in range(n):
        
        # Re-train the model and reset the investment values for each iteration
        model, test_idx, test_x = fit(df, random_state=rs[a])
        investment = investment_amt
        investment_logistic = investment_logistic_amt
        
        # Iterate through each day and update the investment for the baseline
        for idx in reversed(test_idx):
            investment *= df.loc[idx, 'Multiplier']

        # Evaluate the investment based on Logistic Regression model's predictions
        test_probabilities = model.predict_proba(test_x)[:, 1]
    
        # Calculate the investment based on predicted probabilities
        for idx, prob in zip(reversed(test_idx), reversed(test_probabilities)):
            
            # Calculate investments based on probabilities
            investment_short = investment_logistic * (prob * (1 / df.loc[idx, 'Multiplier']))
            investment_profit = investment_logistic * ((1 - prob) * (df.loc[idx, 'Multiplier']))

        # Combine investments
        investment_logistic = investment_profit + investment_short

        # Add to the total variables for each iteration
        il += investment_logistic
        i += investment

    # Formatting the output so that it can be used in the creation of a dataframe and returning it
    results = ['', 9, '', round(il/n, 2), round(i/n,2)]
    return results

# This function tests every model at once and outputs the results as a CSV
def batch_test(df_list, num_iters = 10):
    output_df = []
    output_df2 = []
    df_num = 0
    mult_list = []
    mult_list2 = []

    # This picks random state variables to keep the testing consistent between models
    rs = random.sample(range(5000), num_iters)
    if len(rs) >= 10:
        rs2 = random.sample(rs, 10)
    else:
        rs2 = random.sample(range(5000), 10)
    
    # Loops through every investment method for every stock and adds the data to the dataframe data
    for df in df_list:
        
        # Add the output lists from the investment methods to the list storing the dataframe data
        m, d = method1(df, 10000, 10000, rs, num_iters)
        mult_list.append(m)
        output_df.append(d)
        output_df.append(method2(df, 10000, 10000, rs, num_iters))
        output_df.append(method3(df, 10000, 10000, rs, num_iters))
        output_df.extend(method4(df, 10000, 10000, rs, num_iters))
        output_df.extend(method5(df, 10000, 10000, rs, num_iters))
        output_df.extend(method6(df, 10000, 10000, rs, num_iters))
        output_df.extend(method7(df, 10000, 10000, rs, num_iters))
        output_df.append(method8(df, 10000, 10000, rs, num_iters))
        output_df.append(method9(df, 10000, 10000, rs, num_iters))
        a, b, c = method4_with_investment_update(df, 10000, 10000, rs2, 10)
        f, g = method6_with_investment_update(df, 10000, 10000, rs2, 10)
        t_l = [a, b, f]
        mult_list2.append(t_l)
        output_df2.extend(c)
        output_df2.extend(g)
        
        # Status Update for User in Terminal
        df_num+=1
        print(f"Stock {df_num}/11 Processing Complete!")
        
    #Create a dataframe with the investment method outputs
    odf = pd.DataFrame(output_df, columns=['Stock', 'Investment_Method', 'Confidence', 'Calculated_Investment', 'Basic_Investment'])
    odf2 = pd.DataFrame(output_df2, columns = ['Stock', 'Investment_Method', 'Confidence', 'Calculated_Investment', 'Basic_Investment'])

    #Output the results dataframe as a Human-Readable CSV File for Data Analysis
    odf.to_csv('AverageResults.csv', index=False)
    odf2.to_csv('LongTermResults.csv', index=False)
    
    return mult_list, mult_list2

# This is the code used to run the file from the terminal
def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Run batch test for stock data.')
    parser.add_argument('iterations', type=int, help='Number of iterations for batch testing')

    # Parse the command line arguments
    args = parser.parse_args()

    # List of file paths
    path_list = ['Stocks/AAPL.csv', 'Stocks/AMD.csv', 'Stocks/AMZN.csv', 'Stocks/CSCO.csv', 'Stocks/META.csv', 'Stocks/MSFT.csv', 'Stocks/NFLX.csv', 'Stocks/QCOM.csv', 'Stocks/SBUX.csv', 'Stocks/TSLA.csv', 'Stocks/SPX.csv']

    # Load and format the data
    df_list = [format_data(path) for path in path_list]

    # Print the start message
    print('Output CSV Generation Started!')
    
    # Run the batch test
    gl, gl2 = batch_test(df_list, args.iterations)
    
    # Print the completion message
    print('Output CSV Generation Completed! File can be found as AverageResults.csv and LongTermResults.csv.')

    # Create a graph file
    graph(gl, 10000, 'AverageResults.pdf')
    graph2(gl2, 10000, 'LongTermResults.pdf')

    # Print the completion message for the graph file
    print('Investment Graph PDF Generation Completed! File can be found as AverageResults.pdf and LongTermResults.pdf.')

if __name__ == '__main__':
    main()