import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

sns.set_style("dark")

def stock_closing_prices(symbol:str, period:str="5y"):
    """
    Return the closing prices of the stock with the specified symbol over a period (in years) specified 
    by the input period

    Note the prices will always be returned in USD, regardless of the locale of the asset
    """
    print("Retrieving historical closing prices...")

    stock = yf.Ticker(symbol)
    data = stock.history(period= period) 
    
    closing_prices = np.array(data["Close"])
    tList = np.arange(0, len(closing_prices), 1)
    
    currency = stock.fast_info["currency"]
    if currency is None:
        currency = "unknown"
        closing_prices *= 0 
        print("unidentified currency detected")

    if currency != "USD" and currency != "unknown":
        pair = currency+"USD=X"
        forex = yf.Ticker(pair)
        rate = forex.fast_info["last_price"]
        if rate is None:
            rate = forex.history(period="1d")["Close"].iloc[-1]
        closing_prices = rate * closing_prices

    return tList, closing_prices

def forex_rate(input_currency:str):
    """
    Find the forex rate for converting the input currency to USD
    
    :param input_currency: symbol of input currency
    """
    if input_currency != "USD":
        pair = input_currency+"USD=X"
        forex = yf.Ticker(pair)
        rate = forex.fast_info["last_price"]
        if rate is None:
            rate = forex.history(period="1d")["Close"].iloc[-1]
    else:
        rate = 1
    
    return rate

def find_vols(symbol:str, period:str="5y", lamb = 0.94):
    """
    Compute the annualised EWMA volatility of the stock with the specified symbol over a period (in years) specified 
    by the input period using the closing prices

    Note that this returns volatiltiy data from t=1 to the end of the period and DOES NOT align with the prices timeframe (t=0 to end)
    this is because we take the difference here and that means the first item is empty
    """
    print("Retrieving historical volatility...")
    tList, closing_prices = stock_closing_prices(symbol, period)
    returns = (closing_prices[1:] - closing_prices[:-1])
    num = len(returns)
    log_returns = np.log(closing_prices[1:] / closing_prices[:-1])

    vol_squaredList = [log_returns[0]**2]


    for i, rogRet in enumerate(log_returns):
        if i==0:
            continue
        vol_squaredList.append(vol_squaredList[-1] * lamb + (1-lamb) * rogRet**2)
    
    volList = np.sqrt(np.array(vol_squaredList)) * np.sqrt(252)

    return tList[1:len(volList)+1], np.array(volList)

def find_vols_by_list(tList, closing_prices, lamb = 0.94):
    """
    Compute the annualised EWMA volatility of the stock with the specified symbol over a period (in years) specified 
    by the input period using the closing prices

    Note that this returns volatiltiy data from t=1 to the end of the period and DOES NOT align with the prices timeframe (t=0 to end)
    this is because we take the difference here and that means the first item is empty
    """
    print("Computing historical volatility...")
    returns = (closing_prices[1:] - closing_prices[:-1])
    num = len(returns)
    log_returns = np.log(closing_prices[1:] / closing_prices[:-1])

    vol_squaredList = [log_returns[0]**2]


    for i, rogRet in enumerate(log_returns):
        if i==0:
            continue
        vol_squaredList.append(vol_squaredList[-1] * lamb + (1-lamb) * rogRet**2)
    
    volList = np.sqrt(np.array(vol_squaredList)) * np.sqrt(252)

    return tList[1:len(volList)+1], np.array(volList)

def plot_sumamry(symbol:str, period:str="5y"):
    tList, closing_prices = stock_closing_prices(symbol, period)
    tListVol, vols = find_vols(symbol, period)

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8))

    axs[0].plot(tList, closing_prices)
    axs[0].set_xlabel("Time (trading days)")
    axs[0].set_ylabel("Closing Price (USD)")
    axs[0].set_title("Closing Prices vs Trading Days")

    axs[1].plot(tListVol, vols)
    axs[1].set_xlabel("Time (trading days)")
    axs[1].set_ylabel("Volatility")
    axs[1].set_title("Volatility vs Trading Days")

    print("start: ", tList[0], "&", tListVol[0])
    print("end: ", tList[-1], "&", tListVol[-1])


    plt.suptitle(f"Performance Summary of {symbol} Over a Period of {period}")
    plt.show()


