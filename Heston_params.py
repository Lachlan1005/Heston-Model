import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import base_params
import seaborn as sns

sns.set_style("dark")

def hist_vol_theta(tList, vols, period:str="5y"):
    """
    Estimate the historical volatility of the stock with the specified symbol over the specified period 
    """
    print("Estimating mean historical volatility (theta)...")
    print("Estimation Complete.")
    return (np.mean(vols**2)) #Variance is vols**2 (Heston wants variance)

def drift_mu(tList, prices, period:str="5y"):
    """
    Estimate the historical drift of the stock with the specified symbol over the specified period 
    by linear regression of the log prices
    """
    print("Estimating Drift Coefficient (mu)...")

    log_prices = np.log(prices)
    tList = tList.reshape(-1, 1)
    model = LinearRegression()
    model.fit(tList, log_prices)

    print("Estimation Complete.")
    return np.minimum(model.coef_[0] * 252, 0.15)

def vol_of_vol_sigma(tList, vols, period:str = "5y", lamb:float = 0.94):
    """
    Estimate the volatility of volatility of the stock with the specified symbol over the specified period 
    """
    print("Estimating Volatility of Volatiltiy (sigma)...")
    dVols = vols[1:] - vols[:-1]
    num = len(vols)
    aVol = 0
    for i, vol in enumerate(vols):
        aVol += lamb**i * vols[num - (i+1)]**2 
    aVol *= (1-lamb)

    print("Estimation Complete.")
    return np.sqrt(np.var(dVols, ddof = 1)/(aVol)) 

def correlation_rho(tList, vols, prices, sigma:float, period:str = "5y"):
    """
    Estimate the correlation between dU and dW (see tex notes) of 
    the stock with the specified symbol over the specified period 

    Make sure to use the sigma from vol_of_vol_sigma(), or a custom sigma that 
    you are extremely confident in
    """
    print("Estimating Correlation Coefficient (rho)...")
    rhos = []
    prices = prices[1:]

    dV = vols[1:] - vols[:-1]
    dS = prices[1:] - prices[:-1]
    root_V = np.sqrt(vols[1:])

    A = dV/ (sigma * root_V / np.sqrt(252))
    B = dS/(prices[1:] * np.sqrt(vols)[1:])

    print("Estimation Complete.")

    return np.corrcoef(A, B)[0,1]

def MVS_kappa(tList, vols, theta:float, period:str = "5y"):
    """
    Estimate the mean reverstion speed (kappa) of 
    the stock with the specified symbol over the specified period 

    Make sure to use the theta from hist_vol_theta(), or a custom theta that 
    you are extremely confident in
    """
    print("Estimating Mean Reversal Speed (kappa)...")
    eps = theta/1000 
    vols_dot = np.gradient(vols, 1/252, axis = 0) #1/252 needed to annualise, gradient just computes the ratio of differences, no anallyical continuity needed
    print("Estimation Complete.")
    return abs(np.median(vols_dot/ (theta - vols + eps))) 


def param_summary(symbol:str, period:str = "5y", lamb:float = .94):
    tList, vols = base_params.find_vols(symbol, period)
    tList2, prices = base_params.stock_closing_prices(symbol, period)

    theta = hist_vol_theta(tList, vols, period)
    mu = drift_mu(tList2, prices, period)
    sigma = vol_of_vol_sigma(tList, vols, period, lamb)
    rho = correlation_rho(tList, vols, prices, sigma, period)
    kappa = MVS_kappa(tList, vols, theta, period)
    print(f"Estimation Complete.\nHeston Parameters for {symbol}:\ntheta = {theta}, mu={mu}, sigma = {sigma}, rho = {rho}, kappa = {kappa}\n")
    return theta, mu, sigma, rho, kappa

def param_summary_from_list(tList_prices, prices, vols, period:str = "5y", lamb:float = .94):
    print("Initialising Heston parameters estimator...")
    tList2 = tList_prices 
    tList = tList_prices[1:len(vols)+1]

    theta = hist_vol_theta(tList, vols, period)
    mu = drift_mu(tList2, prices, period)
    sigma = vol_of_vol_sigma(tList, vols, period, lamb)
    rho = correlation_rho(tList, vols, prices, sigma, period)
    kappa = MVS_kappa(tList, vols, theta, period)
    return theta, mu, sigma, rho, kappa


#param_summary("SPY", "10y")
#param_summary("TSLA", "10y")
#param_summary("^HSI", "10y")
#param_summary("AAPL", "10y")
#param_summary("META", "10y")