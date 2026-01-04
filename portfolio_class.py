import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
import pandas as pd
import base_params, Heston_params, Heston_Engine, profit_loss, shutil, os
import seaborn as sns

plt.style.use("dark_background")

Saved_Results_Name = "Saved_Results"

# Get the full path relative to the script
Saved_Results_Path = Path(__file__).parent / Saved_Results_Name

class portfolio:
    def __init__(self, assets:dict, name:str):
        """
        Initialise the portfolio with name and assets
        
        :param assets: Dictionary of assets {asset_1_symbol:number_of_shares_1, asset_2_symbol:number_of_shares_2, ...}
        :type assets: dict
        :param name: Name or portfolio
        :type name: str
        """
        print("Initialising portfolio...")
        self.name = name 
        self.assets = assets 
        self.asset_vals = {}
        self.total_val = 0
        for key in self.assets:
            stock = yf.Ticker(key)
            cur_price = float(stock.history(period="1d")["Close"].iloc[-1])
            currency = stock.fast_info["currency"]
            if currency is None:
                currency = "unknown"
                cur_price = 0.0 
            if currency != "USD" and currency != "unknown":
                pair = currency+"USD=X"
                data = yf.Ticker(pair)
                rate = data.fast_info["last_price"]
                rate = float(data.history(period="1d")["Close"].iloc[-1])
                cur_price *= rate
            
            self.asset_vals[key] = cur_price*self.assets[key]
            self.total_val += self.asset_vals[key] 

    def __repr__(self):
        return f"Summary for {self.name}\n"+30*"-"+f"\nCurrent Holdings: \n{self.assets}\n\nTotal Value: {self.total_val}\n\nValuation Breakdown:{self.asset_vals}"
    
    def individual_heston_analysis(self, horizon:float, ref_data_period:str, num_paths, start_at_theta:bool, plot:bool=True, plot_nth_path:int=10):
        Heston_Engine.clear_results()
        num_paths = int(num_paths)
        for key in self.assets:
            Heston_Engine.heston_for_stock(key, horizon, ref_data_period, num_paths, start_at_theta, plot, plot_nth_path)
        print("Analysis Complete. Check Saved_Results for analysis results. ")
    
    def individual_PnL_analysis(self, horizon:float, ref_data_period:str, num_paths, start_at_theta:bool, plot:bool=True):
        Heston_Engine.clear_results()
        num_paths = int(num_paths)
        reports = [f"\n\n=== PnL Reports for {self.name} ({horizon} Year Horizon)===\n"]
        for key in self.assets: 
            report = profit_loss.PnL(key, horizon, ref_data_period, num_paths, start_at_theta, True)
            reports.append(report)
        
        for rep in reports:
            print(rep)
        return reports
    
    def get_history(self, ref_data_period:str, plot:bool = False):
        """
        Get the price and volatility history of the portfolio over a period specified by ref_data_period
        
        :param ref_data_period: Reference period
        :type ref_data_period: str
        """
        print("Computing portfolio history...")
        Heston_Engine.clear_results()
        symbols = []
        for key in self.assets:
            symbols.append(key)
        prices = yf.download(symbols, period = ref_data_period)["Close"]
        prices = prices[self.assets.keys()]
        prices = prices.apply(pd.to_numeric, errors='coerce')
        prices = (prices.dropna())

        for symb in prices.columns:
            stock = yf.Ticker(symb)
            curr = stock.fast_info["currency"]
            rate = base_params.forex_rate(curr)
            prices[symb] *= rate
        
        prices = np.array(prices)

        price_hist = []
        for row in prices: 
            loc_price = 0 
            for i,  item in enumerate(row):
                qty = self.assets[symbols[i]]
                loc_price += item * qty 
            price_hist.append(loc_price)
        price_hist = np.array(price_hist)
        tList_prices = np.arange(0, len(price_hist), 1)

        tList_vol, vol_hist = base_params.find_vols_by_list(tList_prices, price_hist)

        self.history_times = tList_prices #Can get to vol tList by tList_vols = tList_prices[1:len(vol_hist)+1]
        self.price_history = price_hist 
        self.volatility_history = vol_hist
        self.ref_data_period = ref_data_period

        if plot:
            print("Plotting Results...")
            Heston_Engine.clear_results()
            fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8))
            fig.suptitle(f"Historical Performance Estimation for {self.name}")
            axs[0].plot(tList_prices, price_hist)
            axs[0].set_xlabel("Time (trading days)")
            axs[0].set_ylabel("Closing Price (USD)")
            axs[0].set_title("Closing Prices vs Trading Days")

            axs[1].plot(tList_vol, vol_hist)
            axs[1].set_xlabel("Time (trading days)")
            axs[1].set_ylabel("Volatility")
            axs[1].set_title("Volatility vs Trading Days")

            print(f"One or more of the stocks in this portfolio has a maximum history of {(tList_vol[-1]/252):.1f} trading years.")
            print(f"{(tList_vol[-1]/252):.1f} ago occurs at t= ", tList_prices[0], "&", tList_vol[0], "in plot")
            print("Today is referred to in plot as t = ", tList_prices[-1], "&", tList_vol[-1])
            fig.savefig(f"{Saved_Results_Path}/{self.name}_Portfolio_History.png")

        print("Latest portfolio value:",price_hist[-1])

        return tList_prices, price_hist, vol_hist
    
    def heston_analysis(self, num_paths, horizon, start_at_theta, plot_nth_path, trials):
        theta, mu, sigma, rho, kappa = Heston_params.param_summary_from_list(self.history_times, self.price_history, self.volatility_history, self.ref_data_period)
        self.theta, self.mu, self.sigma, self.rho, self.kappa = theta, mu, sigma, rho, kappa
        print(f"Estimation Complete.\nHeston Parameters for {self.name}:\ntheta = {theta}, mu={mu}, sigma = {sigma}, rho = {rho}, kappa = {kappa}\n")
        times, prices = self.history_times, self.price_history

        num_paths = int(num_paths)
        if start_at_theta:
            V0 = theta
        else:
            times2, vols = times[1:len(self.volatility_history)+1], self.volatility_history
            V0 = vols[-1] ** 2

        S0 = prices[-1]
        return Heston_Engine.manyHeston(num_paths, S0, V0, 1/252, horizon, mu, kappa, theta, sigma, rho, True, plot_nth_path, self.name, trials)
    
    def PnL_analysis(self, num_paths, horizon:float, start_at_theta, plot_nth_path, trials):
        num_paths = int(num_paths)
        S0, final_medS, final_medV  = self.heston_analysis(num_paths, horizon, start_at_theta, plot_nth_path, trials)
        report, counts, bin_edges, patches  = profit_loss.PnL_from_list(self.name, S0, final_medS, final_medV, horizon, self.ref_data_period, num_paths, start_at_theta, True)
        print(self)

        self.RelRet_hist_count = counts 
        self.RelRet_hist_binEdges = bin_edges
        self.RelRet_hist_patches = patches 

        return report, counts, bin_edges, patches 
    
    def PnL_analysis_site(self, num_paths, horizon:float, start_at_theta, plot_nth_path, trials):
        num_paths = int(num_paths)
        S0, final_medS, final_medV  = self.heston_analysis(num_paths, horizon, start_at_theta, plot_nth_path, trials)
        report, profit_avg, vol_avg, profit_percentiles, vol_percentiles  = profit_loss.PnL_from_list_site(self.name, S0, final_medS, final_medV, horizon, self.ref_data_period, num_paths, start_at_theta, True)
        print(self)

        self.profit_avg = profit_avg
        self.vol_avg = vol_avg 
        self.profit_percentiles = profit_percentiles
        self.vol_percentiles = vol_percentiles

        return report, profit_avg, vol_avg, profit_percentiles, vol_percentiles
    
    def param_estimator(self):
            theta, mu, sigma, rho, kappa = Heston_params.param_summary_from_list(self.history_times, self.price_history, self.volatility_history, self.ref_data_period)
            self.theta, self.mu, self.sigma, self.rho, self.kappa = theta, mu, sigma, rho, kappa
            print(f"Estimation Complete.\nHeston Parameters for {self.name}:\ntheta = {theta}, mu={mu}, sigma = {sigma}, rho = {rho}, kappa = {kappa}\n")
            return (rf"Estimation Complete. Heston Parameters for {self.name}: $$\\ \theta$$ = {theta:.2f}, $$\ \mu$$={mu:.2f}, $$\ \sigma$$ = {sigma:.2f}, $$\ \rho$$ = {rho:.2f}, $$\ \kappa$$ = {kappa:.2f}")
#Make PnL analysis and heston analysis for the overall portfolio

#portfolio1 = portfolio({"TSLA":1, "SPY":20,"^HSI":3 ,"BMO":2, "NVDA":.1, "AAPL":.5}, "Test Portfolio")
#portfolio1.individual_PnL_analysis(5, "10y", 1e5, True, True)
#portfolio1.get_history("10y")
#portfolio1.heston_analysis(1e5, 5, True, 100)
#portfolio1.PnL_analysis(2e5, 3, True, 10000,  1000)

#tech_portfolio = portfolio({"TSLA":6, "META":1, "AAPL":2, "NVDA":6, "GOOG":3}, "Tech_portfolio")

#print(tech_portfolio)
#ETF_portfolio = portfolio({"IVV":1, "VOO":6, "XEQT.TO":10, "BMO":2, "CGL.TO":1}, "ETF_portfolio")
#print(ETF_portfolio)
#ETF_portfolio.get_history("10y")
#ETF_portfolio.PnL_analysis(1e4, 6, False, 300, 1000)

