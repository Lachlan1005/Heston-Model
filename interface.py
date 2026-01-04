import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
import pandas as pd
import portfolio_class, base_params, Heston_params, Heston_Engine, profit_loss, shutil, os
import seaborn as sns

sns.set_style("dark")

def interface_run():
    print("========== Heston Model Analysis ==========")
    name = input("Enter name of portfolio: ") 
    
    enter_assets = True 
    assets_dict = {}
    print(f"Enter Assets for {name}. Make sure the symbols exist and can be found via Yahoo Finance.")
    while enter_assets:
        symbol = input(f"Asset Symbol: ")
        amount = float(input(f"Number of {symbol} in {name}:"))
        add = input(f"Adding {amount} of {symbol} to {name}. Continue? (y/n): ")
        if add == "y":
            assets_dict[symbol] = amount
        else: 
            print("Process Aborted. ")
        enter = input(f"Continue adding assets? (y/n): ")
        if enter == "n":
            enter_assets = False
    cur_portfolio = portfolio_class.portfolio(assets_dict, name)
    print("\n\n", cur_portfolio)
    input("Continue to analysis? (press any key): ")

    while True:
        print("\n\n---------- Analysis Options ----------")
        print("key --> Function")
        print("1   --> Full Heston analysis with PnL and VaR")
        print("2   --> Check historical performance of portfolio")
        print("3   --> Compute Heston parameters")
        print("q   --> Quit")
        option = int(input("Your Option: "))

        if option == 1:
            print("\n\nEnter the following relevant parameters.")
            num_paths = int(input("Number of simulated paths (Recommended range: 1000 - 100000): "))
            horizon = int(input("Simulation horizon (in trading years , recommended to not exceed 10): "))
            plot_nth_path = int(input("Plot one in how many paths? (Recommended to be 10 to 100 times less than the number of simulated paths):"))        
            trials = int(input("Number of trials to conduct (Recommended range: 100 - 1000):"))
            ref_data_period = int(input("Number of years of historical data to use to caliberate model (Recommended 10 years): "))
            ref_data_period  =str(ref_data_period)+"y"
            input("Continue to solver? (press any key): ")
            cur_portfolio.get_history(ref_data_period)
            cur_portfolio.PnL_analysis(num_paths, horizon, True, plot_nth_path, trials)
            print("See Saved_Results for visual resports.")
        
        if option == 2:
            ref_data_period = int(input("Number of years of historical data to retrieve: "))
            ref_data_period  =str(ref_data_period)+"y"
            input("Continue to calculation? (press any key): ")
            cur_portfolio.get_history(ref_data_period, plot = True)
            print("Check Saved_Results for historical performance")

        if option == 3:
            print("\n\nEnter the following relevant parameters.")
            ref_data_period = int(input("Number of years of historical data to use to caliberate model (Recommended 10 years): "))
            ref_data_period  =str(ref_data_period)+"y"
            input("Continue to calculation? (press any key): ")
            cur_portfolio.get_history(ref_data_period)
            cur_portfolio.param_estimator()
        
        cont = input("Press any key to return to menu or press q to exit")
        if cont == "q":
            print("Exitting program.")
            break
        
        



interface_run()