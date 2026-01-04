import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
import pandas as pd
import base_params, Heston_params, Heston_Engine, portfolio_class ,profit_loss, shutil, os
import seaborn as sns
import json

plt.style.use("dark_background")

Saved_Results_Name = "Saved_Results"

save_nth_timestep = 3

# Same local location across all machines
Saved_Results_Path = Path(__file__).parent / Saved_Results_Name
results_file = os.path.join(Saved_Results_Path, "Heston_Results.txt")
results_file2 = os.path.join(Saved_Results_Path, "Heston_Results_data_for_web.txt")
results_file3 = os.path.join(Saved_Results_Path, "Heston_Results_dict_for_web.txt")

if "data" not in st.session_state:
    st.session_state.data = {"name":""}
    st.header(f"Heston Analysis")
else:
    cur_name = st.session_state.data["name"]
    st.header(f"Heston Analysis for {cur_name}")

if "assets" not in st.session_state:
    st.session_state.assets = {}

if "myPortfolio" not in st.session_state:
    st.session_state.myPortfolio = None

if "assetDistrib" not in st.session_state:
    st.session_state.assetDistrib = None

if "part1" not in st.session_state:
    st.session_state.part1 = False

if "part2" not in st.session_state:
    st.session_state.part2 = False

if "readiness" not in st.session_state:
    st.session_state.readiness = False

if "ref_period_button" not in st.session_state:
    st.session_state.ref_period_button = False

if "history_img_path" not in st.session_state:
    st.session_state.history_img_path = None

if "results_dict" not in st.session_state:
    st.session_state.results_dict = {}

print(st.session_state.part2)


st.subheader("Step 1: Portfolio Setup")
st.markdown("Tip: Always double click on buttons, or else they may not work!")
name = st.text_input("Enter name of portfolio:")


if st.button("Confirm Name"):
    st.session_state.data["name"] = name

st.subheader("Enter your assets")
asset = st.text_input("Symbol of asset")
count = st.text_input(f"Number of {asset}")
st.markdown("Make sure the symbol is valid on Yahoo Finance and that you entered a number in the field that asks for a number before clicking the add button.")
if st.button("Add to my portfolio"):
    st.session_state.assets[asset] = float(count)

potName = st.session_state.data["name"]
st.markdown(f"#### Assets in {potName}:")
for ass in st.session_state.assets:
    st.markdown(f"{ass} x {str(st.session_state.assets[ass])}")

print(st.session_state.assets, " is a ", type(st.session_state.assets))

complete_button = st.button("Complete Portfolio")
st.markdown("Note that once the complete button is pressed, no further changes to the portfolio will be registered.")
if complete_button:
    st.session_state.part1 = True
    placeholder1 = st.empty()
    placeholder1.markdown("Initializing portfolio...")
    st.session_state.myPortfolio = portfolio_class.portfolio(st.session_state.assets, st.session_state.data["name"])
    placeholder1.empty() 

if st.session_state.part1 ==True:
    st.markdown(st.session_state.myPortfolio)
    st.session_state.assetDistrib = st.session_state.myPortfolio.asset_vals
    fig1, ax1 = plt.subplots()
    ax1.set_title(f"Distribution of Holdings for {st.session_state.data["name"]}")
    ax1.pie(st.session_state.assetDistrib.values(), labels = st.session_state.assetDistrib.keys(), startangle = 90)
    st.pyplot(fig1)


    st.subheader("Step 2: Parameter Estimation")
    st.markdown("We now search for parameters to run our simulation. How many years of data should we use? (A common choice is 10)")
    ref_data_period = st.slider(min_value=1, max_value=20, step =1 ,label = "Years of History")
    st.markdown("(If the following button raises an error, click it again until the error goes away.)")
    if st.button("Confirm Reference Data Period"):
    #    st.markdown("**Got an Error?** Check if your symbols are all valid symbols searchable on Yahoo Finance. If everything loks fine and you encounter an error, press the 'Confirm Reference Data Period' button again until the error goes away. If all else fails, reload the site.")
        ref_data_period = str(ref_data_period)+"y"
        st.session_state.data["ref_data_period"] = ref_data_period
        st.markdown(f"Using {st.session_state.data["ref_data_period"]} of closing price history. We now search for the history of all the assets in the portfolio to deduce a portfolio history. Note that the reference data period will be cut short if one or more assets have less history than specified. ")
        placeholder2 = st.empty()
        placeholder2.markdown("Retrieving individual historical asset data and calculating portfolio history...")
        tList_prices, price_hist, vol_hist = st.session_state.myPortfolio.get_history(st.session_state.data["ref_data_period"], plot = True)
        st.session_state.data["ref_tList_prices"] = tList_prices
        placeholder2.empty() 
        history_img_path = f"{Saved_Results_Path}/{st.session_state.data["name"]}_Portfolio_History.png"
        st.image(history_img_path)
        st.markdown(f"One or more stock/s in {st.session_state.data["name"]} has a maximum history of {(st.session_state.data["ref_tList_prices"][-1]/252):.1f} trading years. Therefore this will be our historical reference period from now on.")
        st.markdown(r"We now estimate the Heston parameters. They are $\mu$, the historical drift; $\sigma$, the volatility of volatility; $\theta$, the historical variance; $\rho$, the correlation parameter; and $\kappa$, the mean reversion speed of variance ")
        param_str = st.session_state.myPortfolio.param_estimator()
        st.markdown(param_str)
        st.markdown("Does these numbers look good? If so, we can configure the simulation now!")
        st.markdown("(This will take you to the next step and you will not be able to access portfolio history again.)")
        st.session_state.readiness = True
        st.session_state.ref_period_button = True
        st.session_state.history_img_path = history_img_path
        
   
    if st.session_state.readiness:
        if st.button("Configure the simulation"):
            st.session_state.part2 = True     


    print("st.session_state.part2 is ", {st.session_state.part2})
    if st.session_state.part2 == True:
        st.subheader("Step 3: Run Simulation")
        horizon = st.slider(label = "Simulation horizon (in trading years; accuracy decreases after 6-8 years)", min_value=1.0, max_value=10.0, step = 1/10)
        num_paths = st.slider(label = "Number of simulated paths (per trial; Recommended: 50000-100000)", min_value=1, max_value=100000, step =1)
        trials = st.slider(label = "Number of trials (Recommended: 600 - 1000)", min_value=1, max_value=1000, step = 1)
        plot_nth_path = st.slider(label = "Plot one path out of how many? (Recommedned: path number/100)", min_value= 0 ,max_value=int(num_paths+1), step = 1)
        if st.button("Confirm settings and start solver"):
            Heston_Engine.clear_results()
            with st.spinner("Solver Running... "):
                placeholder3 = st.empty()
                placeholder3.markdown("(This may take a while, often up to 30 minutes, maybe even an hour or two depending on your settings. Go get yourself a coffee!)")
                st.session_state.myPortfolio.get_history(st.session_state.data["ref_data_period"])
                report, profit_avg, vol_avg, profit_percentiles, vol_percentiles  = st.session_state.myPortfolio.PnL_analysis_site(num_paths, horizon, True, plot_nth_path, trials)
                placeholder3.empty()
            st.header("Results")
            histogram_img_path = (f"{Saved_Results_Path}/PnL_{st.session_state.data["name"]}.png")
            st.image(histogram_img_path, caption = "Right: Histogram of expected returns (Not in percent! Multiply the x-axis by 100x to get percent.)")

            expt_img_path = (f"{Saved_Results_Path}/Heston_{name}_trial-0.png")
            st.image(expt_img_path, caption = "Left: The simulated paths of one of the trials. Right: The simulated variance (square of volatility) of one of the trials")
            st.markdown(report)

            st.session_state.results_dict["profit_avg"] = profit_avg 
            st.session_state.results_dict["vol_avg"] = vol_avg
            st.session_state.results_dict["profit_percentiles"] = profit_percentiles
            st.session_state.results_dict["vol_percentiles"] = vol_percentiles
            colors = [
    "#67000d",  # darkest red (biggest loss)
    "#a50f15",
    "#cb181d",
    "#ef3b2c",  # lightest red
    "#f0f0f0",  # neutral / break-even (almost white)
    "#aff49e",  # lightest green (small profit)
    "#42d353",
    "#1ca23f",
    "#00692A"   # darkest green (biggest profit)
]
            percentiles_labels = profit_percentiles.keys()
            profit_distrib = profit_percentiles.values()
            vol_distrib = vol_percentiles.values()

            fig2, ax2 = plt.subplots()

            ax2.bar( x=[f"{p}th" for p in percentiles_labels], height=profit_distrib, color=colors)

            ax2.set_yticks(np.arange( np.floor(min(profit_distrib) ), np.ceil(max(profit_distrib)) + 20, 20))

            ax2.set_xlabel("Percentile")
            ax2.set_ylabel("Percentage Return")
            fig2.suptitle( f"Percentiles of Percentage Returns for {st.session_state.data['name']}")
            st.pyplot(fig2)
            st.markdown("To answer the question of 'How much will I make if I'm in the luckiest X% of people' or 'How much will I lose if I'm in the unluckiest X% of people', " \
            "simply look up the number X on the percentile axis and read off the bar chart to find the percentage return for that percentile")



            



        
