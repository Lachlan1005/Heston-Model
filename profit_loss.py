import numpy as np 
import matplotlib.pyplot as plt
import base_params
import Heston_params
import Heston_Engine
from pathlib import Path
import os
import shutil
import seaborn as sns
import json

plt.style.use("dark_background")
Saved_Results_Name = "Saved_Results"

Saved_Results_Path = Path(__file__).parent / Saved_Results_Name
results_file = os.path.join(Saved_Results_Path, "Heston_Results.txt")
results_file2 = os.path.join(Saved_Results_Path, "Heston_Results_data_for_web.txt")
results_file3 = os.path.join(Saved_Results_Path, "Heston_Results_dict_for_web.json")

def PnL(symbol:str, horizon:float, ref_data_period:str, num_paths, start_at_theta:bool, plot:bool=True):
    print("Generating Heston Analysis...")
    tList, VListT, SListT = Heston_Engine.heston_for_stock(symbol, horizon, ref_data_period, num_paths, start_at_theta, True, num_paths/100)
    plt.cla()

    print("Generating PnL report...")
    S_final = SListT[-1]
    S_init = SListT[0]

    profit = S_final/S_init-1  #profit percent
    plt.hist(profit, bins = 21, alpha = 0.8)
    plt.title(f"Relative Returns on {symbol} over a horizon of {horizon} trading years")
    plt.xlabel("Relative Returns (S/S_0)")
    plt.ylabel("Number of Paths (1)")
    
    mean_prof = np.mean(profit)
    med_prof = np.median(profit)
    stdev  = np.std(profit)
    plt.axvline(mean_prof, color='r', label='mean')
    plt.axvline(med_prof, color='b', label='mean')

    plt.axvline(mean_prof - stdev, color='k', linestyle='--')
    plt.axvline(mean_prof + stdev, color='k', linestyle='--')
    plt.axvline(mean_prof - 2*stdev, color='k', linestyle='--')
    plt.axvline(mean_prof + 2*stdev, color='k', linestyle='--')

    plt.savefig(f"{Saved_Results_Path}/PnL_{symbol}.png")
    print(f"PnL Report successfully generated for {symbol}.\nMean rel. returns = {mean_prof}, Median rel. returns = {med_prof}, std. dev of rel. returns = {stdev}")

    return f"PnL Report successfully generated for {symbol}.\nMean rel. returns = {mean_prof}, Median rel. returns = {med_prof}, std. dev of rel. returns = {stdev}"

def PnL_from_list(symbol:str, S0, SList, VList, horizon:float, ref_data_period:str, num_paths, start_at_theta:bool, plot:bool=True):
    print("Generating Heston Analysis...")
    plt.cla()

    print("Generating PnL report...")

    profit = SList/S0-1  #profit percent
    vols = np.sqrt(VList)


    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = (10, 8))
    counts, bin_edges, patches  = axs.hist(profit, bins = 100, alpha = 0.8)
    max_bin_index = np.argmax(counts)
    bin_mode = (bin_edges[max_bin_index] + bin_edges[max_bin_index+1])/2

    fig.suptitle(f"Relative Returns on {symbol} over a horizon of {horizon} trading years")
    axs.set_xlabel("Relative Returns (S/S_0-1)")
    axs.set_ylabel("Number of Paths (1)")
    
    mean_prof = np.mean(profit)
    med_prof = np.median(profit)
    stdev  = np.std(profit)

    axs.axvline(mean_prof, color='r', label='mean')
    axs.axvline(med_prof, color='b', label='median')
    axs.axvline(bin_mode, color='g', label='bin_mode')
    
    axs.axvline(mean_prof - stdev, color='k', linestyle='--', label="standard deviation")
    axs.axvline(mean_prof + stdev, color='k', linestyle='--')
    axs.axvline(mean_prof - 2*stdev, color='k', linestyle='--')
    axs.axvline(mean_prof + 2*stdev, color='k', linestyle='--')

    plt.legend()
    plt.savefig(f"{Saved_Results_Path}/PnL_{symbol}.png")
    print(f"PnL Report successfully generated for {symbol}.\nMean rel. returns = {mean_prof}, Median rel. returns = {med_prof}, std. dev of rel. returns = {stdev}")

    with open(results_file, "a") as f:
        f.writelines(f"\n\n ===== PnL Analysis ===\nHistogram details:\ncounts = {counts}\nbin edges = {bin_edges}, patches = {patches}\n\n")
        f.writelines(f"Relative Returns: Mean= {mean_prof}, Median = {med_prof}, Bin Mode  = {bin_mode}, Standard Deviation = {stdev}\n")
        f.writelines(f"Volatility: Mean = {np.mean(np.sqrt(VList))}, Median = {np.median(np.sqrt(VList))},standard deviation = {np.std(np.sqrt(VList))}\n\n")
        f.writelines("---Percentiles--")
        f.writelines(f"Relative Returns: 1st = {np.percentile(profit, 1)}, 5th = {np.percentile(profit, 5)}, 10th = {np.percentile(profit, 10)}, 25th = {np.percentile(profit, 25)}, 75th = {np.percentile(profit, 75)}, 90th = {np.percentile(profit, 90)}, 99th = {np.percentile(profit, 99)}\n")
        f.writelines(f"Volatility: 1st = {np.percentile(vols, 1)}, 5th = {np.percentile(vols, 5)}, 10th = {np.percentile(vols, 10)}, 25th = {np.percentile(vols, 25)}, 75th = {np.percentile(vols, 75)}, 90th = {np.percentile(vols, 90)}, 99th = {np.percentile(vols, 99)}\n")

    with open(results_file2, "a") as f:
        f.writelines(f"\n\n### Averages ")
        f.writelines(f"\nRelative Returns: Mean= {mean_prof}, Median = {med_prof}, Bin Mode  = {bin_mode}, Standard Deviation = {stdev}\n\n")
        f.writelines(f"Volatility: Mean = {np.mean(np.sqrt(VList))}, Median = {np.median(np.sqrt(VList))},standard deviation = {np.std(np.sqrt(VList))}\n\n")
        f.writelines("### Percentiles")
        f.writelines(f"\nRelative Returns: 1st = {np.percentile(profit, 1)}, 5th = {np.percentile(profit, 5)}, 10th = {np.percentile(profit, 10)}, 25th = {np.percentile(profit, 25)}, 75th = {np.percentile(profit, 75)}, 90th = {np.percentile(profit, 90)}, 99th = {np.percentile(profit, 99)}")
        f.writelines(f"\n\nVolatility: 1st = {np.percentile(vols, 1)}, 5th = {np.percentile(vols, 5)}, 10th = {np.percentile(vols, 10)}, 25th = {np.percentile(vols, 25)}, 75th = {np.percentile(vols, 75)}, 90th = {np.percentile(vols, 90)}, 99th = {np.percentile(vols, 99)}")
        f.writelines(f"\n\n### Histogram Results in Text")
        f.writelines(f"\n\ncounts = {counts}\nbin edges = {bin_edges}, patches = {patches}\n\n")
    
    #The following are in percent
    profit_avg = {"Mean":float(mean_prof), "Median":float(med_prof), "Bin Mode":float(bin_mode), "Standard Deviation":float(bin_mode)}
    vol_avg = {"Mean": float(np.mean(np.sqrt(VList))), "Median": float(np.median(np.sqrt(VList))), "Standard Deviation": float(np.std(np.sqrt(VList)))}
    profit_percentiles = {}
    vol_percentiles = {}
    for i in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        profit_percentiles[i] = float(np.percentile(profit, i))
        vol_percentiles[i] = float(np.percentile(vols, i))

    results = {"profit_avg":profit_avg, "vol_avg":vol_avg, "profit_percentiles":profit_percentiles, "vol_percentiles":vol_percentiles}
    with open(results_file3, "w") as f:
        f.write(json.dumps((results)))


    report = f"PnL Report successfully generated for {symbol}.\n\nPercentage Return: Mean = {(mean_prof*100):.2g}%  \nMedian = {(med_prof*100):.2g}%  \nStandard Deviation = {(stdev*100):.2g}%\n\nVolatility: Mean = {np.mean(np.sqrt(VList))}, Median = {np.median(np.sqrt(VList))},Standard Deviation = {np.std(np.sqrt(VList))}"
    return report, counts, bin_edges, patches 



def PnL_from_list_site(symbol:str, S0, SList, VList, horizon:float, ref_data_period:str, num_paths, start_at_theta:bool, plot:bool=True):
    print("Generating Heston Analysis...")
    plt.cla()

    print("Generating PnL report...")

    profit = SList/S0-1  #profit percent
    vols = np.sqrt(VList)


    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = (10, 8))
    counts, bin_edges, patches  = axs.hist(profit, bins = 100, alpha = 0.8)
    max_bin_index = np.argmax(counts)
    bin_mode = (bin_edges[max_bin_index] + bin_edges[max_bin_index+1])/2

    fig.suptitle(f"Relative Returns on {symbol} over a horizon of {horizon} trading years")
    axs.set_xlabel("Relative Returns (S/S_0-1)")
    axs.set_ylabel("Number of Paths (1)")
    
    mean_prof = np.mean(profit)
    med_prof = np.median(profit)
    stdev  = np.std(profit)

    axs.axvline(mean_prof, color='r', label='mean')
    axs.axvline(med_prof, color='b', label='median')
    axs.axvline(bin_mode, color='g', label='bin_mode')
    
    axs.axvline(mean_prof - stdev, color='k', linestyle='--', label="standard deviation")
    axs.axvline(mean_prof + stdev, color='k', linestyle='--')
    axs.axvline(mean_prof - 2*stdev, color='k', linestyle='--')
    axs.axvline(mean_prof + 2*stdev, color='k', linestyle='--')

    plt.legend()
    plt.savefig(f"{Saved_Results_Path}/PnL_{symbol}.png")
    print(f"PnL Report successfully generated for {symbol}.\nMean rel. returns = {mean_prof}, Median rel. returns = {med_prof}, std. dev of rel. returns = {stdev}")

    with open(results_file, "a") as f:
        f.writelines(f"\n\n ===== PnL Analysis ===\nHistogram details:\ncounts = {counts}\nbin edges = {bin_edges}, patches = {patches}\n\n")
        f.writelines(f"Relative Returns: Mean= {mean_prof}, Median = {med_prof}, Bin Mode  = {bin_mode}, Standard Deviation = {stdev}\n")
        f.writelines(f"Volatility: Mean = {np.mean(np.sqrt(VList))}, Median = {np.median(np.sqrt(VList))},standard deviation = {np.std(np.sqrt(VList))}\n\n")
        f.writelines("---Percentiles--")
        f.writelines(f"Relative Returns: 1st = {np.percentile(profit, 1)}, 5th = {np.percentile(profit, 5)}, 10th = {np.percentile(profit, 10)}, 25th = {np.percentile(profit, 25)}, 75th = {np.percentile(profit, 75)}, 90th = {np.percentile(profit, 90)}, 99th = {np.percentile(profit, 99)}\n")
        f.writelines(f"Volatility: 1st = {np.percentile(vols, 1)}, 5th = {np.percentile(vols, 5)}, 10th = {np.percentile(vols, 10)}, 25th = {np.percentile(vols, 25)}, 75th = {np.percentile(vols, 75)}, 90th = {np.percentile(vols, 90)}, 99th = {np.percentile(vols, 99)}\n")

    with open(results_file2, "a") as f:
        f.writelines(f"\n\n### Averages ")
        f.writelines(f"\nRelative Returns: Mean= {mean_prof}, Median = {med_prof}, Bin Mode  = {bin_mode}, Standard Deviation = {stdev}\n\n")
        f.writelines(f"Volatility: Mean = {np.mean(np.sqrt(VList))}, Median = {np.median(np.sqrt(VList))},standard deviation = {np.std(np.sqrt(VList))}\n\n")
        f.writelines("### Percentiles")
        f.writelines(f"\nRelative Returns: 1st = {np.percentile(profit, 1)}, 5th = {np.percentile(profit, 5)}, 10th = {np.percentile(profit, 10)}, 25th = {np.percentile(profit, 25)}, 75th = {np.percentile(profit, 75)}, 90th = {np.percentile(profit, 90)}, 99th = {np.percentile(profit, 99)}")
        f.writelines(f"\n\nVolatility: 1st = {np.percentile(vols, 1)}, 5th = {np.percentile(vols, 5)}, 10th = {np.percentile(vols, 10)}, 25th = {np.percentile(vols, 25)}, 75th = {np.percentile(vols, 75)}, 90th = {np.percentile(vols, 90)}, 99th = {np.percentile(vols, 99)}")
        f.writelines(f"\n\n### Histogram Results in Text")
        f.writelines(f"\n\ncounts = {counts}\nbin edges = {bin_edges}, patches = {patches}\n\n")
    
    #The following are in percent
    profit_avg = {"Mean":float(mean_prof), "Median":float(med_prof), "Bin Mode":float(bin_mode), "Standard Deviation":float(bin_mode)}
    vol_avg = {"Mean": float(np.mean(np.sqrt(VList))), "Median": float(np.median(np.sqrt(VList))), "Standard Deviation": float(np.std(np.sqrt(VList)))}
    profit_percentiles = {}
    vol_percentiles = {}
    for i in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        profit_percentiles[i] = float(np.percentile(profit, i)) * 100 #Convert into percent
        vol_percentiles[i] = float(np.percentile(vols, i))

    results = {"profit_avg":profit_avg, "vol_avg":vol_avg, "profit_percentiles":profit_percentiles, "vol_percentiles":vol_percentiles}
    with open(results_file3, "w") as f:
        f.write(json.dumps((results)))


    report = f"**Percentage Return | How much of your initial investment will you gain (negative gain = loss!)**\n\nMean = {(mean_prof*100):.2g}%  \nMedian = {(med_prof*100):.2g}%  \nStandard Deviation (Empirical Volatility) = {(stdev*100):.2g}%\n\n\n\n**Volatility | The tendency of the value of your portfolio to change**  \nMean = {np.mean(np.sqrt(VList)):.2g}  \nMedian = {np.median(np.sqrt(VList)):.2g}  \nStandard Deviation = {np.std(np.sqrt(VList)):.2g}"
    return report, profit_avg, vol_avg, profit_percentiles, vol_percentiles




    
