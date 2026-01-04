import numpy as np 
import matplotlib.pyplot as plt
import base_params
import Heston_params
from pathlib import Path
import os
import shutil
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


def heston(S0, V0, dt, tmax, mu, kappa, theta, sigma, rho=0, plot=True):
    """
    Generate a single Heston path. If plot, graph price on left and volatility on right WRT time t
    Uses an unstable Euler solver. Only use for quick estimations in debugging. 

    :param S0: Initial price
    :param V0: Initial volatiltiy
    :param dt: timestep
    :param tmax: simulation time
    :param mu: drift
    :param kappa: mean reversion rate
    :param theta: historical volatility
    :param sigma: volatility of volatility
    :param rho: correlation between random variables
    """

    t=0 
    S = S0 
    V = V0 

    SList = [S]
    VList = [V]
    tList = [t]
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8))

    while t<tmax:
        print(30*"\n"+f"t={t} out of {tmax}")
        eps1 = np.random.randn()
        dS = S*(mu*dt + np.sqrt(V*dt)*eps1)
        eps2 = rho*eps1 + np.sqrt(1-rho**2)*np.random.randn()
        dV = kappa*(theta - V)*dt + sigma*np.sqrt(V*dt)*eps2

        S += dS 
        V+= dV 
        t+=dt

        SList.append(S)
        VList.append(V)
        tList.append(t)
    
    if plot:
        axs[0].plot(tList, SList, "|", color="black")
        axs[1].plot(tList, VList, ":", color="blue")
    
    plt.show()
    
    return tList, VList, SList


def manyHeston(num_paths, S0, V0, dt, tmax, mu, kappa, theta, sigma, rho=0, plot=True, plot_nth_path = 10, name="SampleCoin", trials = 1):
    """
    Generate num_path Heston paths each trial and conduct trials number of trials. If plot, graph price on left and volatility on right WRT time t
    Uses the Andersen QE scheme to numerically solve the Heston equations. 

    Decrease num_paths and increase trials if computer is struggling to handle large arrays (usually happens when num_path > 1e6)
    
    :param S0: Initial price
    :param V0: Initial volatiltiy
    :param dt: timestep
    :param tmax: simulation time
    :param mu: drift
    :param kappa: mean reversion rate
    :param theta: historical volatility
    :param sigma: volatility of volatility
    :param rho: correlation between random variables
    """
    trial = 0
    total_VListT, total_SListT = None, None
    final_medS = []
    final_medV = []
    with open(results_file, "a") as f:
            f.writelines(f"{name} || Heston Model with {num_paths} Paths ({tmax} trading years) || Initial Price={S0:.2f}, Initial Volatility={V0:.2f} || mu={mu:.2f}, kappa = {kappa:.2f}, theta={theta:.2f}, sigma={sigma:.2f}, rho={rho:.2f}")
    while trial<trials:
        t=0 
        S = S0*np.ones(num_paths)
        V = V0*np.ones(num_paths)

        SList = []
        VList = []
        tList = []

        iter = 0
        if plot:
            fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8))

        hist = None 
        while t<tmax:
            print(30*"\n"+f"Heston run for {name} (Trial {trial}) || Solver running...\nt={t} out of {tmax}")
            eps1 = np.random.randn(num_paths)
            eps2 = rho*eps1 + np.sqrt(1-rho**2)*np.random.randn(num_paths)
            S, V = QE_step_many(num_paths, S, V, dt, mu, kappa, theta, sigma, rho, 3/2, 1/2)
            t =dt + t

            
            if iter % save_nth_timestep == 0:
                SList.append(S)
                VList.append(V)
                tList.append(t)
            iter+=1
        
        print("Processing Data...")
        SListT = np.array(SList).T
        VListT = np.array(VList).T
        tList = np.array(tList)
        stdDev_S = np.std(SListT)

        if plot:
            print("Plotting Data...")
            for i, path in enumerate(SListT): 
                if i % plot_nth_path ==0:
                    axs[0].plot(tList, path, color="white", alpha = .2, linewidth=1) #x axis is trading years
                    axs[1].plot(tList, np.sqrt(VListT[i]), color="blue", alpha = .2, linewidth=1)
            axs[0].plot(tList, np.ones_like(tList)*S0, color="red", label = "Initial", linewidth=1)
            axs[0].set_ylim(0, S0+stdDev_S*3)
            axs[1].plot(tList, np.ones_like(tList)*np.sqrt(V0), color="red", linewidth=1)

            fig.suptitle(f"{name} || Heston Model with {num_paths} Paths || S0={S0:.2f}, V0={V0:.2f}, mu={mu:.2f}, kappa = {kappa:.2f}, theta={theta:.2f}, sigma={sigma:.2f}, rho={rho:.2f}")

        print("Plotting Complete.")
        print(f"{name} || Heston Model with {num_paths} Paths ({tmax} trading years) || Initial Price={S0:.2f}, Initial Variance={V0:.2f} || mu={mu:.2f}, kappa = {kappa:.2f}, theta={theta:.2f}, sigma={sigma:.2f}, rho={rho:.2f}")
        
        medianS = np.median(SListT[-1])
        medianV = np.median((VListT[-1])) 
        medianVol = np.median(np.sqrt(VListT[-1])) #median volatiltiy (NOT median V which is variance)
        if plot:
            axs[0].plot(tList, np.ones_like(tList)*medianS, color="green", label = "Final (Mean)", linewidth=1)
            axs[0].set_xlabel("Time (trading years)")
            axs[0].set_ylabel("Asset Price (USD)")
            axs[1].plot(tList, np.ones_like(tList)*medianVol, color="green", linewidth=1)
            axs[1].set_xlabel("Time (trading years)")
            axs[1].set_ylabel("Variance (1)")

        print(f"Median Final Price = {medianS:.2f} || Median Final Volatility = {medianVol:.2f}")

        if plot:
            if trial % 10 == 0:
                fig.savefig(f"{Saved_Results_Path}/Heston_{name}_trial-{trial}.png")
        if total_VListT is None or  total_SListT is None: 
            total_VListT, total_SListT= VListT, SListT
        else: 
            total_VListT = total_VListT+VListT
            total_SListT = total_SListT + SListT

        with open(results_file, "a") as f:
            f.writelines(f"\n\n-----Trial {trial} results -----\nmedian final price = {medianS}, median final volatility = {medianVol} || standard deviation of final price = {np.std(SListT[-1])}, standard deviation of final volatility = {np.std(VListT[-1])}")
            f.writelines(f"median final rel. return = {np.median(SListT[-1]/S0-1)} ||mean final rel. return = {np.mean(SListT[-1]/S0-1)}  || standard deviation of final rel. return = {np.std(SListT[-1]/S0-1)}")

        final_medV.append(medianV)
        final_medS.append(medianS)
        trial +=1 

    final_medV = np.array(final_medV)
    final_medS = np.array(final_medS)

    counts, bin_edges, patches =plt.hist(final_medS, 100)
    max_bin_index = np.argmax(counts)
    most_likely_final = (bin_edges[max_bin_index] + bin_edges[max_bin_index+1])/2


    plt.savefig(f"{Saved_Results_Path}/Heston_{name}_histogram.png")
    print(f"Bin Mode: {most_likely_final/S0 -1}")
    return S0, final_medS, final_medV
            
def QE_step(S, V, dt, mu, kappa, theta, sigma, rho, psi_c = 3/2, gamma1 = 1/2):
    """
    Return the QE step for the Heston model wit timestep dt
    """
    gamma2 = 1-gamma1
    m = theta + (V-theta) * np.exp(-kappa * dt)
    s_squared = (V*sigma**2 /kappa ) * np.exp(-kappa*dt) * (1-np.exp(-kappa * dt)) + theta * sigma**2/(2*kappa) * (1-np.exp(-kappa*dt))**2
    psi = s_squared/m**2

    eps0 = np.random.rand()
    eps1 = np.random.randn()
    
    if psi <= psi_c:
        rho_c = rho
        b_squared = 2/psi -1 + np.sqrt(np.maximum(2/psi * (2/psi -1), 0))
        b = np.sqrt(np.maximum(b_squared, 0))
        a = m/(1+b**2)
        new_V = a*(b+eps1)**2 
    else:
        rho_c = 0 
        beta = 2/(m * (psi+1))
        p = (psi - 1)/(psi + 1)
        if eps0 > p: 
            new_V = np.log((1-p)/(1-eps0))/beta 
        else: 
            new_V = 0 
    eps2 = rho_c*eps1 + np.sqrt(1-rho_c**2)*np.random.randn()

    K0 = -(rho*kappa*theta / sigma ) * dt 
    K1 = gamma1 * dt * (kappa*rho/sigma - 1/2) - rho/sigma
    K2 = gamma2 * dt * (kappa*rho/sigma - 1/2) + rho/sigma
    K3 = gamma1 * dt *(1-rho**2)
    K4 = gamma2 * dt * (1-rho**2)

    new_V = np.maximum(new_V, 0)
    new_S = S*np.exp(K0+mu * dt +K1*V + K2 * new_V + np.sqrt(K3 * V + K4 * new_V) * eps2)
    return new_S, new_V

def QE_step_many(num_path:int, S:np.ndarray, V:np.ndarray, dt, mu, kappa, theta, sigma, rho, psi_c = 3/2, gamma1 = 1/2):
    """
    Return the QE step for the Heston model wit timestep dt woth num_path paths
    """
    gamma2 = 1-gamma1
    m = theta + (V-theta) * np.exp(-kappa * dt)
    s_squared = (V*sigma**2 /kappa ) * np.exp(-kappa*dt) * (1-np.exp(-kappa * dt)) + theta * sigma**2/(2*kappa) * (1-np.exp(-kappa*dt))**2
    psi = s_squared/m**2

    eps0 = np.random.rand(num_path)
    eps1 = np.random.randn(num_path)
    
    mask = psi <= psi_c
    psi_quad = psi[mask]
    psi_exp = psi[~mask]

    b_squared = 2/psi_quad -1 + np.sqrt(np.maximum(2/psi_quad * (2/psi_quad -1), 0))
    b = np.sqrt(np.maximum(b_squared, 0))
    a = m[mask]/(1+b**2)
    V_quad = a * (b+eps1[mask])**2 
    
    beta = 2/(m[~mask] * (psi_exp+1))
    p = (psi_exp - 1)/(psi_exp + 1)

    V_exp = np.maximum(np.log((1-p)/(1-eps0[~mask]))/beta, 0)  #Andersen: V_exp ~ ln((1-p)/(1-u)) has root at u = p => u>p enforces positivity => np.maximum(log...) is equivalent

    new_V = np.zeros(num_path)
    new_V[mask] = V_quad 
    new_V[~mask] = V_exp

    quad_mask_length = len(psi_quad)
    exp_mask_length = len(psi_exp)
    eps2_quad = rho*eps1[mask] + np.sqrt(1-rho**2)*np.random.randn(quad_mask_length)
    eps2_exp = np.random.randn(exp_mask_length)

    eps2 = np.zeros(num_path)
    eps2[mask] = eps2_quad
    eps2[~mask] = eps2_exp

    K0 = -(rho*kappa*theta / sigma ) * dt 
    K1 = gamma1 * dt * (kappa*rho/sigma - 1/2) - rho/sigma
    K2 = gamma2 * dt * (kappa*rho/sigma - 1/2) + rho/sigma
    K3 = gamma1 * dt *(1-rho**2)
    K4 = gamma2 * dt * (1-rho**2)

    new_V = np.maximum(new_V, 0)
    new_S = S*np.exp(K0+mu * dt +K1*V + K2 * new_V + np.sqrt(K3 * V + K4 * new_V) * eps2)
    return new_S, new_V


def heston_for_stock(symbol:str, horizon:float, ref_data_period:str, num_paths, start_at_theta:bool, plot:bool=True, plot_nth_path:int=10):
    """
    Use the heston model to simulate num_paths of stock paths over a horizon of horizon 
    using the Heston model, 
    with parameters estimated using the price history of stock specified by symbol over a 
    history of ref_data_period

    if start_at_theta is set to True, then the solver will start with volatility at the historical mean volatility. 
    """
    theta, mu, sigma, rho, kappa = Heston_params.param_summary(symbol, ref_data_period)

    times, prices = base_params.stock_closing_prices(symbol, ref_data_period)
    if start_at_theta:
        V0 = theta
    else:
        times2, vols = base_params.find_vols(symbol, ref_data_period)
        V0 = vols[-1] ** 2

    S0 = prices[-1]

    return manyHeston(num_paths, S0, V0, 1/252, horizon, mu, kappa, theta, sigma, rho, plot, plot_nth_path, symbol)

TY = 252 #so that x/TY == x trading days

def clear_results():
    if os.path.exists(Saved_Results_Path) and os.path.isdir(Saved_Results_Path):
        shutil.rmtree(Saved_Results_Path)
    os.makedirs(Saved_Results_Path, exist_ok=True)
    results_file = os.path.join(Saved_Results_Path, "Heston_Results.txt")
    results_file2 = os.path.join(Saved_Results_Path, "Heston_Results_data_for_web.txt")
    results_file3 = os.path.join(Saved_Results_Path, "Heston_Results_dict_for_web.txt")

    with open(results_file, "w") as f:
        pass  
    with open(results_file2, "w") as f:
        pass
    with open(results_file3, "w") as f:
        pass

#clear_results()
#base_params.plot_sumamry("^HSI", "15y")
#heston_for_stock("^HSI", 252/TY, "10y", int(1e5), False, True, 500)    


#dt should be in trading years (1 TY == 252 trading days) -> dt = 1/252
#Simulate 10 years max (recommeded 3-6 years)


#manyHeston(num_paths=int(1e5), S0=650.18,V0=1/5,dt=1/TY, tmax=1500/TY, 
 #      mu=.16, kappa = 1, theta = 1/5, 
 #      sigma = 1/2, rho = -1/3, plot = True, plot_nth_path=int(8e2), name="TSLA")    
    

#manyHeston(num_paths=int(1e5), S0=4500,V0=.0225,dt=1/TY, tmax=1500/TY, 
 #      mu=.08, kappa = 1, theta = .0225, 
 #      sigma = .15, rho = -.3, plot = True, plot_nth_path=int(8e2), name="S&P 500")    
    
#clear_results()