# Heston Model 
This project projects the value of user-defined portfolios over a user-requested horizon. Users can acess the project through the webapp online (or locally by running `website.py`) or the local text interface (CLI) in `interface.py`. For local uses of the projects, all results can be acessed through `Saved_Results`. This is my first time using Streamlit, so the website may be unstable. Should any unfixable error arise, users are encouraged to use the much more stable local CLI version over the webapp. 

## Computational Methods
This is done via Monte Carlo inside `Heston_Engine.py` with a user-defined amount of paths.
The determinattion of each path
involves solving the Heston model stochastic differential equations using the quadratic exponential (QE) scheme proposed by Andersen in 2008. The parameters of the 
Heston model are estimated using the statistical methods in `Heston_params.py` and `base_params.py` based on historical market data provided by Yahoo Finance. The results of the simulation 
are aggregated through `profit_loss.py` to produce useful data. 

## Portfolio Creation
Portfolios are defined in a `class` inside `portcolio_class.py`

## Dependencies
If the project is run locally, it is essential that the following dependencies are already installed:
```
streamlit==1.30.0
numpy==1.27.0
pandas==2.1.0
matplotlib==3.8.1
seaborn==0.12.2
yfinance==0.2.30
os
json
shutil
pathlib
```