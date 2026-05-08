import yfinance as yf
import numpy as np
import pandas as pd

def fetch_market_parameters(ticker: str):
    print(f"Fetching real market data and options chain for {ticker}...")
    asset = yf.Ticker(ticker)
    
    hist = asset.history(period="5y")
    if hist.empty:
        raise ValueError(f"No historical data found for {ticker}.")
        
    hist['Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
    hist.dropna(inplace=True)

    std_dev = hist['Returns'].std()
    jump_threshold = 3 * std_dev

    jumps = hist[hist['Returns'].abs() > jump_threshold]['Returns']
    continuous_returns = hist[hist['Returns'].abs() <= jump_threshold]['Returns']

    realized_var = continuous_returns.var() * 252  
    jump_intensity = len(jumps) / 5.0              
    jump_mean = jumps.mean()                       
    jump_std = jumps.std()                         

    if np.isnan(jump_mean): 
        jump_mean, jump_std = -0.05, 0.10

    S0_real = hist['Close'].iloc[-1]
    expirations = asset.options
    
    if not expirations:
        raise ValueError(f"No options chain available for {ticker}. Cannot extract implied volatility.")
        
    target_exp = expirations[min(2, len(expirations)-1)] 
    opt_chain = asset.option_chain(target_exp)
    calls = opt_chain.calls

    calls = calls[(calls['strike'] > S0_real * 0.8) & (calls['strike'] < S0_real * 1.2)]
    calls = calls[calls['impliedVolatility'] > 0.01] 

    if calls.empty:
         raise ValueError(f"No liquid options found near-the-money for {ticker}.")

    atm_call = calls.iloc[(calls['strike'] - S0_real).abs().argsort()[:1]]
    v0_real = (atm_call['impliedVolatility'].values[0])**2
    
    return {
        "S0": S0_real,
        "v0": v0_real,
        "realized_var": realized_var,
        "jump_intensity": jump_intensity,
        "jump_mean": jump_mean,
        "jump_std": jump_std
    }
