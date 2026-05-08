import argparse
import torch
import time
import sys
from lsjv_engine.data_pipeline import fetch_market_parameters
from lsjv_engine.core_math import LSJVEngine
from lsjv_engine.visualizer import plot_empirical_dynamics

def main():
    # Setup CLI Argument Parser
    parser = argparse.ArgumentParser(description="LSJV Options Pricing Engine")
    parser.add_argument("--ticker", type=str, default="SPY", help="Equity ticker symbol (e.g., AAPL, TSLA, NVDA)")
    parser.add_argument("--paths", type=int, default=50000, help="Number of Monte Carlo paths")
    parser.add_argument("--steps", type=int, default=252, help="Number of time steps (default 252 days)")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    print(f"--- Starting LSJV Pricing Pipeline for {ticker} ---")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Engine: {device}")

    # 2. Fetch Live Data
    try:
        mkt_data = fetch_market_parameters(ticker)
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch data for {ticker}. Ensure the ticker is valid and has active options.")
        print(f"Details: {e}")
        sys.exit(1)
    
    print(f"Current Spot Price (S0): ${mkt_data['S0']:.2f}")
    print(f"Current ATM Variance (v0): {mkt_data['v0']:.4f}")

    # 3. Configure Engine Parameters
    rates = {'r': 0.045, 'q': 0.015} 
    
    heston_params = {'kappa': 3.0, 'theta': mkt_data['realized_var'], 'xi': 0.6, 'rho': -0.75}
    jump_params = {'lambda': mkt_data['jump_intensity'], 'mu_J': mkt_data['jump_mean'], 'sig_J': mkt_data['jump_std']} 
    
    sim_params = {'T': 1.0, 'steps': args.steps, 'paths': args.paths}

    # 4. Initialize and Run Engine
    engine = LSJVEngine(S0=mkt_data['S0'], v0=mkt_data['v0'], 
                        rates=rates, heston_params=heston_params, 
                        jump_params=jump_params, sim_params=sim_params, 
                        device=device)

    print(f"\nExecuting LSJV Particle Calibration ({args.paths} paths)...")
    start_time = time.time()
    paths = engine.calibrate_and_simulate()
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")

    # 5. Price Option and Calculate AAD Greeks
    K_real = mkt_data['S0'] # Price an At-The-Money Call
    terminal_S = paths[-1]
    payoff = torch.clamp(terminal_S - K_real, min=0.0)
    price = torch.exp(torch.tensor(-rates['r'] * sim_params['T'], device=device)) * torch.mean(payoff)

    print("\nCalculating Greeks via Backpropagation (AAD)...")
    price.backward()

    print(f"\n====================================")
    print(f"{ticker} 1-Year ATM Call Option Price: ${price.item():.2f}")
    print(f"Pathwise Delta (∂V/∂S0):      {engine.S0.grad.item():.4f}")
    print(f"Pathwise Vega  (∂V/∂v0):      {engine.v0.grad.item():.4f}")
    print(f"====================================\n")

    # 6. Show Visualization
    plot_empirical_dynamics(ticker)

if __name__ == "__main__":
    main()
