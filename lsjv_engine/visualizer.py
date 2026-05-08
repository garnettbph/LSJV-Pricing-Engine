import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def plot_empirical_dynamics(ticker: str):
    print(f"Generating Dashboard for {ticker}...")
    
    # 1. Fetch Data
    asset = yf.Ticker(ticker)
    asset_data = asset.history(period="5y")
    
    if asset_data.empty:
        print(f"[WARNING] Could not generate visualization. No historical data for {ticker}.")
        return

    # 2. Process Data 
    asset_data['Log_Returns'] = np.log(asset_data['Close'] / asset_data['Close'].shift(1))
    asset_data['Rolling_Vol'] = asset_data['Log_Returns'].rolling(window=30).std() * np.sqrt(252) * 100
    asset_data.dropna(inplace=True)

    std_dev_vis = asset_data['Log_Returns'].std()
    jumps_vis = asset_data[asset_data['Log_Returns'].abs() > 3 * std_dev_vis]

    # 3. Build Dashboard
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{ticker} Price History", f"Daily Log Returns (>3σ Jumps in Red)", "30-Day Rolling Volatility")
    )

    fig.add_trace(go.Candlestick(x=asset_data.index, open=asset_data['Open'], high=asset_data['High'],
                                 low=asset_data['Low'], close=asset_data['Close'], name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data['Log_Returns'], mode='lines', 
                             name='Log Returns', line=dict(color='gray', width=1), opacity=0.6), row=2, col=1)

    fig.add_trace(go.Scatter(x=jumps_vis.index, y=jumps_vis['Log_Returns'], mode='markers', 
                             name='>3σ Jumps', marker=dict(color='red', size=6)), row=2, col=1)

    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data['Rolling_Vol'], mode='lines', 
                             name='30D Rolling Vol (%)', line=dict(color='purple', width=1.5)), row=3, col=1)

    fig.update_layout(title=f"Empirical Asset Dynamics: {ticker} (Last 5 Years)", title_x=0.5, height=800, 
                      showlegend=False, template="plotly_white", xaxis_rangeslider_visible=False)

    fig.add_hline(y=3*std_dev_vis, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=-3*std_dev_vis, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)

    output_filename = f"{ticker}_dynamics_dashboard.html"
    
    try:
        fig.write_html(output_filename)
        print(f"Visualization saved successfully: {os.path.abspath(output_filename)}")
        print("Open this file in any web browser to view the interactive dashboard.")
    except Exception as e:
        # Fallback to standard show if write fails
        print(f"[WARNING] Could not save HTML file: {e}. Attempting to launch browser instead...")
        fig.show()
