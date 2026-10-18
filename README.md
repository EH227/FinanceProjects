# Equity Volatility & Moving Average Analyzer

A lightweight quantitative data analysis tool built in Python to pull historical equity data, compute technical indicators, and model asset risk.

## Features
* **Data Ingestion:** Programmatically pulls institutional-grade time-series data using `yfinance`.
* **Technical Indicators:** Computes 50-day and 200-day Simple Moving Averages (SMA) using `pandas`.
* **Risk Modeling:** Calculates daily returns and annualized volatility using standard deviation metrics.
* **Visualization:** Generates professional financial charts using `matplotlib`.

## Tech Stack
* Python
* Pandas
* NumPy
* Matplotlib
* yFinance

# Monte Carlo European Option Pricing Simulator

A quantitative derivatives pricing engine built in Python to simulate thousands of asset price trajectories and estimate theoretical option values using stochastic calculus.

## Features
* **Stochastic Modeling:** Simulates thousands of randomized asset price paths over time using Geometric Brownian Motion (GBM) and standard normal noise.
* **Derivatives Pricing:** Computes European Call Option payoffs at maturity and discounts future cash flows back to present value using risk-free interest rates.
* **Visualization:** Generates professional time-series matplotlib charts displaying stochastic price path dispersion and volatility distributions.

## Tech Stack
* Python
* NumPy
* Matplotlib

## How to Run
1. Ensure dependencies are installed:
   ```bash
   pip install numpy matplotlib
   python montecarloeuro.py
