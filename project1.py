#pip install yfinance pandas matplotlib to install those first

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
# 1. Define the ticker symbol and date range
ticker = "AAPL"
start_date = "2024-01-01"
end_date = "2026-01-01"
# 2. Pull historical data from Yahoo Finance
print(f"Downloading data for {ticker}...")
data = yf.download(ticker, start=start_date, end=end_date)
# 3. Inspect the first few rows of the data
print(data.head())
# 4. Extract the 'Close' price column for calculations
close_prices = data[('Close', 'AAPL')]

# 5. Calculate 50-day and 200-day Simple Moving Averages (SMA)
data['SMA_50'] = close_prices.rolling(window=50).mean()
data['SMA_200'] = close_prices.rolling(window=200).mean()

# 6. Calculate Daily Returns and Annualized Volatility
data['Daily_Return'] = close_prices.pct_change()
# Volatility = Standard deviation of returns multiplied by the square root of trading days in a year (252)
annualized_volatility = data['Daily_Return'].std() * (252 ** 0.5)

print(f"\nAnnualized Volatility for {ticker}: {annualized_volatility:.4f}")
print("\nUpdated Data with Moving Averages:")
print(data[['Close', 'SMA_50', 'SMA_200']].tail())
# 7. Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(data.index, close_prices, label=f"{ticker} Close Price", color="black", alpha=0.6)
plt.plot(data.index, data['SMA_50'], label="50-Day SMA", color="blue", linestyle="--")
plt.plot(data.index, data['SMA_200'], label="200-Day SMA", color="red", linestyle="--")

plt.title(f"{ticker} Quantitative Price & Moving Average Analysis", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend()
plt.grid(True)

# Display the chart window
plt.show()