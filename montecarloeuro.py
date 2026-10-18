import matplotlib.pyplot as plt
import numpy as np

# Option and Market Parameters
S0 = 100.0  # Current stock price ($100)
K = 100.0  # Strike price ($100 - At-the-money)
T = 1.0  # Time to maturity (1 year)
r = 0.05  # Risk-free interest rate (5%)
sigma = 0.20  # Volatility (20%)

# Simulation parameters
num_simulations = 10000  # Number of random price paths to simulate
num_steps = 252  # Number of time steps (trading days in a year)
dt = T / num_steps  # Size of each time step

# Simulate Stock Price Paths using Geometric Brownian Motion
# Initialize a matrix to store prices
price_paths = np.zeros((num_steps + 1, num_simulations))
price_paths[0] = S0

# Generate random normal noise for all steps and simulations at once
np.random.seed(42)  # Set seed for reproducible results
z = np.random.standard_normal((num_steps, num_simulations))

# Step through time using the GBM formula
for t in range(1, num_steps + 1):
  # Drift component + Diffusion (random noise) component
  drift = (r - 0.5 * sigma**2) * dt
  diffusion = sigma * np.sqrt(dt) * z[t - 1]
  price_paths[t] = price_paths[t - 1] * np.exp(drift + diffusion)

# Calculate Payoffs at Expiration (T)
# For a Call option: Max(S_final - Strike, 0)
final_prices = price_paths[-1]
payoffs = np.maximum(final_prices - K, 0)

# Discount the average payoff back to present value using the risk-free rate
option_price = np.exp(-r * T) * np.mean(payoffs)

print(
    f"Estimated European Call Option Price via Monte Carlo: ${option_price:.2f}"
)

# Plot a subset of the simulated price paths
plt.figure(figsize=(12, 6))
plt.plot(price_paths[:, :100], alpha=0.4)  # Plot first 100 paths
plt.title(
    f"Monte Carlo Simulation of Stock Price Paths ({num_simulations} Simulations)",
    fontsize=14,
)
plt.xlabel("Time Steps (Days)", fontsize=12)
plt.ylabel("Stock Price ($)", fontsize=12)
plt.grid(True)
plt.show()