import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS & INITIALIZATION
S0 = 36.0  # Current stock price
K = 40.0  # Strike price
T = 1.0  # Time to maturity (1 year)
r = 0.06  # Risk-free interest rate (6%)
sigma = 0.20  # Volatility (20%)

num_simulations = 10000  # Number of price paths
num_steps = 50  # Number of time steps
dt = T / num_steps
discount = np.exp(-r * dt)

# SIMULATE PRICE PATHS (Geometric Brownian Motion)
np.random.seed(42)
z = np.random.standard_normal((num_steps, num_simulations))
price_paths = np.zeros((num_steps + 1, num_simulations))
price_paths[0] = S0

for t in range(1, num_steps + 1):
  drift = (r - 0.5 * sigma**2) * dt
  diffusion = sigma * np.sqrt(dt) * z[t - 1]
  price_paths[t] = price_paths[t - 1] * np.exp(drift + diffusion)

# LEAST-SQUARES MONTE CARLO (Backward Induction)
# Initialize cash flows at maturity for an American Put: Max(K - S_final, 0)
cash_flows = np.maximum(K - price_paths[-1], 0)

# Work backward from time step T-1 down to 1
for t in range(num_steps - 1, 0, -1):
  # Identify paths that are in-the-money at this specific step
  spot_t = price_paths[t]
  itm = K - spot_t > 0  # Only consider ITM puts

  if np.sum(itm) > 0:
    # X = stock prices of ITM paths, Y = discounted future cash flows
    X = spot_t[itm]
    Y = cash_flows[itm] * discount

    # Perform polynomial regression (degree 2) to estimate continuation value
    regression_coeffs = np.polyfit(X, Y, 2)
    continuation_value = np.polyval(regression_coeffs, X)

    # Immediate exercise value
    exercise_value = K - X

    # Check where early exercise is more valuable than holding
    exercise_early = exercise_value > continuation_value

    # Update cash flows for paths that exercise early
    # First, map back to the full simulation array size
    full_exercise_mask = np.zeros(num_simulations, dtype=bool)
    full_exercise_mask[itm] = exercise_early

    # Update cash flows and zero out subsequent cash flows for early exercises
    cash_flows[full_exercise_mask] = exercise_value[exercise_early]
    # Discount non-exercised paths normally
    cash_flows[~full_exercise_mask] = (
        cash_flows[~full_exercise_mask] * discount
    )
  else:
    cash_flows = cash_flows * discount

# Final present value calculation averaged across all paths
american_put_price = np.mean(cash_flows * discount)

print(
    f"Estimated American Put Option Price via Longstaff-Schwartz LSM:"
    f" ${american_put_price:.2f}"
)

# After running your simulation, you can plot a subset of paths
# along with the strike price boundary
plt.figure(figsize=(12, 6))
plt.plot(price_paths[:, :50], alpha=0.3)  # Plot first 50 paths
plt.axhline(y=K, color="red", linestyle="--", label=f"Strike Price (${K})")

plt.title(
    "Least-Squares Monte Carlo: American Put Price Paths & Exercise Region",
    fontsize=14,
)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Stock Price ($)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()