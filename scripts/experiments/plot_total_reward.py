import numpy as np
import matplotlib.pyplot as plt

# Define scenarios
scenarios = [
    "Deterministic, Stochastic",
    "Stochastic, Deterministic",
    "Stochastic, Stochastic",
    "Deterministic, Deterministic"
]

# Define metrics
metrics = [
    "Total Reward", "Discounted Reward", "Total Emissions", 
    "Discounted Emissions", "Total Volume Domestic", "Total Volume International - Imported"
]

# Define mean values for Rand and POMCPOW
mean_rand = np.array([
    [2147.44, 1473.60, -71.60, -54.03, 7.55, 7.55],
    [2186.275, 1508.06, -73.35, -56.26, 7.45, 7.65],
    [2079.72, 1430.75, -63.65, -49.02, 6.65, 7.00],
    [2011.5, 1374.78, -68.75, -52.54, 8.05, 5.80]
])

mean_pomcpow = np.array([
    [2426.29, 1723.92, -75.00, -59.29, 8.70, 6.60],
    [2449.62, 1738.32, -84.60, -66.21, 10.10, 6.75],
    [2545.74, 1813.15, -77.90, -61.22, 8.50, 7.30],
    [2531.31, 1800.07, -78.95, -61.69, 9.45, 6.55]
])

# Define standard errors (SE)
se_rand = np.array([
    [90.73, 71.83, 3.83, 2.98, 0.54, 0.42],
    [96.00, 75.34, 3.38, 2.49, 0.587, 0.539],
    [117.66, 91.39, 2.85, 2.20, 0.34, 0.42],
    [80.03, 61.58, 61.58, 2.00, 0.54, 0.47]
])

se_pomcpow = np.array([
    [50.83, 38.19, 2.36, 1.83, 0.377, 0.59],
    [71.45, 54.15, 3.74, 2.77, 0.69, 0.60],
    [68.98, 52.02, 3.42, 2.56, 0.51, 0.55],
    [45.26, 35.33, 3.20, 2.36, 0.43, 0.38]
])

# Define colors to match the image
colors_rand = ["#f4a261", "#e63946", "#a8dadc", "#457b9d"]  # Orange, Red, Light Blue, Dark Blue
colors_pomcpow = ["#ffcc33", "#ff6666", "#66c2a5", "#228B22"]  # Yellow, Pink, Greenish, Dark Green

# Define hatch patterns for POMCPOW
hatch_patterns = ["//", "xx", "oo", "\\\\"]

# Plot grouped bar chart
x = np.arange(len(metrics))  # X-axis positions
bar_width = 0.15  # Width of each bar

fig, ax = plt.subplots(figsize=(12, 6))

# **First draw POMCPOW bars (semi-transparent)**
for i, scenario in enumerate(scenarios):
    shift = (i - len(scenarios)/2) * bar_width  # Centering correction
    
    ax.bar(x + shift, mean_pomcpow[i], bar_width, 
           yerr=se_pomcpow[i], label=f"POMCPOW {scenario}", hatch=hatch_patterns[i], 
           alpha=0.7, capsize=4, color=colors_pomcpow[i], edgecolor="black")

# **Then draw Rand bars (fully visible)**
for i, scenario in enumerate(scenarios):
    shift = (i - len(scenarios)/2) * bar_width  # Centering correction
    
    ax.bar(x + shift, mean_rand[i], bar_width, yerr=se_rand[i],
           label=f"Rand {scenario}", alpha=.85, capsize=4, color=colors_rand[i], edgecolor="black")

# Labels and formatting
ax.set_xlabel("Metrics", fontsize=12)
ax.set_ylabel("Values", fontsize=12)
ax.set_title("Comparison of Mean Values Across Scenarios (Grouped Bars)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=10)
ax.legend(fontsize=10)
plt.tight_layout()

# Show the plot
plt.show()
