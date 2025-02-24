import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Generate smoothed tradeoff curves
figures = []
for i, emissions_idx in enumerate(emissions_indices):
    for j, volume_idx in enumerate(volume_indices):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Extract data for current combination
        emissions_rand = mean_rand[:, emissions_idx]
        volume_rand = mean_rand[:, volume_idx]
        emissions_pomcpow = mean_pomcpow[:, emissions_idx]
        volume_pomcpow = mean_pomcpow[:, volume_idx]

        # Sort the data for smooth interpolation
        sorted_indices_rand = np.argsort(emissions_rand)
        sorted_indices_pomcpow = np.argsort(emissions_pomcpow)

        emissions_rand_sorted = emissions_rand[sorted_indices_rand]
        volume_rand_sorted = volume_rand[sorted_indices_rand]
        emissions_pomcpow_sorted = emissions_pomcpow[sorted_indices_pomcpow]
        volume_pomcpow_sorted = volume_pomcpow[sorted_indices_pomcpow]

        # Create smooth curves
        x_smooth_rand = np.linspace(emissions_rand_sorted.min(), emissions_rand_sorted.max(), 200)
        x_smooth_pomcpow = np.linspace(emissions_pomcpow_sorted.min(), emissions_pomcpow_sorted.max(), 200)

        spline_rand = make_interp_spline(emissions_rand_sorted, volume_rand_sorted, k=3)
        spline_pomcpow = make_interp_spline(emissions_pomcpow_sorted, volume_pomcpow_sorted, k=3)

        y_smooth_rand = spline_rand(x_smooth_rand)
        y_smooth_pomcpow = spline_pomcpow(x_smooth_pomcpow)

        # Plot smoothed tradeoff curves
        ax.plot(x_smooth_rand, y_smooth_rand, label="Rand", color="tab:blue", linewidth=2)
        ax.plot(x_smooth_pomcpow, y_smooth_pomcpow, label="POMCPOW", color="tab:red", linewidth=2)

        # Scatter original points
        ax.scatter(emissions_rand, volume_rand, color="tab:blue", marker='o', label="Rand Data", s=60)
        ax.scatter(emissions_pomcpow, volume_pomcpow, color="tab:red", marker='s', label="POMCPOW Data", s=60)

        # Labels and formatting
        ax.set_xlabel(f"{emissions_labels[i]} (negative)", fontsize=12)
        ax.set_ylabel(f"{volume_labels[j]}", fontsize=12)
        ax.set_title(f"Smoothed Tradeoff Curve: {emissions_labels[i]} vs. {volume_labels[j]}", fontsize=14)

        # Add scenario labels near points
        for k, scenario in enumerate(scenarios):
            ax.text(emissions_rand[k], volume_rand[k], scenario, fontsize=10, color="tab:blue", ha='right')
            ax.text(emissions_pomcpow[k], volume_pomcpow[k], scenario, fontsize=10, color="tab:red", ha='left')

        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        figures.append(fig)

# Display all figures
for fig in figures:
    fig.show()
