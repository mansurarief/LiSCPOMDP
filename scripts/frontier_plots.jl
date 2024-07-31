using Plots

random_policy = (-26.1, 5.2)
efficiency_policy = (-53.95, 10.9)
efficiency_policy_wuncertainty = (-50.55, 11.3)
emission_aware_policy = (-47.05, 10.9)

POMCPOW_points = [(-0.1, 0.05), (-1.6, 0.6), (-5.75, 2.5), (-12.45, 4.9), (-19.9, 5.9), (-54.5, 11.5), (-58.65, 13.05), (-58.6, 12.65), (-60.2, 12.85), (-63.7, 13.5)]
MCTS_points = [(-21.6, 6.3), (-26.5, 6.8), (-30.0, 6.8), (-36.0, 9.3), (-42.65, 9.55), (-33.35, 7.8), (-39.6, 8.85), (-35.7, 8.6), (-42.45, 8.65), (-39.8, 8.5)]



f(x::Float64) = 4 - x^2
points = [(1, 1), (2, 2), (3,3)]

p = scatter(random_policy, label="Random Policy")
scatter!(efficiency_policy, label="Efficiency Policy")
scatter!(efficiency_policy_wuncertainty, label="Efficiency Policy (with uncertainty)")
scatter!(emission_aware_policy, label="Emission Aware Policy")
scatter!(POMCPOW_points, label = "POMCPOW with varying objective weights")
scatter!(MCTS_points, label="MCTS with varying objective weights")

# Create a scatter plot
# scatter(
#     [random_policy[1], efficiency_policy[1], efficiency_policy_wuncertainty[1], emission_aware_policy[1]],
#     [random_policy[2], efficiency_policy[2], efficiency_policy_wuncertainty[2], emission_aware_policy[2]],
#     label=["Random Policy", "Efficiency Policy", "Efficiency Policy w/Uncertainty", "Emission Aware Policy"],
#     color=["red", "green", "blue", "purple"],
#     legend=:topleft,
# )

# scatter!(
#     [point[1] for point in POMCPOW_points],
#     [point[2] for point in POMCPOW_points],
#     label="POMCPOW",
#     color="orange",
#     xlims=(-70, 10),  # Set x-axis limits
#     ylims=(-5, 15),   # Set y-axis limits
# )

# scatter!(
#     [point[1] for point in MCTS_points],
#     [point[2] for point in MCTS_points],
#     label="MCTS",
#     color="cyan",
#     xlims=(-70, 10),  # Set x-axis limits
#     ylims=(-5, 15),   # Set y-axis limits
# )

xlabel!("Negative Total Emissions")
ylabel!("Total Volume")

# Show the plot
plot(p)
