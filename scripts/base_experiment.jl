using Random
using POMDPs 
using POMDPTools
# using POMDPPolicies
using LiPOMDPs
using MCTS
# using DiscreteValueIteration
using POMCPOW 
# using BasicPOMCP
using Distributions
using Parameters
using ARDESPOT
# using Iterators
using ParticleFilters
using Plots
using Plots.PlotMeasures
using BasicPOMCP


# include("../src/LiPOMDPs.jl")

rng = MersenneTwister(2024) 
pomdp = LiPOMDP(
    p=1.0, ρ=0.6, 
    T=31, w=[0.1, 3.2, 0.1, 0.2], 
    a=[-20, -20, -20, -30], 
    mine_rate=[300, 300, 100, 100])
up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
policy = RandomPolicy(pomdp)

max_steps=30
hr = HistoryRecorder(max_steps=max_steps)
# @time phist = simulate(hr, pomdp, pomcpow_planner, up, b0);
@time rhist = simulate(hr, pomdp, policy, up, b);
println("reward Random Planner: $(round(discounted_reward(rhist), digits=2))")

#print all actions 
df = _get_rewards(pomdp, rhist);
p = _plot_results(pomdp, df);
pall = plot(p.action, p.econ, p.other, layout=(3, 1), size=(1100, 800), margin=5mm)

