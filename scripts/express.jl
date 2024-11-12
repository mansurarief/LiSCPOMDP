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

rng = MersenneTwister(1)
pomdp = LiPOMDP(p=1.0, ρ=0.6, T=30, w=[0.1, 1.0, 0.1, 0.2])
up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
policy = RandomPolicy(pomdp)
heuristic_policy = AusDomPolicy(pomdp, [1, 2, 10, 11])

#use particle ParticleFilters
num_particles = 1000
bf = BootstrapFilter(pomdp, num_particles, rng)
b0 = ParticleCollection(support(initialize_belief(up)))

solver = POMCPOW.POMCPOWSolver(
     tree_queries=1000, 
     estimate_value = RolloutEstimator(heuristic_policy),
     k_observation=2.0, 
     alpha_observation=0.15, 
     max_depth=30, 
     enable_action_pw=false,
     init_N=24  
 ) 

pomcpow_planner = solve(solver, pomdp)
pomcp_solver = POMCPSolver(tree_queries=1000)
pomcp_planner = solve(pomcp_solver, pomdp)


max_steps=25
hr = HistoryRecorder(max_steps=max_steps)
# @time phist = simulate(hr, pomdp, pomcpow_planner, up, b0);
@time rhist = simulate(hr, pomdp, policy, up, b);
@time hhist = simulate(hr, pomdp, heuristic_policy, up, b);



# println("reward POMCPOW Planner: $(round(discounted_reward(phist), digits=2))")
println("reward Random Planner: $(round(discounted_reward(rhist), digits=2))")
println("reward Heuristic Planner: $(round(discounted_reward(hhist), digits=2))")


df = _get_rewards(pomdp, hhist);
p = _plot_results(pomdp, df);
pall = plot(p.action, p.econ, p.other, layout=(3, 1), size=(1100, 800), margin=5mm)
# savefig(pall, "results.pdf")