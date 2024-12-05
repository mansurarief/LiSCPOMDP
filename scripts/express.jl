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
pomdp = LiPOMDP(
    p=1.0, ρ=0.6, 
    T=31, w=[0.1, 3.2, 0.1, 0.2], 
    a=[-20, -20, -20, -30], 
    mine_rate=[300, 300, 100, 100])
up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
policy = RandomPolicy(pomdp)
heuristic_policy = AusDomPolicy(pomdp, [1, 2, 10, 11])
h0_policy = HeuristicPolicy(
    pomdp, [10, 11, 1, 2], [0, 0, 0, 0], 
    [[0], [0], [0], [0]])
h1_policy = HeuristicPolicy(
    pomdp, [10, 11, 2, 1], [30, 30, 0, 12], 
    [[1, 8], [3, 7, 9], [2], [0]])
h2_policy = HeuristicPolicy(
    pomdp, [10, 11, 2, 1], [30, 0, 13, 12], 
    [[3, 5, 6], [4, 8, 9], [0], [0]])
importOnly_policy = HeuristicPolicy(
    pomdp, [0, 0, 2, 1], [0, 0, 0, 0], 
    [[0], [0], [0], [0]])

h3_policy = HeuristicPolicy(
    pomdp, [10, 11, 2, 1], [30, 30, 13, 12], 
    [[3, 5, 6], [4, 8, 9], [2], [0]])


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


max_steps=30
hr = HistoryRecorder(max_steps=max_steps)
# @time phist = simulate(hr, pomdp, pomcpow_planner, up, b0);
@time rhist = simulate(hr, pomdp, policy, up, b);
@time hhist = simulate(hr, pomdp, heuristic_policy, up, b);
@time h0hist = simulate(hr, pomdp, h0_policy, up, b);
@time h1hist = simulate(hr, pomdp, h1_policy, up, b);
@time h2hist = simulate(hr, pomdp, h2_policy, up, b);
@time h3hist = simulate(hr, pomdp, importOnly_policy, up, b);
@time h3hist = simulate(hr, pomdp, h3_policy, up, b);



# println("reward POMCPOW Planner: $(round(discounted_reward(phist), digits=2))")
println("reward Random Planner: $(round(discounted_reward(rhist), digits=2))")
println("reward Heuristic Planner: $(round(discounted_reward(hhist), digits=2))")
println("reward H0 Planner: $(round(discounted_reward(h0hist), digits=2))")
println("reward H1 Planner: $(round(discounted_reward(h1hist), digits=2))")
println("reward H2 Planner: $(round(discounted_reward(h2hist), digits=2))")
println("reward H3 Planner: $(round(discounted_reward(h3hist), digits=2))")




df = _get_rewards(pomdp, h3hist);
p = _plot_results(pomdp, df)
pall = plot(p.action, p.econ, p.other, layout=(3, 1), size=(1100, 800), margin=5mm)

# savefig(pall, "data/importOnly_result.pdf")
# # savefig(p.action, "data/pomcp_action.pdf")
# # savefig(p.econ, "data/pomcp_econ.pdf")
# # savefig(p.other, "data/pomcp_other.pdf")

# # df.t_restore
# #print the actions from HistoryRecorder


# # scatter(
# #         df.t_mine_off, df.mine_off,
# #         label="MINE INACTIVE", 
# #         markersize=10,
# #         markerstrokewidth=2,        # Normal outline width
# #         markerstrokecolor=:gainsboro,    # Gray outline
# #         markercolor=RGBA(1,1,1,0),   # No fill
# #         markershape=:circle,        # Circle shap
# #     )

# #print the new policy h3hist actions

# println("h3hist actions")
# for i in 1:length(h3hist)
#     println(h3hist[i].a)
# end