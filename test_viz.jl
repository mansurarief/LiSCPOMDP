using Random
using POMDPs 
using POMDPTools
using POMDPPolicies
using LiPOMDPs
using MCTS
using DiscreteValueIteration
using POMCPOW 
using Distributions
using Parameters
using ARDESPOT
using Plots
using Plots.PlotMeasures

rng = MersenneTwister(1)

pomdp = initialize_lipomdp(obj_weights=[0.25, 0.25, 1.0, 1.0, 0.25]) 
up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
mdp = GenerativeBeliefMDP(pomdp, up)


random_planner = RandPolicy(pomdp)
strong_planner = EfficiencyPolicy(pomdp, [true, true, true, true])
robust_planner = EfficiencyPolicyWithUncertainty(pomdp, 1., [true, true, true, true])
eco_planner = EmissionAwarePolicy(pomdp, [true, true, true, true])
# mcts_solver = DPWSolver(
#      depth=15,
#      n_iterations = 1000,
#      estimate_value=RolloutEstimator(robust_planner, max_depth=50),
#      enable_action_pw=false,
#      enable_state_pw=true,
#      k_state = 4.,
#     alpha_state = 0.1,
#  )
# mcts_planner = solve(mcts_solver, mdp)

solver = POMCPOW.POMCPOWSolver(
     tree_queries=1000, 
     estimate_value = 0, 
     k_observation=4., 
     alpha_observation=0.06, 
     max_depth=15, 
     enable_action_pw=false,
     init_N=10  
 ) 
pomcpow_planner = solve(solver, pomdp)

hr = HistoryRecorder(rng=rng, max_steps=pomdp.time_horizon)
hist = simulate(hr, pomdp, pomcpow_planner, up, b);

df = get_rewards(pomdp, hist);
p = plot_results(pomdp, df);
pall = plot(p.action, p.econ, p.other, layout=(3, 1), size=(1100, 800), margin=5mm);
savefig(pall, "results.pdf")
